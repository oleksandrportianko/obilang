"""Auditable dataset normalization, filtering, versioning, and splitting."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmt.config.schema import DataConfig, PlatformConfig
from nmt.data.readers import discover_inputs, read_all
from nmt.data.records import ParallelRecord, RejectedRecord, ValidationReport
from nmt.utils.io import atomic_write_bytes, atomic_write_json
from nmt.utils.paths import ProjectPaths

LOGGER = logging.getLogger(__name__)
WHITESPACE = re.compile(r"\s+")
TAG = re.compile(r"</?([A-Za-z][\w:.-]*)(?:\s[^<>]*?)?/?>")
TAG_LIKE = re.compile(r"<[^>]*>|[<>]")
NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
SCRIPT_PREFIXES = {
    "Latn": "LATIN",
    "Cyrl": "CYRILLIC",
    "Grek": "GREEK",
    "Arab": "ARABIC",
    "Hebr": "HEBREW",
    "Deva": "DEVANAGARI",
}


@dataclass(frozen=True)
class PreparedDataset:
    """Paths and counters emitted by a deterministic dataset build."""

    version: str
    processed_path: Path
    split_directory: Path
    report_path: Path
    manifest_path: Path
    report: dict[str, Any]


def normalize_text(value: str, normalization: str = "NFC") -> str:
    """Normalize Unicode and horizontal/vertical whitespace without changing punctuation.

    Args:
        value: Raw sentence from an input reader.
        normalization: A Unicode normalization form supported by ``unicodedata``.

    Returns:
        Trimmed text with every whitespace run replaced by one ASCII space.
    """
    return WHITESPACE.sub(" ", unicodedata.normalize(normalization, value)).strip()


def _markup_is_balanced(text: str) -> bool:
    """Conservatively validate XML-like opening and closing tag order."""
    if not TAG_LIKE.search(text):
        return True
    stack: list[str] = []
    covered = [False] * len(text)
    for match in TAG.finditer(text):
        for index in range(match.start(), match.end()):
            covered[index] = True
        raw = match.group(0)
        name = match.group(1)
        if raw.startswith("</"):
            if not stack or stack.pop() != name:
                return False
        elif not raw.rstrip().endswith("/>"):
            stack.append(name)
    # Any angle bracket not belonging to a syntactically recognized tag is malformed.
    stray_angle = any(char in "<>" and not covered[index] for index, char in enumerate(text))
    return not stack and not stray_angle


def _dominant_script(text: str) -> str | None:
    """Return the most frequent configured Unicode script among alphabetic characters."""
    counts: Counter[str] = Counter()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script, prefix in SCRIPT_PREFIXES.items():
            if prefix in name:
                counts[script] += 1
                break
    return counts.most_common(1)[0][0] if counts else None


def _near_duplicate_signature(record: ParallelRecord) -> str:
    """Group case, punctuation, and whitespace variants into the same split."""
    normalized_source = NON_WORD.sub("", record.source.casefold())
    return normalized_source or record.source.casefold()


def _row_reasons(
    record: ParallelRecord,
    data_config: DataConfig,
    seen_pairs: set[tuple[str, str]],
    source_targets: dict[str, str],
    target_sources: dict[str, str],
) -> tuple[str, ...]:
    """Return all validation failures for one already-normalized row."""
    reasons: list[str] = []
    source, target = record.source, record.target
    if not source or not target:
        reasons.append("empty")
        reasons.append("missing_value")
        return tuple(dict.fromkeys(reasons))
    pair = (source, target)
    if pair in seen_pairs:
        reasons.append("duplicate_pair")
    source_previous = source_targets.get(source)
    if source_previous is not None:
        reasons.append("source_duplicate")
        if source_previous != target:
            reasons.append("conflicting_translation")
    target_previous = target_sources.get(target)
    if target_previous is not None:
        reasons.append("target_duplicate")
        if target_previous != source:
            reasons.append("conflicting_translation")
    source_length, target_length = len(source), len(target)
    if (
        source_length < data_config.minimum_characters
        or target_length < data_config.minimum_characters
        or source_length > data_config.maximum_characters
        or target_length > data_config.maximum_characters
        or len(source.split()) > data_config.maximum_tokens_approximation
        or len(target.split()) > data_config.maximum_tokens_approximation
    ):
        reasons.append("length")
    ratio = max(source_length, target_length) / max(1, min(source_length, target_length))
    if ratio > data_config.maximum_length_ratio:
        reasons.append("length_ratio")
    if data_config.reject_malformed_markup and (
        not _markup_is_balanced(source) or not _markup_is_balanced(target)
    ):
        reasons.append("malformed_markup")
    if data_config.detect_language_mismatch:
        source_script, target_script = _dominant_script(source), _dominant_script(target)
        if (
            data_config.source_scripts
            and source_script
            and source_script not in data_config.source_scripts
        ) or (
            data_config.target_scripts
            and target_script
            and target_script not in data_config.target_scripts
        ):
            reasons.append("language_mismatch")
    return tuple(dict.fromkeys(reasons))


def validate_records(
    records: list[ParallelRecord], data_config: DataConfig
) -> tuple[list[ParallelRecord], list[RejectedRecord], ValidationReport]:
    """Normalize and audit raw examples without silently dropping any row.

    Args:
        records: Materialized reader output with input provenance.
        data_config: Validated filtering and normalization settings.

    Returns:
        Accepted normalized rows, rejected rows with reasons, and aggregate report.
    """
    report = ValidationReport(total_rows=len(records))
    accepted: list[ParallelRecord] = []
    rejected: list[RejectedRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    source_targets: dict[str, str] = {}
    target_sources: dict[str, str] = {}
    for raw in records:
        normalized = ParallelRecord(
            normalize_text(raw.source, data_config.unicode_normalization),
            normalize_text(raw.target, data_config.unicode_normalization),
            raw.origin,
            raw.row_number,
            raw.domain,
            raw.metadata,
        )
        reasons = _row_reasons(
            normalized, data_config, seen_pairs, source_targets, target_sources
        )
        if reasons:
            rejected.append(
                RejectedRecord(
                    normalized.source,
                    normalized.target,
                    normalized.origin,
                    normalized.row_number,
                    reasons,
                )
            )
            report.reject(reasons)
            continue
        accepted.append(normalized)
        seen_pairs.add((normalized.source, normalized.target))
        source_targets[normalized.source] = normalized.target
        target_sources[normalized.target] = normalized.source
    report.accepted_rows = len(accepted)
    return accepted, rejected, report


def _file_hash(path: Path) -> str:
    """Calculate SHA-256 by chunks so large corpora do not enter memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(inputs: list[Path], dataset_root: Path, data_config: DataConfig) -> tuple[str, dict[str, str]]:
    """Fingerprint ordered source bytes and all preprocessing/split settings."""
    hashes = {str(path.relative_to(dataset_root)): _file_hash(path) for path in sorted(inputs)}
    payload = {
        "files": hashes,
        "data_config": data_config.model_dump(mode="json"),
        "fingerprint_schema": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], hashes


def split_records(
    records: list[ParallelRecord], data_config: DataConfig
) -> dict[str, list[ParallelRecord]]:
    """Assign records to stable hash splits while grouping near-duplicate sources.

    Args:
        records: Unique, accepted examples.
        data_config: Split fractions and seed.

    Returns:
        Mapping with `train`, `validation`, and `test` lists. Assignment depends
        only on the seeded near-duplicate signature, never row order or corpus size.
    """
    result: dict[str, list[ParallelRecord]] = {"train": [], "validation": [], "test": []}
    train_limit = data_config.split.train
    validation_limit = train_limit + data_config.split.validation
    for record in records:
        signature = f"{data_config.split.seed}:{_near_duplicate_signature(record)}"
        bucket = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16], 16) / 2**64
        split = "train" if bucket < train_limit else "validation" if bucket < validation_limit else "test"
        result[split].append(record)
    return result


def _jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    """Serialize a deterministic sequence of objects to JSON Lines bytes."""
    return b"".join(
        (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for value in values
    )


def validate_dataset(config: PlatformConfig, paths: ProjectPaths) -> dict[str, Any]:
    """Audit current inputs and return a report without writing processed artifacts.

    Raises:
        FileNotFoundError: If the pair contains no supported input.
        DatasetFormatError: From strict readers for malformed inputs.
    """
    dataset_root = paths.dataset(config.language_pair.id)
    inputs = discover_inputs(dataset_root)
    records = list(read_all(dataset_root, inputs))
    _, _, report = validate_records(records, config.data)
    report.input_files = [str(path.relative_to(dataset_root)) for path in inputs]
    return report.as_dict()


def prepare_dataset(config: PlatformConfig, paths: ProjectPaths) -> PreparedDataset:
    """Validate, version, persist, and deterministically split a pair dataset.

    Side effects:
        Writes immutable versioned JSONL, rejected rows, report, split files,
        reproducibility manifest, and the pair's `metadata/current.json` pointer.
    """
    dataset_root = paths.dataset(config.language_pair.id)
    inputs = discover_inputs(dataset_root)
    version, input_hashes = dataset_fingerprint(inputs, dataset_root, config.data)
    records = list(read_all(dataset_root, inputs))
    accepted, rejected, report = validate_records(records, config.data)
    report.input_files = [str(path.relative_to(dataset_root)) for path in inputs]
    splits = split_records(accepted, config.data)
    report_document = report.as_dict() | {
        "dataset_version": version,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
    }
    processed_directory = dataset_root / "processed" / version
    split_directory = dataset_root / "splits" / version
    rejected_path = dataset_root / "rejected" / f"{version}.jsonl"
    processed_path = processed_directory / "pairs.jsonl"
    report_path = dataset_root / "metadata" / f"report-{version}.json"
    manifest_path = processed_directory / "manifest.json"
    manifest = {
        "dataset_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language_pair": config.language_pair.model_dump(mode="json"),
        "input_sha256": input_hashes,
        "data_config": config.data.model_dump(mode="json"),
        "artifacts": {
            "processed": str(processed_path.relative_to(dataset_root)),
            "rejected": str(rejected_path.relative_to(dataset_root)),
            "splits": str(split_directory.relative_to(dataset_root)),
            "report": str(report_path.relative_to(dataset_root)),
        },
        "report": report_document,
    }
    atomic_write_bytes(processed_path, _jsonl_bytes([row.as_dict() for row in accepted]))
    atomic_write_bytes(rejected_path, _jsonl_bytes([row.as_dict() for row in rejected]))
    for split_name, split_rows in splits.items():
        atomic_write_bytes(
            split_directory / f"{split_name}.jsonl",
            _jsonl_bytes([row.as_dict() for row in split_rows]),
        )
    atomic_write_json(report_path, report_document)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        dataset_root / "metadata" / "current.json",
        {"dataset_version": version, "manifest": str(manifest_path.relative_to(dataset_root))},
    )
    LOGGER.info(
        "Prepared dataset %s with %d accepted and %d rejected rows",
        version,
        len(accepted),
        len(rejected),
        extra={"context": report_document},
    )
    return PreparedDataset(
        version, processed_path, split_directory, report_path, manifest_path, report_document
    )


def load_parallel_jsonl(path: Path, reverse: bool = False) -> list[ParallelRecord]:
    """Load platform-produced JSONL, optionally swapping translation direction.

    Args:
        path: Trusted processed or split JSONL artifact.
        reverse: Swap source and target text for the reverse directional model.

    Returns:
        Materialized records suitable for tokenization or evaluation.
    """
    records: list[ParallelRecord] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                value = json.loads(line)
                source, target = str(value["source"]), str(value["target"])
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Corrupted processed dataset {path}, row {line_number}: {exc}") from exc
            records.append(
                ParallelRecord(
                    target if reverse else source,
                    source if reverse else target,
                    str(value.get("origin", path.name)),
                    int(value.get("row_number", line_number)),
                    value.get("domain"),
                    value.get("metadata") or {},
                )
            )
    return records
