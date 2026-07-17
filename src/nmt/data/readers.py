"""Strict UTF-8 readers for aligned text, TSV, CSV, and JSONL corpora."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nmt.data.records import ParallelRecord


class DatasetFormatError(ValueError):
    """Raised when a supported data file is encoded or structured incorrectly."""


class DatasetAlignmentError(DatasetFormatError):
    """Raised when parallel text files contain different row counts."""


def _read_lines_strict(path: Path) -> list[str]:
    """Read text as UTF-8 while turning decode failures into actionable errors."""
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            f"Unsupported encoding in {path} at byte {exc.start}. Convert the file to UTF-8."
        ) from exc


def read_parallel_text(source_path: Path, target_path: Path, root: Path) -> Iterator[ParallelRecord]:
    """Yield line-aligned examples from `source.txt` and `target.txt`.

    Args:
        source_path: UTF-8 source-language text file.
        target_path: UTF-8 target-language text file.
        root: Dataset root used to record portable relative provenance.

    Yields:
        ``ParallelRecord`` values with one-based line numbers.

    Raises:
        DatasetAlignmentError: If line counts differ.
        DatasetFormatError: If either file is not valid UTF-8.
    """
    source_lines = _read_lines_strict(source_path)
    target_lines = _read_lines_strict(target_path)
    if len(source_lines) != len(target_lines):
        raise DatasetAlignmentError(
            f"Dataset alignment failed: {source_path.name} contains {len(source_lines):,} lines "
            f"while {target_path.name} contains {len(target_lines):,} lines. The files must "
            "contain the same number of rows."
        )
    origin = f"{source_path.relative_to(root)} + {target_path.relative_to(root)}"
    for row_number, (source, target) in enumerate(zip(source_lines, target_lines), start=1):
        yield ParallelRecord(source, target, origin, row_number)


def read_tsv(path: Path, root: Path) -> Iterator[ParallelRecord]:
    """Yield examples from a two-or-more-column UTF-8 TSV file.

    The first two columns are source and target. An optional third column is a
    domain label. A `source_text<TAB>target_text` header is recognized.
    """
    lines = _read_lines_strict(path)
    reader = csv.reader(lines, delimiter="\t")
    for row_number, row in enumerate(reader, start=1):
        if row_number == 1 and len(row) >= 2 and row[0] in {"source", "source_text"}:
            continue
        if len(row) < 2:
            yield ParallelRecord(row[0] if row else "", "", str(path.relative_to(root)), row_number)
            continue
        yield ParallelRecord(
            row[0], row[1], str(path.relative_to(root)), row_number, row[2] or None if len(row) > 2 else None
        )


def read_csv(path: Path, root: Path) -> Iterator[ParallelRecord]:
    """Yield records from CSV fields `source_text` and `target_text`.

    Raises:
        DatasetFormatError: If required headers are absent or UTF-8 decoding fails.
    """
    lines = _read_lines_strict(path)
    reader = csv.DictReader(lines)
    if not reader.fieldnames or not {"source_text", "target_text"}.issubset(reader.fieldnames):
        raise DatasetFormatError(
            f"CSV {path} must contain source_text and target_text headers; found {reader.fieldnames}."
        )
    for row_number, row in enumerate(reader, start=2):
        yield ParallelRecord(
            row.get("source_text") or "",
            row.get("target_text") or "",
            str(path.relative_to(root)),
            row_number,
            row.get("domain") or None,
            {key: value for key, value in row.items() if key not in {"source_text", "target_text", "domain"}},
        )


def read_jsonl(path: Path, root: Path) -> Iterator[ParallelRecord]:
    """Yield records from JSON objects containing source and target strings.

    Raises:
        DatasetFormatError: If UTF-8, JSON syntax, or top-level value is invalid.
    """
    for row_number, line in enumerate(_read_lines_strict(path), start=1):
        try:
            value: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"Malformed JSONL in {path}, row {row_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetFormatError(f"JSONL {path}, row {row_number} must be an object.")
        source = value.get("source", value.get("source_text", ""))
        target = value.get("target", value.get("target_text", ""))
        metadata = {k: v for k, v in value.items() if k not in {"source", "target", "source_text", "target_text", "domain"}}
        yield ParallelRecord(
            source if isinstance(source, str) else "",
            target if isinstance(target, str) else "",
            str(path.relative_to(root)),
            row_number,
            value.get("domain") if isinstance(value.get("domain"), str) else None,
            metadata,
        )


def discover_inputs(dataset_root: Path, include_incoming: bool = True) -> list[Path]:
    """Return ordered supported input files under raw and optionally incoming.

    Raises:
        FileNotFoundError: If no supported dataset source is present.
        DatasetFormatError: If only one aligned-text file exists.
    """
    directories = [dataset_root / "raw"]
    if include_incoming:
        directories.append(dataset_root / "incoming")
    inputs: list[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        source_path, target_path = directory / "source.txt", directory / "target.txt"
        if source_path.exists() != target_path.exists():
            raise DatasetFormatError(
                f"{directory} must contain both source.txt and target.txt, not just one."
            )
        if source_path.exists():
            inputs.extend([source_path, target_path])
        inputs.extend(sorted(path for path in directory.glob("*.tsv") if path.is_file()))
        inputs.extend(sorted(path for path in directory.glob("*.csv") if path.is_file()))
        inputs.extend(sorted(path for path in directory.glob("*.jsonl") if path.is_file()))
    if not inputs:
        raise FileNotFoundError(
            f"No supported data found in {dataset_root / 'raw'} or {dataset_root / 'incoming'}. "
            "Add source.txt + target.txt, TSV, CSV, or JSONL files."
        )
    return inputs


def read_all(dataset_root: Path, inputs: list[Path]) -> Iterator[ParallelRecord]:
    """Dispatch discovered files to readers without reading aligned files twice."""
    consumed: set[Path] = set()
    for path in inputs:
        if path in consumed:
            continue
        if path.name == "source.txt":
            target_path = path.with_name("target.txt")
            consumed.update({path, target_path})
            yield from read_parallel_text(path, target_path, dataset_root)
        elif path.name == "target.txt":
            continue
        elif path.suffix.lower() == ".tsv":
            yield from read_tsv(path, dataset_root)
        elif path.suffix.lower() == ".csv":
            yield from read_csv(path, dataset_root)
        elif path.suffix.lower() == ".jsonl":
            yield from read_jsonl(path, dataset_root)
