"""SentencePiece BPE/Unigram lifecycle using only versioned project data."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm

from nmt.config.schema import PlatformConfig, TokenizerConfig
from nmt.data.pipeline import load_parallel_jsonl
from nmt.utils.io import atomic_write_json, load_json
from nmt.utils.paths import ProjectPaths

LOGGER = logging.getLogger(__name__)


class TokenizerCompatibilityError(ValueError):
    """Raised when weights and tokenizer identity or vocabulary dimensions differ."""


def _sha256(path: Path) -> str:
    """Return a complete SHA-256 digest for a tokenizer artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_version(dataset_version: str, config: TokenizerConfig) -> str:
    """Derive an immutable tokenizer ID from its corpus and trainer settings."""
    payload = {
        "dataset_version": dataset_version,
        "config": config.model_dump(mode="json"),
        "trainer_schema": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class SentencePieceTokenizer:
    """Small typed wrapper around one immutable SentencePiece model."""

    path: Path
    processor: spm.SentencePieceProcessor

    @classmethod
    def load(cls, path: Path) -> "SentencePieceTokenizer":
        """Load a local tokenizer model.

        Args:
            path: Trusted SentencePiece `.model` created by this platform.

        Returns:
            A wrapper ready for thread-safe encode/decode calls.

        Raises:
            FileNotFoundError: If the model is missing.
            RuntimeError: If SentencePiece cannot parse it.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer model does not exist: {path}")
        processor = spm.SentencePieceProcessor(model_file=str(path))
        return cls(path, processor)

    @property
    def vocabulary_size(self) -> int:
        """Return the fixed number of token IDs represented by the model."""
        return self.processor.vocab_size()

    @property
    def pad_id(self) -> int:
        """Return the padding token ID used for batch shapes and ignored loss."""
        return self.processor.pad_id()

    @property
    def unknown_id(self) -> int:
        """Return the fallback ID for characters absent from the vocabulary."""
        return self.processor.unk_id()

    @property
    def bos_id(self) -> int:
        """Return the autoregressive beginning-of-sentence ID."""
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        """Return the end-of-sentence ID that terminates generation."""
        return self.processor.eos_id()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """Encode one sentence into stable token IDs.

        Args:
            text: Unicode sentence.
            add_bos: Prefix the configured BOS token.
            add_eos: Suffix the configured EOS token.

        Returns:
            Token IDs including requested boundary tokens.
        """
        ids = list(self.processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def pieces(self, text: str) -> list[str]:
        """Expose SentencePiece text pieces for UI token inspection."""
        return list(self.processor.encode(text, out_type=str))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs while ignoring explicit PAD/BOS/EOS boundaries."""
        ignored = {self.pad_id, self.bos_id, self.eos_id}
        return self.processor.decode([token_id for token_id in token_ids if token_id not in ignored])


@dataclass(frozen=True)
class TokenizerBundle:
    """Source/target tokenizer references and compatibility metadata."""

    version: str
    dataset_version: str
    source: SentencePieceTokenizer
    target: SentencePieceTokenizer
    shared: bool
    manifest_path: Path

    @classmethod
    def load(
        cls, dataset_root: Path, version: str | None = None
    ) -> "TokenizerBundle":
        """Load a tokenizer bundle by version or the current pointer.

        Raises:
            FileNotFoundError: If no tokenizer has been trained or an artifact is absent.
            TokenizerCompatibilityError: If its manifest is incomplete or inconsistent.
        """
        if version is None:
            current = load_json(dataset_root / "tokenizer" / "current.json")
            if not current:
                raise FileNotFoundError(
                    f"No tokenizer is registered for {dataset_root.name}. Run `nmt tokenizer train`."
                )
            version = str(current["tokenizer_version"])
        directory = dataset_root / "tokenizer" / version
        manifest_path = directory / "manifest.json"
        manifest = load_json(manifest_path)
        if not manifest:
            raise FileNotFoundError(f"Tokenizer manifest does not exist: {manifest_path}")
        shared = bool(manifest["config"]["shared"])
        source_name = str(manifest["models"]["source"])
        target_name = str(manifest["models"]["target"])
        source = SentencePieceTokenizer.load(directory / source_name)
        target = source if shared and source_name == target_name else SentencePieceTokenizer.load(directory / target_name)
        if source.pad_id != 0 or source.unknown_id != 1 or source.bos_id != 2 or source.eos_id != 3:
            raise TokenizerCompatibilityError(
                f"Tokenizer {version} special IDs differ from platform contract PAD=0, UNK=1, BOS=2, EOS=3."
            )
        return cls(version, str(manifest["dataset_version"]), source, target, shared, manifest_path)


def _write_corpus(path: Path, records: list[Any], side: str | None) -> int:
    """Write SentencePiece's text-only training input and return sentence count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            if side in (None, "source"):
                output_file.write(record.source + "\n")
                count += 1
            if side in (None, "target"):
                output_file.write(record.target + "\n")
                count += 1
    return count


def _train_one(corpus: Path, prefix: Path, config: TokenizerConfig) -> None:
    """Run deterministic SentencePiece training with platform special-token IDs."""
    model_type = "bpe" if config.type == "sentencepiece_bpe" else "unigram"
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        model_type=model_type,
        vocab_size=config.vocabulary_size,
        character_coverage=config.character_coverage,
        byte_fallback=config.byte_fallback,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=config.pad_token,
        unk_piece=config.unknown_token,
        bos_piece=config.bos_token,
        eos_piece=config.eos_token,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        hard_vocab_limit=False,
        num_threads=1,
    )


def train_tokenizer(config: PlatformConfig, paths: ProjectPaths) -> TokenizerBundle:
    """Train immutable tokenizer artifacts solely from the current training split.

    Existing content-addressed artifacts are loaded rather than overwritten. This
    prevents a normal fine-tuning run from silently changing token IDs.

    Raises:
        FileNotFoundError: If no prepared dataset exists.
        RuntimeError: If the training corpus cannot support the requested model.
    """
    dataset_root = paths.dataset(config.language_pair.id)
    current_dataset = load_json(dataset_root / "metadata" / "current.json")
    if not current_dataset:
        raise FileNotFoundError(
            f"No prepared dataset for {config.language_pair.id}. Run `nmt dataset prepare` first."
        )
    dataset_version = str(current_dataset["dataset_version"])
    version = _tokenizer_version(dataset_version, config.tokenizer)
    output_directory = dataset_root / "tokenizer" / version
    manifest_path = output_directory / "manifest.json"
    if manifest_path.exists():
        LOGGER.info("Tokenizer %s already exists; reusing immutable artifacts", version)
        atomic_write_json(dataset_root / "tokenizer" / "current.json", {"tokenizer_version": version})
        return TokenizerBundle.load(dataset_root, version)
    train_path = dataset_root / "splits" / dataset_version / "train.jsonl"
    records = load_parallel_jsonl(train_path)
    if not records:
        raise ValueError(f"Training split is empty: {train_path}")
    output_directory.mkdir(parents=True, exist_ok=False)
    model_names: dict[str, str]
    sentence_counts: dict[str, int]
    if config.tokenizer.shared:
        corpus = output_directory / "training-shared.txt"
        count = _write_corpus(corpus, records, side=None)
        _train_one(corpus, output_directory / "shared", config.tokenizer)
        model_names = {"source": "shared.model", "target": "shared.model"}
        sentence_counts = {"shared": count}
    else:
        source_corpus = output_directory / "training-source.txt"
        target_corpus = output_directory / "training-target.txt"
        source_count = _write_corpus(source_corpus, records, side="source")
        target_count = _write_corpus(target_corpus, records, side="target")
        _train_one(source_corpus, output_directory / "source", config.tokenizer)
        _train_one(target_corpus, output_directory / "target", config.tokenizer)
        model_names = {"source": "source.model", "target": "target.model"}
        sentence_counts = {"source": source_count, "target": target_count}
    artifacts = sorted(output_directory.glob("*.model")) + sorted(output_directory.glob("*.vocab"))
    manifest: dict[str, Any] = {
        "tokenizer_version": version,
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config.tokenizer.model_dump(mode="json"),
        "models": model_names,
        "training_statistics": {
            "parallel_examples": len(records),
            "training_sentences": sentence_counts,
        },
        "artifact_sha256": {artifact.name: _sha256(artifact) for artifact in artifacts},
        "special_token_ids": {"pad": 0, "unknown": 1, "bos": 2, "eos": 3},
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(dataset_root / "tokenizer" / "current.json", {"tokenizer_version": version})
    LOGGER.info("Trained tokenizer %s from dataset %s", version, dataset_version)
    return TokenizerBundle.load(dataset_root, version)
