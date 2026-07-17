"""Shared isolated project factory for integration and smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_document() -> dict[str, Any]:
    """Return a tiny but complete deterministic CPU test configuration."""
    return {
        "language_pair": {
            "id": "xx-yy",
            "source_language": "xx",
            "target_language": "yy",
            "bidirectional": True,
        },
        "data": {
            "maximum_characters": 100,
            "maximum_tokens_approximation": 32,
            "maximum_length_ratio": 4.0,
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15, "seed": 42},
        },
        "tokenizer": {
            "type": "sentencepiece_bpe",
            "shared": True,
            "vocabulary_size": 64,
            "character_coverage": 1.0,
        },
        "model": {
            "embedding_dimension": 16,
            "attention_heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "feedforward_dimension": 32,
            "dropout": 0.0,
            "maximum_sequence_length": 16,
            "tied_target_embeddings": True,
            "share_source_target_embeddings": True,
            "initialization": "xavier_uniform",
        },
        "training": {
            "batch_tokens": 512,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "scheduler": "constant",
            "epochs": 2,
            "label_smoothing": 0.0,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "checkpoint_every_steps": 1,
            "validate_every_steps": 1,
            "keep_periodic_checkpoints": 2,
            "patience_validations": 4,
            "seed": 42,
            "deterministic": True,
            "device": "cpu",
            "num_workers": 0,
            "log_every_steps": 1,
            "maximum_steps": 2,
        },
        "fine_tuning": {
            "learning_rate": 0.0005,
            "epochs": 1,
            "replay_enabled": True,
            "replay_strategy": "balanced",
            "new_data_ratio": 0.5,
            "historical_data_ratio": 0.5,
            "freeze_embeddings": False,
            "freeze_encoder_layers": 0,
        },
        "evaluation": {"sample_count": 3, "beam_width": 2},
        "promotion": {
            "minimum_chrf_change": -100.0,
            "maximum_bleu_regression": 100.0,
            "maximum_number_accuracy_regression": 1.0,
            "maximum_placeholder_accuracy_regression": 1.0,
        },
    }


def create_test_project(root: Path, row_count: int = 60) -> Path:
    """Create a complete isolated pair namespace and simple parallel TSV corpus."""
    (root / "configs" / "language_pairs").mkdir(parents=True)
    (root / "datasets" / "xx-yy" / "raw").mkdir(parents=True)
    for name in ("incoming", "processed", "splits", "rejected", "metadata", "tokenizer"):
        (root / "datasets" / "xx-yy" / name).mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test-nmt'\nversion='0.0.0'\n", encoding="utf-8")
    with (root / "configs" / "language_pairs" / "xx-yy.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(project_document(), file, sort_keys=False)
    lines = ["source_text\ttarget_text\tdomain"]
    lines.extend(f"item {index}\tline {index}\tgeneral" for index in range(1, row_count + 1))
    (root / "datasets" / "xx-yy" / "raw" / "corpus.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return root
