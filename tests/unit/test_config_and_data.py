"""Configuration, reader, normalization, filtering, and split unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nmt.config.schema import DataConfig, ModelConfig
from nmt.data.pipeline import normalize_text, split_records, validate_records
from nmt.data.readers import DatasetAlignmentError, read_parallel_text
from nmt.data.records import ParallelRecord


def test_model_configuration_rejects_invalid_head_width() -> None:
    """Attention projections require an exactly divisible embedding width."""
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig(embedding_dimension=30, attention_heads=8)


def test_normalization_preserves_punctuation_and_numbers() -> None:
    """Normalization changes Unicode/spacing but not semantically important symbols."""
    assert normalize_text("  Price:\t 1,234.50!  ") == "Price: 1,234.50!"


def test_validation_retains_reasons_for_duplicates_conflicts_and_empty() -> None:
    """Every rejected row receives all applicable auditable reason codes."""
    records = [
        ParallelRecord("aa", "bb", "x.tsv", 1),
        ParallelRecord("aa", "bb", "x.tsv", 2),
        ParallelRecord("aa", "cc", "x.tsv", 3),
        ParallelRecord("dd", "bb", "x.tsv", 4),
        ParallelRecord("", "ee", "x.tsv", 5),
    ]
    accepted, rejected, report = validate_records(records, DataConfig(maximum_length_ratio=4))
    assert [(item.source, item.target) for item in accepted] == [("aa", "bb")]
    assert report.total_rows == 5
    assert report.rejected_rows == 4
    assert report.duplicate_rows == 1
    assert report.conflicting_pairs == 2
    assert report.empty_rows == 1
    assert "conflicting_translation" in rejected[1].reasons


def test_near_duplicate_sources_cannot_cross_splits() -> None:
    """Case, punctuation, and whitespace variants share a deterministic split."""
    records = [
        ParallelRecord("Hello, world!", "one", "x", 1),
        ParallelRecord("hello world", "two", "x", 2),
    ]
    splits = split_records(records, DataConfig())
    occupied = [name for name, rows in splits.items() if rows]
    assert len(occupied) == 1
    assert len(splits[occupied[0]]) == 2


def test_parallel_reader_reports_both_alignment_counts(tmp_path: Path) -> None:
    """Misalignment fails before examples can be paired incorrectly."""
    source, target = tmp_path / "source.txt", tmp_path / "target.txt"
    source.write_text("a\nb\n", encoding="utf-8")
    target.write_text("x\n", encoding="utf-8")
    with pytest.raises(DatasetAlignmentError, match="2 lines.*1 lines"):
        list(read_parallel_text(source, target, tmp_path))
