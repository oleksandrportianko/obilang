"""Quality metric and historical replay strategy unit tests."""

from __future__ import annotations

from nmt.config.schema import FineTuningConfig
from nmt.data.records import ParallelRecord
from nmt.evaluation.metrics import calculate_text_metrics
from nmt.fine_tuning.replay import build_replay_mixture


def rows(prefix: str, count: int) -> list[ParallelRecord]:
    """Create distinct parallel rows for sampling tests."""
    return [ParallelRecord(f"{prefix} source {i}", f"{prefix} target {i}", "test", i) for i in range(count)]


def test_exact_metrics_and_preservation() -> None:
    """Perfect candidates score exact and preserve numbers/placeholders/markup."""
    source = ["Save {name} as <b>file 12</b>!"]
    reference = ["Save {name} as <b>file 12</b>!"]
    metrics = calculate_text_metrics(source, reference, reference, token_loss=0.5)
    assert metrics["exact_match"] == 1.0
    assert metrics["number_accuracy"] == 1.0
    assert metrics["placeholder_accuracy"] == 1.0
    assert metrics["markup_accuracy"] == 1.0
    assert metrics["perplexity"] > 1


def test_balanced_replay_uses_equal_new_and_history() -> None:
    """Recommended balanced mode deterministically gives both domains equal weight."""
    mixture, report = build_replay_mixture(
        rows("new", 3), rows("old", 10), FineTuningConfig(replay_strategy="balanced"), 42
    )
    assert len(mixture) == 6
    assert report["sampled_historical_examples"] == 3
    assert report["effective_new_fraction"] == 0.5


def test_new_data_only_emits_forgetting_flag() -> None:
    """Exclusive new-data mode never claims preservation of previous capability."""
    mixture, report = build_replay_mixture(
        rows("new", 2), rows("old", 5), FineTuningConfig(replay_enabled=False), 42
    )
    assert len(mixture) == 2
    assert report["catastrophic_forgetting_warning"] is True
