"""Local structured experiment manifests, metrics, samples, and terminal state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nmt.utils.io import append_jsonl, atomic_write_json


@dataclass(frozen=True)
class ExperimentTracker:
    """Append-only run metrics and atomically updated result metadata."""

    experiment_id: str
    directory: Path

    @property
    def metrics_path(self) -> Path:
        """Return the UI-tail-able newline JSON metric stream."""
        return self.directory / "metrics.jsonl"

    def start(self, manifest: dict[str, Any]) -> None:
        """Write the reproducibility manifest before expensive work starts."""
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.directory / "manifest.json",
            manifest
            | {
                "experiment_id": self.experiment_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "training",
            },
        )

    def metric(self, event: dict[str, Any]) -> None:
        """Append one real training/validation/checkpoint metric event."""
        append_jsonl(
            self.metrics_path,
            {"timestamp": datetime.now(timezone.utc).isoformat()} | event,
        )

    def samples(self, step: int, examples: list[dict[str, Any]]) -> None:
        """Persist validation translations used by the training UI."""
        atomic_write_json(self.directory / "samples" / f"step-{step:08d}.json", examples)

    def finish(self, result: dict[str, Any]) -> None:
        """Write success, interruption, or failure details atomically."""
        atomic_write_json(
            self.directory / "result.json",
            result | {"finished_at": datetime.now(timezone.utc).isoformat()},
        )
