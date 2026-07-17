"""Safe readers for UI-tail-able experiment JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nmt.utils.io import load_json
from nmt.utils.paths import ProjectPaths, validate_identifier


def read_metric_events(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON metric lines while ignoring an actively-written partial line."""
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as metric_file:
        for line in metric_file:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
    return events


def list_experiments(paths: ProjectPaths, pair: str) -> list[dict[str, Any]]:
    """Return newest-first experiment manifest, result, and latest metrics."""
    root = paths.root / "experiments" / validate_identifier(pair, "pair")
    if not root.exists():
        return []
    experiments = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        manifest = load_json(directory / "manifest.json", {})
        result = load_json(directory / "result.json", None)
        metrics = read_metric_events(directory / "metrics.jsonl")
        experiments.append(
            {
                "experiment_id": directory.name,
                "manifest": manifest,
                "result": result,
                "latest_metric": metrics[-1] if metrics else None,
                "status": result.get("status") if result else "training",
            }
        )
    return sorted(experiments, key=lambda item: item["experiment_id"], reverse=True)
