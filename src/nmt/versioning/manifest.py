"""Self-contained reproducibility manifest export for completed or failed experiments."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from nmt.registry.local import LocalModelRegistry
from nmt.utils.io import atomic_write_json, load_json
from nmt.utils.paths import ProjectPaths, validate_identifier


def export_experiment_manifest(
    paths: ProjectPaths, pair: str, experiment_id: str, output: Path | None = None
) -> Path:
    """Export run, model, data, tokenizer, source, and dependency identities.

    Args:
        paths: Project layout.
        pair: Language-pair namespace containing the experiment.
        experiment_id: Safe local experiment directory name.
        output: Optional destination, defaulting under `reports/reproducibility`.

    Returns:
        Written JSON manifest path.

    Raises:
        FileNotFoundError: If the experiment manifest is absent.
    """
    validate_identifier(pair, "pair")
    validate_identifier(experiment_id, "experiment ID")
    experiment = paths.experiment(pair, experiment_id)
    manifest = load_json(experiment / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"Experiment manifest does not exist: {experiment / 'manifest.json'}")
    result = load_json(experiment / "result.json", None)
    direction = str(manifest["direction"])
    registry = LocalModelRegistry(paths, pair, direction)
    version = registry.resolve(str(manifest["model_version"]))
    dataset_version = str(manifest["dataset_version"])
    tokenizer_version = str(manifest["tokenizer_version"])
    dependencies: dict[str, str] = {}
    for package in (
        "torch",
        "numpy",
        "pydantic",
        "PyYAML",
        "typer",
        "sentencepiece",
        "sacrebleu",
        "filelock",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    document: dict[str, Any] = {
        "schema_version": 1,
        "experiment": manifest,
        "result": result,
        "model_version": version.model_dump(mode="json"),
        "dataset_manifest": load_json(
            paths.dataset(pair) / "processed" / dataset_version / "manifest.json"
        ),
        "tokenizer_manifest": load_json(
            paths.dataset(pair) / "tokenizer" / tokenizer_version / "manifest.json"
        ),
        "dependencies": dependencies,
    }
    destination = output or (
        paths.root / "reports" / "reproducibility" / f"{experiment_id}.json"
    )
    atomic_write_json(destination, document)
    return destination
