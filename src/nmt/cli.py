"""Single discoverable command-line interface for every local NMT workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer

from nmt.config.loader import load_config
from nmt.config.schema import ModelConfig, TrainingConfig
from nmt.data.pipeline import prepare_dataset, validate_dataset
from nmt.evaluation.runner import evaluate_version
from nmt.export.portable import export_model as export_model_artifact
from nmt.fine_tuning.replay import fine_tune_model
from nmt.inference.service import load_runtime
from nmt.registry.local import LocalModelRegistry, RegistryError
from nmt.tokenization.sentencepiece import train_tokenizer
from nmt.training.checkpoint import load_checkpoint
from nmt.training.trainer import train_model
from nmt.utils.io import atomic_write_bytes, load_json
from nmt.utils.logging import configure_logging
from nmt.utils.paths import ProjectPaths, discover_project_root
from nmt.versioning.comparison import compare_versions
from nmt.versioning.manifest import export_experiment_manifest

app = typer.Typer(
    name="nmt",
    help="Train and operate isolated bilingual Transformers from random weights.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Validate, prepare, inspect, and version parallel datasets.")
tokenizer_app = typer.Typer(help="Train and inspect immutable from-scratch tokenizers.")
train_app = typer.Typer(help="Start or resume directional model training.", invoke_without_command=True)
versions_app = typer.Typer(help="Inspect and manage immutable model version lineage.")
experiment_app = typer.Typer(help="Inspect and export reproducible experiment metadata.")
export_app = typer.Typer(help="Export registered inference artifacts.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(train_app, name="train")
app.add_typer(versions_app, name="versions")
app.add_typer(experiment_app, name="experiment")
app.add_typer(export_app, name="export")


@app.callback()
def main_callback(
    log_level: str = typer.Option("INFO", help="Structured log level."),
) -> None:
    """Configure structured console logging before dispatching a workflow."""
    configure_logging(log_level)


def _paths() -> ProjectPaths:
    """Resolve project artifacts from the current directory or environment override."""
    return ProjectPaths(discover_project_root())


def _print(value: Any) -> None:
    """Print stable UTF-8 JSON suitable for scripts and human inspection."""
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _direction(config: Any, requested: str | None) -> str:
    """Use the canonical forward direction when a command omits one."""
    selected = requested or config.language_pair.directions()[0]
    config.language_pair.languages_for_direction(selected)
    return selected


def _find_version(version_id: str) -> tuple[str, str]:
    """Locate a unique pair/direction registry containing a version identity."""
    paths = _paths()
    matches: list[tuple[str, str]] = []
    for registry_path in (paths.root / "models").glob("*/*/registry.json"):
        document = load_json(registry_path, {})
        if version_id in document.get("versions", {}) or any(
            item.get("version_label") == version_id for item in document.get("versions", {}).values()
        ):
            matches.append((registry_path.parents[1].name, registry_path.parent.name))
    if len(matches) != 1:
        raise RegistryError(
            f"Version selector {version_id!r} matched {len(matches)} registries; provide an exact unique ID."
        )
    return matches[0]


@dataset_app.command("validate")
def dataset_validate(
    pair: str = typer.Option(..., help="Language-pair ID."),
    config_files: list[Path] = typer.Option([], "--config", help="YAML overlay; repeatable."),
) -> None:
    """Audit current raw/incoming rows without writing processed artifacts."""
    paths = _paths()
    _print(validate_dataset(load_config(pair, config_files, paths.root), paths))


@dataset_app.command("prepare")
def dataset_prepare(
    pair: str = typer.Option(..., help="Language-pair ID."),
    config_files: list[Path] = typer.Option([], "--config", help="YAML overlay; repeatable."),
) -> None:
    """Create an immutable processed version, rejected rows, and stable splits."""
    paths = _paths()
    result = prepare_dataset(load_config(pair, config_files, paths.root), paths)
    _print(
        {
            "dataset_version": result.version,
            "processed_path": result.processed_path,
            "split_directory": result.split_directory,
            "report_path": result.report_path,
            "manifest_path": result.manifest_path,
            "report": result.report,
        }
    )


@dataset_app.command("report")
def dataset_report(pair: str = typer.Option(..., help="Language-pair ID.")) -> None:
    """Print the current immutable data-quality report."""
    paths = _paths()
    current = load_json(paths.dataset(pair) / "metadata" / "current.json")
    if not current:
        raise typer.BadParameter("No prepared dataset; run dataset prepare first.")
    version = current["dataset_version"]
    _print(load_json(paths.dataset(pair) / "metadata" / f"report-{version}.json"))


@tokenizer_app.command("train")
def tokenizer_train(
    pair: str = typer.Option(..., help="Language-pair ID."),
    config_files: list[Path] = typer.Option([], "--config", help="YAML overlay; repeatable."),
) -> None:
    """Train SentencePiece only from the current versioned training split."""
    paths = _paths()
    bundle = train_tokenizer(load_config(pair, config_files, paths.root), paths)
    _print(
        {
            "tokenizer_version": bundle.version,
            "dataset_version": bundle.dataset_version,
            "shared": bundle.shared,
            "source_vocabulary_size": bundle.source.vocabulary_size,
            "target_vocabulary_size": bundle.target.vocabulary_size,
            "manifest": bundle.manifest_path,
        }
    )


@train_app.callback(invoke_without_command=True)
def train_new(
    context: typer.Context,
    pair: str | None = typer.Option(None, help="Language-pair ID."),
    direction: str | None = typer.Option(None, help="Canonical direction; defaults forward."),
    config_files: list[Path] = typer.Option([], "--config", help="YAML overlay; repeatable."),
    notes: str = typer.Option("", help="Version notes."),
) -> None:
    """Start a fresh major model line when no training subcommand is supplied."""
    if context.invoked_subcommand is not None:
        return
    if not pair:
        raise typer.BadParameter("--pair is required for fresh training.")
    paths = _paths()
    config = load_config(pair, config_files, paths.root)
    result = train_model(config, paths, _direction(config, direction), notes=notes)
    _print(result.__dict__)


@train_app.command("resume")
def train_resume(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False, help="Trusted local .pt file."),
    device: str | None = typer.Option(None, help="Optional device override."),
) -> None:
    """Resume an interrupted version with optimizer, scheduler, scaler, and RNG state."""
    paths = _paths()
    payload = load_checkpoint(checkpoint.resolve())
    pair, direction = str(payload["language_pair"]), str(payload["direction"])
    config = load_config(pair, root=paths.root)
    model_config = ModelConfig.model_validate(payload["model_configuration"])
    training_config = TrainingConfig.model_validate(payload["training_configuration"])
    if device:
        training_config = training_config.model_copy(update={"device": device})
    config = config.model_copy(update={"model": model_config, "training": training_config})
    result = train_model(config, paths, direction, resume_checkpoint=checkpoint.resolve())
    _print(result.__dict__)


@app.command("fine-tune")
def fine_tune(
    pair: str = typer.Option(..., help="Language-pair ID."),
    from_version: str = typer.Option(..., "--from-version", help="Immutable parent version."),
    direction: str | None = typer.Option(None, help="Canonical direction; defaults forward."),
    config_files: list[Path] = typer.Option([], "--config", help="YAML overlay; repeatable."),
    new_data_only: bool = typer.Option(False, help="Disable replay (catastrophic-forgetting risk)."),
    notes: str = typer.Option("", help="Child version notes."),
) -> None:
    """Fine-tune to a new child version and regress against the parent benchmark."""
    paths = _paths()
    config = load_config(pair, config_files, paths.root)
    if new_data_only:
        config = config.model_copy(
            update={"fine_tuning": config.fine_tuning.model_copy(update={"replay_enabled": False})}
        )
    result = fine_tune_model(config, paths, _direction(config, direction), from_version, notes)
    _print(
        {
            "training": result.training.__dict__,
            "replay": result.replay_report,
            "promotion_gates": result.comparison["promotion_gates"],
            "comparison_report": result.comparison["report_path"],
        }
    )


@app.command("evaluate")
def evaluate(
    pair: str = typer.Option(..., help="Language-pair ID."),
    version: str = typer.Option(..., help="Version ID, label, or production."),
    direction: str | None = typer.Option(None, help="Canonical direction; defaults forward."),
    dataset_version: str | None = typer.Option(None, help="Optional fixed test dataset version."),
    device: str = typer.Option("auto", help="auto, cpu, cuda, or mps."),
) -> None:
    """Evaluate a model using loss, BLEU, chrF, TER, and preservation metrics."""
    paths = _paths()
    config = load_config(pair, root=paths.root)
    _print(evaluate_version(config, paths, _direction(config, direction), version, dataset_version, device))


@app.command("compare")
def compare(
    version_a: str = typer.Option(..., help="Baseline version ID."),
    version_b: str = typer.Option(..., help="Candidate version ID."),
    pair: str | None = typer.Option(None, help="Pair override; inferred from A by default."),
    direction: str | None = typer.Option(None, help="Direction override; inferred from A by default."),
    dataset_version: str | None = typer.Option(None, help="Fixed common benchmark."),
    device: str = typer.Option("auto", help="Evaluation device."),
) -> None:
    """Create a side-by-side quality, data, config, speed, and output report."""
    inferred_pair, inferred_direction = _find_version(version_a)
    pair, direction = pair or inferred_pair, direction or inferred_direction
    paths = _paths()
    config = load_config(pair, root=paths.root)
    _print(compare_versions(config, paths, direction, version_a, version_b, dataset_version=dataset_version, device=device))


@versions_app.command("list")
def versions_list(
    pair: str = typer.Option(..., help="Language-pair ID."),
    direction: str | None = typer.Option(None, help="One direction or all configured directions."),
) -> None:
    """List version lineage and lifecycle status."""
    paths = _paths()
    config = load_config(pair, root=paths.root)
    directions = [direction] if direction else config.language_pair.directions()
    _print(
        {
            item: [
                version.model_dump(mode="json")
                for version in LocalModelRegistry(paths, pair, item).list_versions()
            ]
            for item in directions
        }
    )


@versions_app.command("inspect")
def versions_inspect(version: str = typer.Option(..., help="Unique version ID.")) -> None:
    """Print all reproducibility and lineage metadata for a version."""
    pair, direction = _find_version(version)
    _print(LocalModelRegistry(_paths(), pair, direction).resolve(version).model_dump(mode="json"))


@versions_app.command("promote")
def versions_promote(
    version: str = typer.Option(..., help="Unique approved version ID."),
    override: bool = typer.Option(False, help="Manually override failed/unrun promotion gates."),
) -> None:
    """Set the directional production pointer while protecting the artifact."""
    pair, direction = _find_version(version)
    metadata = LocalModelRegistry(_paths(), pair, direction).promote(version, override)
    _print(metadata.model_dump(mode="json"))


@versions_app.command("rollback")
def versions_rollback(
    pair: str = typer.Option(..., help="Language-pair ID."),
    to: str = typer.Option(..., "--to", help="Prior version ID."),
    direction: str | None = typer.Option(None, help="Direction inferred from target by default."),
) -> None:
    """Move production back to a prior approved/protected version."""
    inferred_pair, inferred_direction = _find_version(to)
    if inferred_pair != pair:
        raise typer.BadParameter(f"Rollback target belongs to {inferred_pair}, not {pair}.")
    metadata = LocalModelRegistry(_paths(), pair, direction or inferred_direction).rollback(to)
    _print(metadata.model_dump(mode="json"))


@versions_app.command("set-status")
def versions_set_status(
    version: str = typer.Option(..., help="Unique version ID."),
    status: str = typer.Option(..., help="candidate, approved, rejected, archived, or failed."),
) -> None:
    """Record a manual review decision without changing the production pointer."""
    allowed = {"candidate", "approved", "rejected", "archived", "failed"}
    if status not in allowed:
        raise typer.BadParameter(f"status must be one of {sorted(allowed)}")
    pair, direction = _find_version(version)
    _print(LocalModelRegistry(_paths(), pair, direction).update(version, status=status).model_dump(mode="json"))


@app.command("translate")
def translate(
    pair: str = typer.Option(..., help="Language-pair ID."),
    direction: str = typer.Option(..., help="Canonical translation direction."),
    version: str = typer.Option("production", help="Version ID, label, or production."),
    text: str = typer.Option(..., help="Source sentence."),
    decoding: str = typer.Option("beam", help="beam or greedy."),
    beam_width: int = typer.Option(4, min=1, max=16),
    length_penalty: float = typer.Option(0.6, min=0),
    maximum_length: int | None = typer.Option(None, min=2),
    device: str = typer.Option("auto"),
) -> None:
    """Translate text using a selected local registered model."""
    paths = _paths()
    runtime = load_runtime(load_config(pair, root=paths.root), paths, direction, version, device)
    result = runtime.translate_batch(
        [text], decoding=decoding, beam_width=beam_width, length_penalty=length_penalty, maximum_length=maximum_length
    )[0]
    _print(result.__dict__)


@app.command("translate-file")
def translate_file(
    pair: str = typer.Option(...),
    direction: str = typer.Option(...),
    version: str = typer.Option("production"),
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    output_path: Path = typer.Option(..., "--output", dir_okay=False),
    decoding: str = typer.Option("beam"),
    beam_width: int = typer.Option(4, min=1, max=16),
    device: str = typer.Option("auto"),
    batch_size: int = typer.Option(16, min=1),
) -> None:
    """Translate a UTF-8 line-oriented file in bounded batches."""
    try:
        texts = input_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(f"Input must be UTF-8: {exc}") from exc
    paths = _paths()
    runtime = load_runtime(load_config(pair, root=paths.root), paths, direction, version, device)
    translations: list[str] = []
    for start in range(0, len(texts), batch_size):
        rows = runtime.translate_batch(texts[start : start + batch_size], decoding=decoding, beam_width=beam_width)
        translations.extend(row.translation for row in rows)
    atomic_write_bytes(output_path.resolve(), ("\n".join(translations) + "\n").encode("utf-8"))
    _print({"input_rows": len(texts), "output": output_path.resolve(), "model_version": runtime.version.version_id})


@export_app.command("model")
def export_model(
    pair: str = typer.Option(...),
    direction: str = typer.Option(...),
    version: str = typer.Option("production"),
) -> None:
    """Export a safe forward graph, exact tokenizers, and inference manifest."""
    paths = _paths()
    destination = export_model_artifact(
        load_config(pair, root=paths.root), paths, direction, version
    )
    _print({"export_directory": destination})


@experiment_app.command("export-manifest")
def experiment_export_manifest(
    experiment: str = typer.Option(..., "--experiment", help="Experiment ID."),
    pair: str = typer.Option(..., help="Language-pair ID."),
    output: Path | None = typer.Option(None, help="Optional JSON destination."),
) -> None:
    """Export all recorded inputs needed to reproduce and audit one run."""
    _print({"manifest": export_experiment_manifest(_paths(), pair, experiment, output)})


@app.command("api")
def api(
    host: str = typer.Option("127.0.0.1", help="Bind address; localhost is safest."),
    port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    """Run the local REST API; no cloud service is required."""
    import uvicorn
    from nmt.api.app import create_app

    uvicorn.run(create_app(_paths().root), host=host, port=port)


@app.command("ui")
def ui(
    host: str = typer.Option("127.0.0.1", help="Bind address; localhost is safest."),
    port: int = typer.Option(8501, min=1, max=65535),
) -> None:
    """Run the complete local Streamlit monitoring and management UI."""
    from streamlit.web import cli as streamlit_cli

    app_path = _paths().root / "ui" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    app()
