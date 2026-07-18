"""Registered model evaluation on fixed, versioned test sets."""

from __future__ import annotations

import time
from typing import Any

from nmt.config.schema import PlatformConfig, TrainingConfig
from nmt.data.pipeline import load_parallel_jsonl
from nmt.inference.service import load_runtime
from nmt.registry.local import LocalModelRegistry
from nmt.training.data import TranslationDataset
from nmt.training.trainer import _data_loader, evaluate_loader
from nmt.utils.io import atomic_write_json
from nmt.utils.paths import ProjectPaths


def evaluate_version(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    version: str,
    dataset_version: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate a registered model on an immutable test dataset version.

    Args:
        config: Validated language-pair settings.
        paths: Project artifact layout.
        direction: Independent directional model ID.
        version: Exact/label/production registry selector.
        dataset_version: Fixed benchmark version, defaulting to the model's latest
            recorded dataset.
        device: CPU/CUDA/MPS selection.

    Returns:
        Metrics, real source/reference/candidate examples, throughput, memory,
        parameter count, checkpoint bytes, and artifact identities.

    Side effects:
        Writes a report under `reports/<pair>/<direction>` and updates registry
        test metrics when evaluating the version's own latest dataset.
    """
    runtime = load_runtime(config, paths, direction, version, device)
    metadata = runtime.version
    selected_dataset = dataset_version or metadata.dataset_versions[-1]
    reverse = direction != config.language_pair.directions()[0]
    records = load_parallel_jsonl(
        paths.dataset(config.language_pair.id)
        / "splits"
        / selected_dataset
        / "test.jsonl",
        reverse=reverse,
    )
    dataset = TranslationDataset(records, runtime.tokenizers, runtime.model.config.maximum_sequence_length)
    training_config = TrainingConfig.model_validate(metadata.training_configuration).model_copy(
        update={"device": device, "mixed_precision": False, "num_workers": 0}
    )
    loader, _ = _data_loader(
        dataset,
        training_config,
        runtime.tokenizers,
        shuffle=False,
        token_budget=config.evaluation.batch_tokens,
    )
    started = time.perf_counter()
    metrics, samples = evaluate_loader(
        runtime.model,
        loader,
        runtime.tokenizers,
        runtime.selection.device,
        training_config.label_smoothing,
        maximum_generation_length=config.evaluation.maximum_generation_length,
    )
    duration = time.perf_counter() - started
    checkpoint = paths.root / str(metadata.checkpoint_path)
    report: dict[str, Any] = {
        "language_pair": config.language_pair.id,
        "direction": direction,
        "model_version": metadata.version_id,
        "dataset_version": selected_dataset,
        "tokenizer_version": metadata.tokenizer_version,
        "metrics": metrics,
        "samples": samples,
        "performance": {
            "evaluation_seconds": duration,
            "sentences_per_second": len(dataset) / max(duration, 1e-9),
            "parameter_count": sum(parameter.numel() for parameter in runtime.model.parameters()),
            "checkpoint_bytes": checkpoint.stat().st_size,
        },
    }
    destination = (
        paths.root
        / "reports"
        / config.language_pair.id
        / direction
        / f"evaluation-{metadata.version_id}-{selected_dataset}.json"
    )
    atomic_write_json(destination, report)
    if selected_dataset == metadata.dataset_versions[-1]:
        LocalModelRegistry(paths, config.language_pair.id, direction).update(
            metadata.version_id,
            test_metrics={key: value for key, value in metrics.items() if isinstance(value, (int, float))},
        )
    report["report_path"] = str(destination.relative_to(paths.root))
    return report
