"""New-data detection, replay sampling, fine-tuning, and parent regression checks."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from nmt.config.schema import FineTuningConfig, PlatformConfig
from nmt.data.pipeline import load_parallel_jsonl
from nmt.data.records import ParallelRecord
from nmt.registry.local import LocalModelRegistry
from nmt.training.trainer import TrainingResult, train_model
from nmt.utils.io import load_json
from nmt.utils.paths import ProjectPaths
from nmt.versioning.comparison import compare_versions

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineTuningResult:
    """New child version, replay composition, and parent-regression comparison."""

    training: TrainingResult
    replay_report: dict[str, Any]
    comparison: dict[str, Any]


def _sample(records: list[ParallelRecord], count: int, randomizer: random.Random) -> list[ParallelRecord]:
    """Sample deterministically, using replacement only when the count exceeds history."""
    if count <= 0 or not records:
        return []
    if count <= len(records):
        return randomizer.sample(records, count)
    return [randomizer.choice(records) for _ in range(count)]


def build_replay_mixture(
    new_records: list[ParallelRecord],
    historical_records: list[ParallelRecord],
    config: FineTuningConfig,
    seed: int,
) -> tuple[list[ParallelRecord], dict[str, Any]]:
    """Mix new and historical directional examples according to replay strategy.

    Args:
        new_records: Current training rows absent from the parent's full dataset.
        historical_records: Parent training split available for replay.
        config: Strategy, ratios, count, and replay enabled state.
        seed: Experiment seed for reproducible samples.

    Returns:
        Training row sequence and a detailed sampling report.

    Raises:
        ValueError: If no genuinely new train rows exist or replay needs history.
    """
    if not new_records:
        raise ValueError(
            "No new training examples were detected relative to the parent dataset. Add data "
            "under datasets/<pair>/incoming and run dataset prepare before fine-tuning."
        )
    randomizer = random.Random(seed)
    if not config.replay_enabled:
        LOGGER.warning(
            "New-data-only fine-tuning can cause catastrophic forgetting; compare against the "
            "parent's original test set before promotion."
        )
        mixture = list(new_records)
        historical_sample: list[ParallelRecord] = []
    else:
        if not historical_records:
            raise ValueError("Replay is enabled but the parent training split is empty.")
        if config.replay_strategy == "fixed_count":
            historical_count = int(config.historical_example_count or 0)
            historical_sample = _sample(historical_records, historical_count, randomizer)
        elif config.replay_strategy == "balanced":
            historical_sample = _sample(historical_records, len(new_records), randomizer)
        elif config.replay_strategy in {"percentage", "weighted"}:
            ratio = config.historical_data_ratio / max(config.new_data_ratio, 1e-9)
            historical_sample = _sample(
                historical_records, round(len(new_records) * ratio), randomizer
            )
        elif config.replay_strategy == "domain_aware":
            new_domains = {record.domain for record in new_records if record.domain}
            domain_history = [record for record in historical_records if record.domain in new_domains]
            pool = domain_history or historical_records
            historical_sample = _sample(pool, len(new_records), randomizer)
        else:  # Pydantic prevents this branch, retained for defensive library use.
            raise ValueError(f"Unsupported replay strategy: {config.replay_strategy}")
        mixture = list(new_records) + historical_sample
        randomizer.shuffle(mixture)
    report = {
        "mode": "replay" if config.replay_enabled else "new_data_only",
        "strategy": config.replay_strategy if config.replay_enabled else None,
        "new_examples": len(new_records),
        "available_historical_examples": len(historical_records),
        "sampled_historical_examples": len(historical_sample),
        "training_examples": len(mixture),
        "effective_new_fraction": len(new_records) / max(1, len(mixture)),
        "catastrophic_forgetting_warning": not config.replay_enabled,
    }
    return mixture, report


def _pair_keys(records: list[ParallelRecord]) -> set[tuple[str, str]]:
    """Return exact directional pair identities used for new-data detection."""
    return {(record.source, record.target) for record in records}


def fine_tune_model(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    parent_version: str,
    notes: str = "",
) -> FineTuningResult:
    """Create a new immutable child through new-data or replay fine-tuning.

    The parent tokenizer/model architecture is reused exactly. Validation/test
    membership remains hash-stable. After training, both parent and child are
    evaluated on the parent's original fixed test set and promotion gates update
    the child to approved only when all mandatory regressions pass.
    """
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    parent = registry.resolve(parent_version)
    current = load_json(paths.dataset(config.language_pair.id) / "metadata" / "current.json")
    if not current:
        raise FileNotFoundError("No prepared current dataset for fine-tuning.")
    current_version = str(current["dataset_version"])
    parent_dataset_version = parent.dataset_versions[-1]
    if current_version == parent_dataset_version:
        raise ValueError(
            "Current dataset version is identical to the parent dataset; there is no new data."
        )
    reverse = direction != config.language_pair.directions()[0]
    parent_all = load_parallel_jsonl(
        paths.dataset(config.language_pair.id)
        / "processed"
        / parent_dataset_version
        / "pairs.jsonl",
        reverse=reverse,
    )
    historical_train = load_parallel_jsonl(
        paths.dataset(config.language_pair.id)
        / "splits"
        / parent_dataset_version
        / "train.jsonl",
        reverse=reverse,
    )
    current_train = load_parallel_jsonl(
        paths.dataset(config.language_pair.id) / "splits" / current_version / "train.jsonl",
        reverse=reverse,
    )
    historical_keys = _pair_keys(parent_all)
    new_train = [record for record in current_train if (record.source, record.target) not in historical_keys]
    mixture, replay_report = build_replay_mixture(
        new_train, historical_train, config.fine_tuning, config.training.seed
    )
    fine_training = config.training.model_copy(
        update={
            "learning_rate": config.fine_tuning.learning_rate,
            "epochs": config.fine_tuning.epochs,
        }
    )
    fine_config = config.model_copy(update={"training": fine_training})
    training = train_model(
        fine_config,
        paths,
        direction,
        parent_version=parent.version_id,
        training_records_override=mixture,
        freeze_embeddings=config.fine_tuning.freeze_embeddings,
        freeze_encoder_layers=config.fine_tuning.freeze_encoder_layers,
        notes=(notes + "\n" if notes else "") + f"Replay: {replay_report}",
    )
    comparison = compare_versions(
        fine_config,
        paths,
        direction,
        parent.version_id,
        training.version_id,
        dataset_version=parent_dataset_version,
        device=config.training.device,
        update_candidate_status=True,
    )
    return FineTuningResult(training, replay_report, comparison)
