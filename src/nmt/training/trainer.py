"""End-to-end optimization loop with validation, checkpoints, resume, and recovery."""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from nmt.config.schema import ModelConfig, PlatformConfig, TrainingConfig
from nmt.data.pipeline import load_parallel_jsonl
from nmt.data.records import ParallelRecord
from nmt.evaluation.metrics import calculate_text_metrics
from nmt.model.loss import translation_loss
from nmt.model.transformer import NMTTransformer, build_model
from nmt.registry.local import LocalModelRegistry, ModelVersion, new_model_version
from nmt.tokenization.sentencepiece import TokenizerBundle, TokenizerCompatibilityError
from nmt.training.checkpoint import (
    apply_retention,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from nmt.training.data import (
    TokenBatchSampler,
    TranslationBatch,
    TranslationCollator,
    TranslationDataset,
)
from nmt.training.device import device_memory_megabytes, select_device
from nmt.training.experiment import ExperimentTracker
from nmt.utils.io import atomic_copy, load_json
from nmt.utils.paths import ProjectPaths
from nmt.utils.reproducibility import environment_manifest, set_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Terminal model/version/checkpoint and evaluation summary."""

    version_id: str
    experiment_id: str
    final_checkpoint: Path
    best_validation_loss: float
    test_metrics: dict[str, Any]
    global_step: int
    duration_seconds: float


def _direction_is_reverse(config: PlatformConfig, direction: str) -> bool:
    """Validate a direction and return whether canonical dataset columns must swap."""
    config.language_pair.languages_for_direction(direction)
    return direction != config.language_pair.directions()[0]


def directional_tokenizers(bundle: TokenizerBundle, reverse: bool) -> TokenizerBundle:
    """Swap independent source/target processors for the reverse directional model."""
    if not reverse or bundle.shared:
        return bundle
    return TokenizerBundle(
        bundle.version,
        bundle.dataset_version,
        bundle.target,
        bundle.source,
        False,
        bundle.manifest_path,
    )


def create_scheduler(
    optimizer: AdamW, config: TrainingConfig, estimated_optimizer_steps: int
) -> LRScheduler:
    """Create a warmup plus inverse-square-root, cosine, or constant schedule."""
    warmup = config.warmup_steps

    def multiplier(step_index: int) -> float:
        step = step_index + 1
        if warmup and step <= warmup:
            return step / warmup
        if config.scheduler == "inverse_sqrt":
            return math.sqrt(max(1, warmup) / step)
        if config.scheduler == "cosine":
            remaining = max(1, estimated_optimizer_steps - warmup)
            progress = min(1.0, max(0.0, (step - warmup) / remaining))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _data_loader(
    dataset: TranslationDataset,
    config: TrainingConfig,
    tokenizers: TokenizerBundle,
    shuffle: bool,
    epoch: int = 0,
    token_budget: int | None = None,
) -> tuple[DataLoader[TranslationBatch], TokenBatchSampler]:
    """Create a dynamic-token DataLoader and its epoch-aware sampler."""
    sampler = TokenBatchSampler(dataset, token_budget or config.batch_tokens, shuffle, config.seed, epoch)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=TranslationCollator(tokenizers.source.pad_id, tokenizers.target.pad_id),
        num_workers=config.num_workers,
        pin_memory=config.device in {"auto", "cuda"} and torch.cuda.is_available(),
    )
    return loader, sampler


def _move_batch(batch: TranslationBatch, device: torch.device) -> TranslationBatch:
    """Move tensors to the training device while retaining CPU source/reference text."""
    return replace(
        batch,
        source_ids=batch.source_ids.to(device, non_blocking=True),
        target_input_ids=batch.target_input_ids.to(device, non_blocking=True),
        target_output_ids=batch.target_output_ids.to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate_loader(
    model: NMTTransformer,
    loader: DataLoader[TranslationBatch],
    tokenizers: TokenizerBundle,
    device: torch.device,
    label_smoothing: float,
    sample_limit: int | None = None,
    maximum_generation_length: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Measure loss and generate real translations for an entire stable split.

    Args:
        model: Trained directional Transformer.
        loader: Non-shuffled validation or test loader.
        tokenizers: Matching immutable bundle.
        device: Device holding model parameters.
        label_smoothing: Training-compatible loss setting.
        sample_limit: Limit only returned examples; metrics always use all rows.

    Returns:
        Metric dictionary and source/reference/candidate sample records.

    Raises:
        ValueError: If the split is empty.
    """
    model.eval()
    loss_sum = 0.0
    non_padding_tokens = 0
    sources: list[str] = []
    references: list[str] = []
    candidates: list[str] = []
    unknown_count = 0
    generated_count = 0
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        logits = model(batch.source_ids, batch.target_input_ids)
        loss = translation_loss(
            logits, batch.target_output_ids, tokenizers.target.pad_id, label_smoothing
        )
        token_count = int(batch.target_output_ids.ne(tokenizers.target.pad_id).sum().item())
        loss_sum += float(loss.item()) * token_count
        non_padding_tokens += token_count
        # Reference length informs a bounded evaluation maximum without reading gold
        # tokens through the model. The factor accommodates normal length expansion.
        generation_limit = min(
            model.config.maximum_sequence_length,
            max(8, batch.target_output_ids.size(1) * 2),
        )
        if maximum_generation_length is not None:
            generation_limit = min(generation_limit, maximum_generation_length)
        generated = model.greedy_generate(
            batch.source_ids,
            tokenizers.target.bos_id,
            tokenizers.target.eos_id,
            generation_limit,
        )
        for source, reference, ids in zip(
            cpu_batch.source_texts, cpu_batch.target_texts, generated.token_ids
        ):
            sources.append(source)
            references.append(reference)
            candidates.append(tokenizers.target.decode(ids))
            unknown_count += sum(token_id == tokenizers.target.unknown_id for token_id in ids)
            generated_count += max(0, len(ids) - 1)
    if not sources or non_padding_tokens == 0:
        raise ValueError(
            "Evaluation split is empty after token-length filtering. Add more data or adjust "
            "the stable split/maximum sequence settings."
        )
    mean_loss = loss_sum / non_padding_tokens
    metrics = calculate_text_metrics(
        sources,
        references,
        candidates,
        token_loss=mean_loss,
        unknown_token_count=unknown_count,
        generated_token_count=generated_count,
    )
    samples = [
        {"source": source, "reference": reference, "candidate": candidate}
        for source, reference, candidate in zip(sources, references, candidates)
    ]
    return metrics, samples[:sample_limit] if sample_limit is not None else samples


def _checkpoint_state(
    config: PlatformConfig,
    version: ModelVersion,
    dataset_version: str,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_loss: float,
    validations_without_improvement: int,
    processed_tokens: int,
    processed_examples: int,
) -> dict[str, Any]:
    """Construct metadata stored alongside every serialized trainable state."""
    return {
        "language_pair": config.language_pair.id,
        "direction": version.direction,
        "model_version": version.version_id,
        "experiment_id": version.experiment_id,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_metric": best_loss,
        "early_stopping_state": {
            "validations_without_improvement": validations_without_improvement
        },
        "model_configuration": config.model.model_dump(mode="json"),
        "training_configuration": config.training.model_dump(mode="json"),
        "tokenizer_version": version.tokenizer_version,
        "dataset_version": dataset_version,
        "processed_tokens": processed_tokens,
        "processed_examples": processed_examples,
    }


def _experiment_id() -> str:
    """Return a sortable UTC run identity with collision-resistant suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"exp-{timestamp}-{uuid.uuid4().hex[:8]}"


def _relative_to_root(path: Path, paths: ProjectPaths) -> str:
    """Store portable artifact paths and reject accidental external locations."""
    try:
        return str(path.resolve().relative_to(paths.root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes project root: {path}") from exc


def train_model(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    *,
    resume_checkpoint: Path | None = None,
    parent_version: str | None = None,
    training_records_override: list[ParallelRecord] | None = None,
    freeze_embeddings: bool = False,
    freeze_encoder_layers: int = 0,
    notes: str = "",
) -> TrainingResult:
    """Train, resume, or continue one independent directional model.

    Args:
        config: Pair/model/training settings validated before allocation.
        paths: Repository artifact layout.
        direction: Enabled canonical direction such as `en-to-uk`.
        resume_checkpoint: Existing interrupted/periodic checkpoint. Resume keeps
            its model version and restores optimizer, scheduler, scaler, and RNG.
        parent_version: Existing immutable model used to initialize a new child;
            optimizer state is intentionally reset for fine-tuning.
        training_records_override: Optional replay/new-data training mixture. Stable
            validation and test files still come from the versioned dataset.
        freeze_embeddings: Disable gradients for source/target token embeddings.
        freeze_encoder_layers: Disable gradients for the first N encoder layers.
        notes: Human context stored in version metadata.

    Returns:
        Final version/checkpoint, test metrics, step, and duration information.

    Raises:
        ValueError: For empty splits, incompatible config/tokenizer/checkpoint, NaN,
            or attempting resume and parent fine-tuning simultaneously.
        RuntimeError: With actionable CUDA out-of-memory guidance.

    Side effects:
        Creates experiment, checkpoint, registry, metric, sample, and version files.
        Keyboard interruption writes a recoverable checkpoint before propagating.
    """
    if resume_checkpoint and parent_version:
        raise ValueError("Choose either exact resume or parent fine-tuning, not both.")
    reverse = _direction_is_reverse(config, direction)
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    resume_payload = load_checkpoint(resume_checkpoint) if resume_checkpoint else None
    current_dataset = load_json(paths.dataset(config.language_pair.id) / "metadata" / "current.json")
    if not current_dataset:
        raise FileNotFoundError("No prepared dataset. Run `nmt dataset prepare` before training.")
    dataset_version = str(
        resume_payload["dataset_version"] if resume_payload else current_dataset["dataset_version"]
    )
    model_config = config.model
    if resume_payload:
        if resume_payload.get("language_pair") != config.language_pair.id or resume_payload.get("direction") != direction:
            raise ValueError("Checkpoint language pair/direction does not match this training request.")
        model_config = ModelConfig.model_validate(resume_payload["model_configuration"])
        version = registry.resolve(str(resume_payload["model_version"]))
        experiment_id = str(resume_payload["experiment_id"])
        tokenizer_version = str(resume_payload["tokenizer_version"])
    elif parent_version:
        parent = registry.resolve(parent_version)
        if not parent.checkpoint_path:
            raise ValueError(f"Parent version {parent.version_id} has no completed checkpoint.")
        model_config = ModelConfig.model_validate(parent.model_configuration)
        tokenizer_version = parent.tokenizer_version
        experiment_id = _experiment_id()
        version = new_model_version(
            registry,
            experiment_id,
            model_config.model_dump(mode="json"),
            config.training.model_dump(mode="json"),
            tokenizer_version,
            list(dict.fromkeys(parent.dataset_versions + [dataset_version])),
            environment_manifest(paths.root),
            config.training.seed,
            parent.version_id,
            notes,
        )
        registry.add(version)
    else:
        current_tokenizer = load_json(paths.dataset(config.language_pair.id) / "tokenizer" / "current.json")
        if not current_tokenizer:
            raise FileNotFoundError("No tokenizer. Run `nmt tokenizer train` before training.")
        tokenizer_version = str(current_tokenizer["tokenizer_version"])
        experiment_id = _experiment_id()
        version = new_model_version(
            registry,
            experiment_id,
            model_config.model_dump(mode="json"),
            config.training.model_dump(mode="json"),
            tokenizer_version,
            [dataset_version],
            environment_manifest(paths.root),
            config.training.seed,
            notes=notes,
        )
        registry.add(version)
    effective_config = config.model_copy(update={"model": model_config})
    base_bundle = TokenizerBundle.load(paths.dataset(config.language_pair.id), tokenizer_version)
    tokenizers = directional_tokenizers(base_bundle, reverse)
    train_records = training_records_override or load_parallel_jsonl(
        paths.dataset(config.language_pair.id) / "splits" / dataset_version / "train.jsonl",
        reverse=reverse,
    )
    validation_records = load_parallel_jsonl(
        paths.dataset(config.language_pair.id) / "splits" / dataset_version / "validation.jsonl",
        reverse=reverse,
    )
    test_records = load_parallel_jsonl(
        paths.dataset(config.language_pair.id) / "splits" / dataset_version / "test.jsonl",
        reverse=reverse,
    )
    train_dataset = TranslationDataset(train_records, tokenizers, model_config.maximum_sequence_length)
    validation_dataset = TranslationDataset(
        validation_records, tokenizers, model_config.maximum_sequence_length
    )
    test_dataset = TranslationDataset(test_records, tokenizers, model_config.maximum_sequence_length)
    if not train_dataset:
        raise ValueError("Training split is empty after precise token-length filtering.")
    if not validation_dataset or not test_dataset:
        raise ValueError(
            "Stable validation and test splits must each contain an encodable row. Add more data "
            "or adjust split percentages, then create a new dataset version."
        )
    selection = select_device(config.training.device, config.training.mixed_precision)
    LOGGER.info(
        "Training on %s with %s precision (%s)",
        selection.device,
        selection.precision,
        selection.description,
        extra={"experiment_id": experiment_id, "model_version": version.version_id},
    )
    set_seed(config.training.seed, config.training.deterministic)
    model = build_model(model_config, tokenizers).to(selection.device)
    if freeze_embeddings:
        for parameter in model.source_embedding.parameters():
            parameter.requires_grad = False
        for parameter in model.target_embedding.parameters():
            parameter.requires_grad = False
    if freeze_encoder_layers > len(model.encoder_layers):
        raise ValueError(
            f"Cannot freeze {freeze_encoder_layers} encoder layers; model has "
            f"{len(model.encoder_layers)}."
        )
    for layer in model.encoder_layers[:freeze_encoder_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_loader, train_sampler = _data_loader(
        train_dataset, config.training, tokenizers, shuffle=True
    )
    validation_loader, _ = _data_loader(
        validation_dataset,
        config.training,
        tokenizers,
        shuffle=False,
        token_budget=config.evaluation.batch_tokens,
    )
    test_loader, _ = _data_loader(
        test_dataset, config.training, tokenizers, shuffle=False, token_budget=config.evaluation.batch_tokens
    )
    estimated_steps = max(
        1,
        math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
        * config.training.epochs,
    )
    scheduler = create_scheduler(optimizer, config.training, estimated_steps)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=selection.mixed_precision_enabled)
    except (AttributeError, TypeError):  # Compatibility with older supported PyTorch.
        scaler = torch.cuda.amp.GradScaler(enabled=selection.mixed_precision_enabled)
    checkpoint_directory = registry.root / "checkpoints" / version.version_id
    tracker = ExperimentTracker(experiment_id, paths.experiment(config.language_pair.id, experiment_id))
    if not resume_payload:
        tracker.start(
            {
                "model_version": version.version_id,
                "parent_version": version.parent_version,
                "language_pair": config.language_pair.id,
                "direction": direction,
                "dataset_version": dataset_version,
                "tokenizer_version": tokenizer_version,
                "model_configuration": model_config.model_dump(mode="json"),
                "training_configuration": config.training.model_dump(mode="json"),
                "environment": environment_manifest(paths.root),
                "dataset_sizes": {
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                    "test": len(test_dataset),
                    "overlength_skipped": train_dataset.skipped_overlength
                    + validation_dataset.skipped_overlength
                    + test_dataset.skipped_overlength,
                },
                "device": {
                    "type": str(selection.device),
                    "description": selection.description,
                    "precision": selection.precision,
                },
            }
        )
    if parent_version and not resume_payload:
        parent = registry.resolve(parent_version)
        parent_payload = load_checkpoint(paths.root / str(parent.checkpoint_path))
        if parent_payload["tokenizer_version"] != tokenizer_version:
            raise TokenizerCompatibilityError("Parent checkpoint tokenizer differs from parent registry.")
        model.load_state_dict(parent_payload["model_state"], strict=True)
    start_epoch = 0
    resume_batch = -1
    global_step = 0
    best_loss = math.inf
    validations_without_improvement = 0
    processed_tokens = 0
    processed_examples = 0
    if resume_payload:
        if resume_payload["tokenizer_version"] != tokenizer_version:
            raise TokenizerCompatibilityError("Resume tokenizer does not match checkpoint token IDs.")
        restore_training_state(resume_payload, model, optimizer, scheduler, scaler)
        start_epoch = int(resume_payload["epoch"])
        resume_batch = int(resume_payload.get("batch_in_epoch", -1))
        global_step = int(resume_payload["global_step"])
        best_loss = float(resume_payload.get("best_metric", math.inf))
        validations_without_improvement = int(
            resume_payload.get("early_stopping_state", {}).get(
                "validations_without_improvement", 0
            )
        )
        processed_tokens = int(resume_payload.get("processed_tokens", 0))
        processed_examples = int(resume_payload.get("processed_examples", 0))
    started = time.monotonic()
    last_validation_step = -1
    stop_early = False
    last_epoch, last_batch = start_epoch, resume_batch

    def state(epoch: int, batch_index: int) -> dict[str, Any]:
        return _checkpoint_state(
            effective_config,
            version,
            dataset_version,
            epoch,
            batch_index,
            global_step,
            best_loss,
            validations_without_improvement,
            processed_tokens,
            processed_examples,
        )

    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, config.training.epochs):
            last_epoch = epoch
            train_sampler.set_epoch(epoch)
            model.train()
            for batch_index, cpu_batch in enumerate(train_loader):
                last_batch = batch_index
                if epoch == start_epoch and batch_index <= resume_batch:
                    continue
                batch = _move_batch(cpu_batch, selection.device)
                with torch.autocast(
                    device_type="cuda", enabled=selection.mixed_precision_enabled
                ):
                    logits = model(batch.source_ids, batch.target_input_ids)
                    raw_loss = translation_loss(
                        logits,
                        batch.target_output_ids,
                        tokenizers.target.pad_id,
                        config.training.label_smoothing,
                    )
                    loss = raw_loss / config.training.gradient_accumulation_steps
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(
                        f"Non-finite loss {raw_loss.item()} at epoch {epoch + 1}, batch {batch_index}. "
                        "Reduce learning rate, inspect data, or disable mixed precision."
                    )
                scaler.scale(loss).backward()
                non_padding = int(batch.target_output_ids.ne(tokenizers.target.pad_id).sum().item())
                processed_tokens += non_padding + int(
                    batch.source_ids.ne(tokenizers.source.pad_id).sum().item()
                )
                processed_examples += batch.source_ids.size(0)
                should_step = (
                    (batch_index + 1) % config.training.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(train_loader)
                )
                if not should_step:
                    continue
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at optimizer step {global_step + 1}."
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                elapsed = time.monotonic() - started
                if global_step % config.training.log_every_steps == 0 or global_step == 1:
                    event = {
                        "event": "train",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "training_loss": float(raw_loss.item()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(gradient_norm.item()),
                        "tokens_processed": processed_tokens,
                        "examples_processed": processed_examples,
                        "elapsed_seconds": elapsed,
                        "estimated_progress": min(1.0, global_step / estimated_steps),
                        "device_memory": device_memory_megabytes(selection.device),
                    }
                    tracker.metric(event)
                    LOGGER.info(
                        "epoch=%d step=%d loss=%.5f lr=%.3g",
                        epoch + 1,
                        global_step,
                        raw_loss.item(),
                        optimizer.param_groups[0]["lr"],
                        extra={
                            "experiment_id": experiment_id,
                            "model_version": version.version_id,
                            "epoch": epoch + 1,
                            "step": global_step,
                        },
                    )
                if global_step % config.training.checkpoint_every_steps == 0:
                    periodic = checkpoint_directory / f"step-{global_step:08d}.pt"
                    save_checkpoint(periodic, model, optimizer, scheduler, scaler, state(epoch, batch_index))
                    atomic_copy(periodic, checkpoint_directory / "latest.pt")
                    apply_retention(
                        checkpoint_directory, config.training.keep_periodic_checkpoints
                    )
                    tracker.metric(
                        {"event": "checkpoint", "epoch": epoch + 1, "step": global_step, "path": _relative_to_root(periodic, paths)}
                    )
                if global_step % config.training.validate_every_steps == 0:
                    metrics, samples = evaluate_loader(
                        model,
                        validation_loader,
                        tokenizers,
                        selection.device,
                        config.training.label_smoothing,
                        config.evaluation.sample_count,
                        config.evaluation.maximum_generation_length,
                    )
                    last_validation_step = global_step
                    validation_loss = float(metrics["loss"])
                    tracker.metric({"event": "validation", "epoch": epoch + 1, "step": global_step} | metrics)
                    tracker.samples(global_step, samples)
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        validations_without_improvement = 0
                        save_checkpoint(
                            checkpoint_directory / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            state(epoch, batch_index),
                        )
                    else:
                        validations_without_improvement += 1
                    model.train()
                    if validations_without_improvement >= config.training.patience_validations:
                        stop_early = True
                if config.training.maximum_steps and global_step >= config.training.maximum_steps:
                    stop_early = True
                if stop_early:
                    break
            resume_batch = -1
            if global_step != last_validation_step:
                metrics, samples = evaluate_loader(
                    model,
                    validation_loader,
                    tokenizers,
                    selection.device,
                    config.training.label_smoothing,
                    config.evaluation.sample_count,
                    config.evaluation.maximum_generation_length,
                )
                validation_loss = float(metrics["loss"])
                tracker.metric({"event": "validation", "epoch": epoch + 1, "step": global_step} | metrics)
                tracker.samples(global_step, samples)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    validations_without_improvement = 0
                    save_checkpoint(
                        checkpoint_directory / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        state(epoch, last_batch),
                    )
                else:
                    validations_without_improvement += 1
                last_validation_step = global_step
            if stop_early:
                break
        # Final represents the exact final optimizer state; best remains available separately.
        final_checkpoint = checkpoint_directory / "final.pt"
        save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            state(last_epoch, last_batch),
        )
        atomic_copy(final_checkpoint, checkpoint_directory / "latest.pt")
        test_metrics, test_samples = evaluate_loader(
            model,
            test_loader,
            tokenizers,
            selection.device,
            config.training.label_smoothing,
            config.evaluation.sample_count,
            config.evaluation.maximum_generation_length,
        )
        tracker.samples(global_step + 1, test_samples)
        duration = time.monotonic() - started
        checkpoint_reference = _relative_to_root(final_checkpoint, paths)
        version = registry.update(
            version.version_id,
            status="candidate",
            checkpoint_path=checkpoint_reference,
            best_validation_score=best_loss,
            test_metrics={key: value for key, value in test_metrics.items() if isinstance(value, (int, float))},
            training_duration_seconds=duration,
        )
        tracker.finish(
            {
                "status": "candidate",
                "model_version": version.version_id,
                "checkpoint_path": checkpoint_reference,
                "best_validation_loss": best_loss,
                "test_metrics": test_metrics,
                "global_step": global_step,
                "duration_seconds": duration,
            }
        )
        return TrainingResult(
            version.version_id,
            experiment_id,
            final_checkpoint,
            best_loss,
            test_metrics,
            global_step,
            duration,
        )
    except KeyboardInterrupt:
        interrupted = checkpoint_directory / "interrupted.pt"
        save_checkpoint(
            interrupted,
            model,
            optimizer,
            scheduler,
            scaler,
            state(last_epoch, last_batch),
        )
        atomic_copy(interrupted, checkpoint_directory / "latest.pt")
        tracker.finish(
            {
                "status": "interrupted",
                "model_version": version.version_id,
                "resume_checkpoint": _relative_to_root(interrupted, paths),
                "global_step": global_step,
            }
        )
        LOGGER.warning("Training interrupted; resumable checkpoint saved to %s", interrupted)
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if selection.device.type == "cuda":
                torch.cuda.empty_cache()
            guidance = (
                "Out of memory during training. Resume from latest.pt after reducing "
                "training.batch_tokens, model.maximum_sequence_length, gradient accumulation "
                "microbatch size, or model dimensions. Original error: " + str(exc)
            )
            registry.update(version.version_id, status="failed", failure_reason=guidance)
            tracker.finish({"status": "failed", "failure_reason": guidance, "global_step": global_step})
            raise RuntimeError(guidance) from exc
        registry.update(version.version_id, status="failed", failure_reason=str(exc))
        tracker.finish({"status": "failed", "failure_reason": str(exc), "global_step": global_step})
        raise
    except Exception as exc:
        registry.update(version.version_id, status="failed", failure_reason=str(exc))
        tracker.finish({"status": "failed", "failure_reason": str(exc), "global_step": global_step})
        raise
