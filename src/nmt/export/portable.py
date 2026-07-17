"""Portable dynamic-shape PyTorch inference export with exact tokenizer artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from nmt.config.schema import PlatformConfig
from nmt.inference.service import load_runtime
from nmt.utils.io import atomic_write_json
from nmt.utils.paths import ProjectPaths


def export_model(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    version: str,
) -> Path:
    """Export a registered dynamic-shape forward graph and exact tokenizers.

    `torch.export` records tensor computation without optimizer/checkpoint pickle
    state. Source/target lengths and batch size are dynamic up to the configured
    model maximum. Consumers implement autoregressive selection using the logits,
    BOS/EOS IDs, and manifest; the platform CLI/API already supply both decoders.

    Returns:
        Versioned export directory containing `model.pt2`, tokenizer models, and
        a JSON manifest. The saved program is loaded and executed before success.
    """
    runtime = load_runtime(config, paths, direction, version, device="cpu")
    destination = (
        paths.model_direction(config.language_pair.id, direction)
        / "exports"
        / runtime.version.version_id
    )
    destination.mkdir(parents=True, exist_ok=True)
    source = torch.tensor(
        [
            [
                runtime.tokenizers.source.bos_id,
                runtime.tokenizers.source.eos_id,
                runtime.tokenizers.source.pad_id,
            ],
            [
                runtime.tokenizers.source.bos_id,
                runtime.tokenizers.source.unknown_id,
                runtime.tokenizers.source.eos_id,
            ],
        ],
        dtype=torch.long,
    )
    target = torch.tensor(
        [
            [runtime.tokenizers.target.bos_id, runtime.tokenizers.target.eos_id],
            [runtime.tokenizers.target.bos_id, runtime.tokenizers.target.unknown_id],
        ],
        dtype=torch.long,
    )
    runtime.model.eval()
    maximum = runtime.model.maximum_sequence_length
    batch_dimension = torch.export.Dim("batch", min=1, max=1024)
    source_dimension = torch.export.Dim("source_length", min=2, max=maximum)
    target_dimension = torch.export.Dim("target_length", min=1, max=maximum)
    exported = torch.export.export(
        runtime.model,
        (source, target),
        dynamic_shapes=(
            {0: batch_dimension, 1: source_dimension},
            {0: batch_dimension, 1: target_dimension},
        ),
    )
    model_path = destination / "model.pt2"
    torch.export.save(exported, model_path)
    loaded = torch.export.load(model_path).module()
    verification_source = source[:1, :2]
    verification_target = target[:1, :1]
    output = loaded(verification_source, verification_target)
    if output.shape != (1, 1, runtime.tokenizers.target.vocabulary_size):
        raise RuntimeError(
            f"Export verification produced unexpected shape {tuple(output.shape)}."
        )
    copied: dict[str, str] = {}
    for side, tokenizer in (
        ("source", runtime.tokenizers.source),
        ("target", runtime.tokenizers.target),
    ):
        name = f"{side}-{tokenizer.path.name}"
        target_path = destination / name
        if not target_path.exists():
            shutil.copyfile(tokenizer.path, target_path)
        copied[side] = name
    manifest: dict[str, Any] = {
        "format": "torch-export-pt2-v1",
        "model_file": model_path.name,
        "dynamic_shapes": {
            "batch": [1, 1024],
            "source_length": [2, maximum],
            "target_length": [1, maximum],
        },
        "model_version": runtime.version.version_id,
        "language_pair": config.language_pair.id,
        "direction": direction,
        "model_configuration": runtime.version.model_configuration,
        "tokenizer_version": runtime.version.tokenizer_version,
        "tokenizer_models": copied,
        "special_token_ids": {
            "source_pad": runtime.tokenizers.source.pad_id,
            "source_bos": runtime.tokenizers.source.bos_id,
            "source_eos": runtime.tokenizers.source.eos_id,
            "target_pad": runtime.tokenizers.target.pad_id,
            "target_bos": runtime.tokenizers.target.bos_id,
            "target_eos": runtime.tokenizers.target.eos_id,
        },
        "security": (
            "Inference graph only. PyTorch training checkpoints use pickle and must "
            "only be loaded from trusted sources."
        ),
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return destination
