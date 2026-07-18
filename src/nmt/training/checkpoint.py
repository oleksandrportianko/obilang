"""Atomic trusted checkpoint serialization with full training and RNG state."""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointError(RuntimeError):
    """Raised when checkpoint bytes are missing, corrupted, or incompatible."""


def capture_rng_state() -> dict[str, Any]:
    """Capture generators needed to continue stochastic training closely."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore Python, NumPy, CPU, and available CUDA generator states."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    state: dict[str, Any],
) -> None:
    """Atomically save model/optimizer/scheduler/scaler, progress, and RNG state.

    Args:
        path: Trusted local `.pt` destination.
        model: Directional Transformer whose parameter state is serialized.
        optimizer: AdamW state including momentum and parameter steps.
        scheduler: Learning-rate schedule position.
        scaler: CUDA gradient scaler or ``None`` for full precision.
        state: JSON-like training progress/config/artifact references. It must
            include epoch, global step, best metric, early-stopping state,
            tokenizer version, and dataset version.

    Side effects:
        Creates a temporary file beside `path`, fsyncs it, and atomically replaces
        the destination. PyTorch serialization uses pickle; only load trusted files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    # AdamW moments and temporary atomic coexistence can require several times the
    # raw parameter size. A 64 MiB floor catches nearly-full disks even for toy models.
    estimated_required = max(64 * 1024**2, parameter_bytes * 5)
    free_bytes = shutil.disk_usage(path.parent).free
    if free_bytes < estimated_required:
        raise CheckpointError(
            f"Insufficient disk space for checkpoint {path}: approximately "
            f"{estimated_required / 1024**2:.1f} MiB is required but only "
            f"{free_bytes / 1024**2:.1f} MiB is free. Free space or change the "
            "artifact location before resuming."
        )
    payload = dict(state)
    payload.update(
        {
            "checkpoint_schema": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "rng_state": capture_rng_state(),
        }
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        # Windows requires a writable handle for fsync; a read-only descriptor
        # raises OSError(9, "Bad file descriptor") there.
        with temporary.open("r+b") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, path)
    except (OSError, RuntimeError) as exc:
        raise CheckpointError(f"Could not atomically save checkpoint {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    """Load and minimally validate a trusted PyTorch checkpoint.

    Warning:
        PyTorch checkpoint files may execute Python code during unpickling. Never
        load a checkpoint obtained from an untrusted source.

    Raises:
        CheckpointError: If the file is absent, unreadable, or lacks required keys.
    """
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    try:
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:  # PyTorch before the weights_only keyword.
            payload = torch.load(path, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(
            f"Checkpoint {path} is corrupted, incompatible, or untrusted: {exc}"
        ) from exc
    required = {
        "checkpoint_schema",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "epoch",
        "global_step",
        "model_configuration",
        "training_configuration",
        "tokenizer_version",
        "dataset_version",
    }
    missing = required.difference(payload) if isinstance(payload, dict) else required
    if missing:
        raise CheckpointError(f"Checkpoint {path} is missing required fields: {sorted(missing)}")
    return payload


def restore_training_state(
    payload: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Restore trainable and stochastic state after compatibility checks by caller."""
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    restore_rng_state(payload["rng_state"])


def apply_retention(directory: Path, keep: int) -> None:
    """Retain the newest periodic checkpoints while never touching named anchors."""
    periodic = sorted(directory.glob("step-*.pt"), key=lambda item: item.stat().st_mtime)
    for stale in periodic[: max(0, len(periodic) - keep)]:
        stale.unlink()
