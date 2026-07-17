"""Random-seed and environment capture utilities."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Non-negative experiment seed.
        deterministic: Request deterministic PyTorch algorithms. Unsupported
            kernels then raise instead of silently becoming nondeterministic.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)


def git_commit(root: Path) -> str | None:
    """Return the current Git commit without changing repository state."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(root: Path) -> dict[str, Any]:
    """Capture runtime versions needed to interpret a model artifact.

    Args:
        root: Repository used for source-control metadata.

    Returns:
        JSON-compatible system, Python, PyTorch, device, and Git metadata.
    """
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "git_commit": git_commit(root),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "process_id": os.getpid(),
    }
