"""Cross-platform accelerator selection and precision reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class DeviceUnavailableError(RuntimeError):
    """Raised when a manually requested accelerator is unavailable."""


@dataclass(frozen=True)
class DeviceSelection:
    """Selected torch device and effective numerical precision mode."""

    device: torch.device
    precision: str
    mixed_precision_enabled: bool
    description: str


def select_device(requested: str = "auto", mixed_precision: bool = True) -> DeviceSelection:
    """Select CUDA, MPS, or CPU and report the effective precision.

    Args:
        requested: `auto`, `cpu`, `cuda`, or `mps`.
        mixed_precision: Request lower-precision operations where safely supported.

    Returns:
        Device, precision label, and human-readable hardware description.

    Raises:
        DeviceUnavailableError: If a manual CUDA/MPS request cannot be honored.
    """
    cuda_available = torch.cuda.is_available()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if requested == "auto":
        name = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    else:
        name = requested
    if name == "cuda" and not cuda_available:
        raise DeviceUnavailableError(
            "CUDA was requested but torch.cuda.is_available() is false. Install a CUDA-enabled "
            "PyTorch build and driver, or use --device cpu."
        )
    if name == "mps" and not mps_available:
        raise DeviceUnavailableError(
            "MPS was requested but this PyTorch/macOS combination does not expose MPS. Use CPU."
        )
    device = torch.device(name)
    if name == "cuda":
        effective_mixed = mixed_precision
        precision = "float16 (AMP)" if effective_mixed else "float32"
        description = torch.cuda.get_device_name(device)
    elif name == "mps":
        # MPS autocast support varies by PyTorch/macOS version; FP32 is the portable path.
        effective_mixed = False
        precision = "float32"
        description = "Apple Metal Performance Shaders"
    else:
        effective_mixed = False
        precision = "float32"
        description = "CPU"
    return DeviceSelection(device, precision, effective_mixed, description)


def device_memory_megabytes(device: torch.device) -> dict[str, float]:
    """Return best-effort allocated/reserved memory counters for monitoring."""
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "maximum_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return {"allocated_mb": torch.mps.current_allocated_memory() / 1024**2}
    try:
        import psutil

        process = psutil.Process()
        return {"resident_mb": process.memory_info().rss / 1024**2}
    except (ImportError, OSError):
        return {}
