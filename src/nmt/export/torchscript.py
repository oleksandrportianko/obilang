"""Backward-compatible import for the platform's current portable exporter."""

from nmt.export.portable import export_model

# Kept for callers of the 0.1 development API. New code should use export_model.
export_torchscript = export_model

__all__ = ["export_torchscript"]
