"""Padding-aware teacher-forced translation loss."""

from __future__ import annotations

from torch import Tensor, nn


def translation_loss(
    logits: Tensor,
    target_ids: Tensor,
    padding_id: int,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Calculate mean cross entropy over non-padding target positions.

    Args:
        logits: Model scores ``[batch, target_length, vocabulary]``.
        target_ids: Next-token labels ``[batch, target_length]``.
        padding_id: Label ignored by reduction.
        label_smoothing: Probability mass spread across non-target classes.

    Returns:
        Scalar differentiable loss. PyTorch averages only non-ignored positions.

    Raises:
        ValueError: If batch/sequence dimensions do not match.
    """
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            f"Logit positions {logits.shape[:2]} do not match target shape {target_ids.shape}."
        )
    criterion = nn.CrossEntropyLoss(
        ignore_index=padding_id,
        label_smoothing=label_smoothing,
    )
    return criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
