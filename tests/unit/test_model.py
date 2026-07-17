"""Mask, attention flow, tensor shape, generation, and loss tests."""

from __future__ import annotations

import torch

from nmt.config.schema import ModelConfig
from nmt.model.loss import translation_loss
from nmt.model.transformer import NMTTransformer, causal_mask


def tiny_model() -> NMTTransformer:
    """Return a deterministic-shape random model for structural tests."""
    return NMTTransformer(
        32,
        32,
        0,
        0,
        ModelConfig(
            embedding_dimension=16,
            attention_heads=4,
            encoder_layers=2,
            decoder_layers=2,
            feedforward_dimension=32,
            dropout=0,
            maximum_sequence_length=12,
        ),
    )


def test_causal_mask_hides_only_future_positions() -> None:
    """Decoder positions see themselves and history but never teacher-forced future."""
    mask = causal_mask(4)
    assert mask.dtype == torch.bool
    assert mask.shape == (4, 4)
    assert not mask[2, 1]
    assert not mask[2, 2]
    assert mask[2, 3]


def test_encoder_decoder_and_projection_dimensions() -> None:
    """Explicit stacks preserve batch/length and project to target vocabulary."""
    model = tiny_model()
    source = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 3, 0, 0]])
    target = torch.tensor([[2, 8, 9], [2, 10, 0]])
    memory, padding = model.encode(source)
    logits = model.decode(target, memory, padding)
    assert memory.shape == (2, 5, 16)
    assert padding.shape == (2, 5)
    assert logits.shape == (2, 3, 32)
    assert model(source, target).shape == (2, 3, 32)


def test_loss_ignores_padding_labels() -> None:
    """Adding an ignored PAD position cannot change mean cross entropy."""
    logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    labels = torch.tensor([[0, 1]])
    first = translation_loss(logits[:, 1:], labels[:, 1:], padding_id=0)
    second = translation_loss(logits, labels, padding_id=0)
    assert torch.allclose(first, second)


def test_greedy_generation_is_autoregressive_and_bounded() -> None:
    """Random weights still produce a correctly shaped bounded token sequence."""
    model = tiny_model()
    result = model.greedy_generate(torch.tensor([[2, 5, 3]]), bos_id=2, eos_id=3, maximum_length=6)
    assert result.token_ids[0][0] == 2
    assert 2 <= len(result.token_ids[0]) <= 6
    assert len(result.scores) == 1
