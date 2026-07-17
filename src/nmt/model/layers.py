"""Educational pre-normalized Transformer encoder and decoder layers."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sine/cosine positions to batch-first token embeddings.

    Positions are a non-trainable buffer with shape ``[1, max_length, model_dim]``.
    This lets the model distinguish token order without pretrained parameters.
    """

    def __init__(self, model_dimension: int, maximum_length: int, dropout: float) -> None:
        """Construct the positional table.

        Args:
            model_dimension: Embedding width of every token vector.
            maximum_length: Largest supported encoded or generated sequence.
            dropout: Probability applied after token/position addition.
        """
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, model_dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / model_dimension)
        )
        encoding = torch.zeros(maximum_length, model_dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        # Odd model dimensions contain one fewer cosine channel.
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Add positions to embeddings of shape ``[batch, length, model_dim]``.

        Raises:
            ValueError: If sequence length exceeds the configured table.
        """
        sequence_length = embeddings.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds model maximum {self.encoding.size(1)}."
            )
        return self.dropout(embeddings + self.encoding[:, :sequence_length].to(embeddings.dtype))


class FeedForward(nn.Module):
    """Position-wise two-layer GELU projection used after attention."""

    def __init__(
        self, model_dimension: int, feedforward_dimension: int, dropout: float
    ) -> None:
        """Create independent projections applied to each sequence position."""
        super().__init__()
        self.input_projection = nn.Linear(model_dimension, feedforward_dimension)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(feedforward_dimension, model_dimension)

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform ``[batch, length, model_dim]`` without changing its shape."""
        return self.output_projection(self.dropout(self.activation(self.input_projection(inputs))))


class EncoderLayer(nn.Module):
    """Pre-normalized self-attention and feed-forward encoder block."""

    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:
        """Initialize randomly weighted encoder sublayers and residual dropout."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dimension)
        self.self_attention = nn.MultiheadAttention(
            model_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dimension)
        self.feedforward = FeedForward(model_dimension, feedforward_dimension, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, padding_mask: Tensor | None) -> Tensor:
        """Encode a hidden sequence.

        Args:
            inputs: Float tensor ``[batch, source_length, model_dim]``.
            padding_mask: Boolean ``[batch, source_length]`` where ``True`` marks
                padded keys that no real token may attend to.

        Returns:
            Tensor with the same shape as ``inputs``.
        """
        normalized = self.attention_norm(inputs)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = inputs + self.residual_dropout(attended)
        return hidden + self.residual_dropout(self.feedforward(self.feedforward_norm(hidden)))


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder cross-attention, and feed-forward block."""

    def __init__(
        self,
        model_dimension: int,
        attention_heads: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:
        """Initialize one pre-normalized autoregressive decoder block."""
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(model_dimension)
        self.self_attention = nn.MultiheadAttention(
            model_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(model_dimension)
        self.cross_attention = nn.MultiheadAttention(
            model_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(model_dimension)
        self.feedforward = FeedForward(model_dimension, feedforward_dimension, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: Tensor,
        encoder_output: Tensor,
        causal_mask: Tensor,
        target_padding_mask: Tensor | None,
        source_padding_mask: Tensor | None,
    ) -> Tensor:
        """Decode one hidden sequence against encoder memory.

        Args:
            inputs: Target hidden states ``[batch, target_length, model_dim]``.
            encoder_output: Source memory ``[batch, source_length, model_dim]``.
            causal_mask: Boolean square mask where upper-triangle ``True`` entries
                prevent a target position from reading future teacher-forced tokens.
            target_padding_mask: Boolean ``[batch, target_length]`` PAD mask.
            source_padding_mask: Boolean ``[batch, source_length]`` PAD mask.

        Returns:
            Updated target hidden states with the same shape as ``inputs``.
        """
        normalized = self.self_attention_norm(inputs)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )
        hidden = inputs + self.residual_dropout(attended)
        query = self.cross_attention_norm(hidden)
        crossed, _ = self.cross_attention(
            query,
            encoder_output,
            encoder_output,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        hidden = hidden + self.residual_dropout(crossed)
        return hidden + self.residual_dropout(self.feedforward(self.feedforward_norm(hidden)))
