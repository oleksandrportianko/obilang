"""Randomly initialized bilingual encoder-decoder Transformer implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from nmt.config.schema import ModelConfig
from nmt.model.layers import DecoderLayer, EncoderLayer, SinusoidalPositionalEncoding
from nmt.tokenization.sentencepiece import TokenizerBundle


def causal_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Return a decoder mask of shape ``[length, length]``.

    ``True`` above the main diagonal means a query may not attend to that future
    key. The diagonal and past positions remain ``False`` and are visible.

    Raises:
        ValueError: If ``length`` is less than one.
    """
    if length < 1:
        raise ValueError("Causal mask length must be at least one.")
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


@dataclass(frozen=True)
class GenerationResult:
    """Autoregressive token sequences and normalized log-confidence scores."""

    token_ids: list[list[int]]
    scores: list[float]


class NMTTransformer(nn.Module):
    """Explicit batch-first Transformer for one bilingual translation direction."""

    def __init__(
        self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        source_pad_id: int,
        target_pad_id: int,
        config: ModelConfig,
    ) -> None:
        """Create a model with random trainable weights.

        Args:
            source_vocabulary_size: Number of source tokenizer IDs.
            target_vocabulary_size: Number of target tokenizer IDs.
            source_pad_id: Source ID excluded from attention.
            target_pad_id: Target ID excluded from attention and loss.
            config: Validated architecture and initialization settings.

        Raises:
            ValueError: If shared embeddings are requested for unequal vocabularies.
        """
        super().__init__()
        self.config = config
        # Plain scalar mirrors keep the forward path graph-export-compatible while
        # the full Pydantic configuration remains available to Python services.
        self.embedding_scale = math.sqrt(config.embedding_dimension)
        self.maximum_sequence_length = config.maximum_sequence_length
        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size
        self.source_pad_id = source_pad_id
        self.target_pad_id = target_pad_id
        dimension = config.embedding_dimension
        self.source_embedding = nn.Embedding(
            source_vocabulary_size, dimension, padding_idx=source_pad_id
        )
        if config.share_source_target_embeddings:
            if source_vocabulary_size != target_vocabulary_size:
                raise ValueError(
                    "Shared source/target embeddings require equal tokenizer vocabularies. "
                    "Disable model.share_source_target_embeddings for separate tokenizers."
                )
            self.target_embedding = self.source_embedding
        else:
            self.target_embedding = nn.Embedding(
                target_vocabulary_size, dimension, padding_idx=target_pad_id
            )
        self.source_positions = SinusoidalPositionalEncoding(
            dimension, config.maximum_sequence_length, config.dropout
        )
        self.target_positions = SinusoidalPositionalEncoding(
            dimension, config.maximum_sequence_length, config.dropout
        )
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(
                dimension,
                config.attention_heads,
                config.feedforward_dimension,
                config.dropout,
            )
            for _ in range(config.encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(
                dimension,
                config.attention_heads,
                config.feedforward_dimension,
                config.dropout,
            )
            for _ in range(config.decoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(dimension)
        self.decoder_norm = nn.LayerNorm(dimension)
        self.vocabulary_projection = nn.Linear(dimension, target_vocabulary_size, bias=False)
        self._reset_parameters(config.initialization)
        if config.tied_target_embeddings:
            self.vocabulary_projection.weight = self.target_embedding.weight

    def _reset_parameters(self, strategy: str) -> None:
        """Apply the configured random initialization to matrix parameters."""
        for parameter in self.parameters():
            if parameter.dim() <= 1:
                continue
            if strategy == "xavier_uniform":
                nn.init.xavier_uniform_(parameter)
            elif strategy == "xavier_normal":
                nn.init.xavier_normal_(parameter)
            elif strategy == "kaiming_uniform":
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        # Padding vectors remain exact zeros and contribute no learned signal.
        with torch.no_grad():
            self.source_embedding.weight[self.source_pad_id].zero_()
            self.target_embedding.weight[self.target_pad_id].zero_()

    def encode(self, source_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode source IDs into contextual memory.

        Args:
            source_ids: Integer tensor ``[batch, source_length]``.

        Returns:
            Encoder memory ``[batch, source_length, model_dim]`` and boolean source
            padding mask ``[batch, source_length]``.
        """
        if source_ids.ndim != 2:
            raise ValueError(f"source_ids must have shape [batch, length], got {source_ids.shape}.")
        padding = source_ids.eq(self.source_pad_id)
        hidden = self.source_positions(self.source_embedding(source_ids) * self.embedding_scale)
        for layer in self.encoder_layers:
            hidden = layer(hidden, padding)
        return self.encoder_norm(hidden), padding

    def decode(
        self, target_input_ids: Tensor, encoder_output: Tensor, source_padding_mask: Tensor
    ) -> Tensor:
        """Decode shifted target IDs to vocabulary logits.

        Args:
            target_input_ids: BOS-prefixed teacher/generation IDs ``[batch, target_length]``.
            encoder_output: Encoder memory ``[batch, source_length, model_dim]``.
            source_padding_mask: Source PAD mask ``[batch, source_length]``.

        Returns:
            Unnormalized logits ``[batch, target_length, target_vocabulary_size]``.
        """
        if target_input_ids.ndim != 2:
            raise ValueError(
                f"target_input_ids must have shape [batch, length], got {target_input_ids.shape}."
            )
        target_padding = target_input_ids.eq(self.target_pad_id)
        hidden = self.target_positions(
            self.target_embedding(target_input_ids) * self.embedding_scale
        )
        future_mask = causal_mask(target_input_ids.size(1), target_input_ids.device)
        for layer in self.decoder_layers:
            hidden = layer(
                hidden,
                encoder_output,
                future_mask,
                target_padding,
                source_padding_mask,
            )
        return self.vocabulary_projection(self.decoder_norm(hidden))

    def forward(self, source_ids: Tensor, target_input_ids: Tensor) -> Tensor:
        """Run teacher-forced translation and return per-position vocabulary logits.

        Tensor shapes are ``source_ids=[B,S]``, ``target_input_ids=[B,T]``, and
        output ``[B,T,V_target]``. Targets are shifted by the caller so position
        `t` predicts the next token rather than seeing it through the input.
        """
        memory, source_padding = self.encode(source_ids)
        return self.decode(target_input_ids, memory, source_padding)

    @torch.inference_mode()
    def greedy_generate(
        self,
        source_ids: Tensor,
        bos_id: int,
        eos_id: int,
        maximum_length: int,
    ) -> GenerationResult:
        """Generate a batch by repeatedly selecting the highest-probability token.

        Args:
            source_ids: Padded source tensor ``[batch, source_length]``.
            bos_id: First target token for every sequence.
            eos_id: Token that marks an item complete.
            maximum_length: Maximum output length including BOS.

        Returns:
            Generated IDs and mean selected-token log probabilities per sequence.
        """
        if maximum_length > self.maximum_sequence_length:
            raise ValueError(
                f"maximum_length {maximum_length} exceeds configured model limit "
                f"{self.maximum_sequence_length}."
            )
        self.eval()
        memory, source_padding = self.encode(source_ids)
        batch_size = source_ids.size(0)
        generated = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=source_ids.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)
        score_sums = torch.zeros(batch_size, dtype=torch.float32, device=source_ids.device)
        score_counts = torch.zeros(batch_size, dtype=torch.float32, device=source_ids.device)
        for _ in range(maximum_length - 1):
            logits = self.decode(generated, memory, source_padding)[:, -1]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            selected_scores, next_ids = log_probabilities.max(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            active = ~finished
            score_sums += torch.where(active, selected_scores, torch.zeros_like(selected_scores))
            score_counts += active.float()
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        scores = score_sums / score_counts.clamp_min(1)
        return GenerationResult(generated.cpu().tolist(), scores.cpu().tolist())

    @torch.inference_mode()
    def beam_generate(
        self,
        source_ids: Tensor,
        bos_id: int,
        eos_id: int,
        maximum_length: int,
        beam_width: int = 4,
        length_penalty: float = 0.6,
    ) -> GenerationResult:
        """Generate each batch item with standard left-to-right beam search.

        Beam hypotheses store cumulative log probability. Ranking divides by
        ``((5 + length) / 6) ** length_penalty`` to avoid an excessive preference
        for short sentences. This straightforward implementation favors clarity
        over decoder key/value caching.
        """
        if beam_width < 1:
            raise ValueError("beam_width must be at least one.")
        if beam_width == 1:
            return self.greedy_generate(source_ids, bos_id, eos_id, maximum_length)
        if maximum_length > self.maximum_sequence_length:
            raise ValueError("Beam maximum_length exceeds the model sequence limit.")
        self.eval()
        all_ids: list[list[int]] = []
        all_scores: list[float] = []
        for item_index in range(source_ids.size(0)):
            memory, padding = self.encode(source_ids[item_index : item_index + 1])
            beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]
            for _ in range(maximum_length - 1):
                candidates: list[tuple[list[int], float, bool]] = []
                for tokens, score, finished in beams:
                    if finished:
                        candidates.append((tokens, score, True))
                        continue
                    token_tensor = torch.tensor([tokens], device=source_ids.device)
                    log_probabilities = torch.log_softmax(
                        self.decode(token_tensor, memory, padding)[0, -1], dim=-1
                    )
                    top_scores, top_ids = torch.topk(log_probabilities, beam_width)
                    for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                        candidates.append(
                            (tokens + [token_id], score + token_score, token_id == eos_id)
                        )
                def normalized(hypothesis: tuple[list[int], float, bool]) -> float:
                    length_factor = ((5 + max(1, len(hypothesis[0]) - 1)) / 6) ** length_penalty
                    return hypothesis[1] / length_factor
                beams = sorted(candidates, key=normalized, reverse=True)[:beam_width]
                if all(finished for _, _, finished in beams):
                    break
            best = max(beams, key=lambda hypothesis: hypothesis[1] / (((5 + max(1, len(hypothesis[0]) - 1)) / 6) ** length_penalty))
            all_ids.append(best[0])
            all_scores.append(best[1] / max(1, len(best[0]) - 1))
        return GenerationResult(all_ids, all_scores)


def build_model(config: ModelConfig, tokenizers: TokenizerBundle) -> NMTTransformer:
    """Build a randomly initialized Transformer compatible with tokenizer IDs."""
    if not tokenizers.shared and config.share_source_target_embeddings:
        raise ValueError(
            "Separate source/target tokenizers have unrelated token IDs and cannot share "
            "embeddings. Set model.share_source_target_embeddings=false."
        )
    return NMTTransformer(
        tokenizers.source.vocabulary_size,
        tokenizers.target.vocabulary_size,
        tokenizers.source.pad_id,
        tokenizers.target.pad_id,
        config,
    )
