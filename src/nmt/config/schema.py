"""Pydantic schemas for every configurable platform subsystem."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base schema that rejects misspelled or obsolete configuration keys."""

    model_config = ConfigDict(extra="forbid")


class LanguagePairConfig(StrictModel):
    """Language identities and supported directional models."""

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    source_language: str = Field(min_length=2, max_length=16)
    target_language: str = Field(min_length=2, max_length=16)
    bidirectional: bool = True

    def directions(self) -> list[str]:
        """Return canonical direction IDs configured for this pair."""
        forward = f"{self.source_language}-to-{self.target_language}"
        reverse = f"{self.target_language}-to-{self.source_language}"
        return [forward, reverse] if self.bidirectional else [forward]

    def languages_for_direction(self, direction: str) -> tuple[str, str]:
        """Resolve source/target language codes for a canonical direction.

        Raises:
            ValueError: If the direction is not enabled for this pair.
        """
        if direction not in self.directions():
            raise ValueError(
                f"Direction {direction!r} is not configured for {self.id}; "
                f"choose one of {', '.join(self.directions())}."
            )
        source, target = direction.split("-to-", maxsplit=1)
        return source, target


class SplitConfig(StrictModel):
    """Stable hash-based dataset split proportions."""

    train: float = Field(default=0.9, gt=0, lt=1)
    validation: float = Field(default=0.05, gt=0, lt=1)
    test: float = Field(default=0.05, gt=0, lt=1)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> "SplitConfig":
        """Require split fractions to sum to one within float tolerance."""
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-8):
            raise ValueError(f"Dataset split fractions must sum to 1.0, received {total}.")
        return self


class DataConfig(StrictModel):
    """Normalization, quality filtering, and split behavior."""

    unicode_normalization: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFC"
    minimum_characters: int = Field(default=1, ge=1)
    maximum_characters: int = Field(default=1000, ge=1)
    maximum_tokens_approximation: int = Field(default=256, ge=2)
    maximum_length_ratio: float = Field(default=3.0, ge=1.0)
    detect_language_mismatch: bool = False
    source_scripts: list[Literal["Latn", "Cyrl", "Grek", "Arab", "Hebr", "Deva"]] = Field(
        default_factory=list
    )
    target_scripts: list[Literal["Latn", "Cyrl", "Grek", "Arab", "Hebr", "Deva"]] = Field(
        default_factory=list
    )
    reject_malformed_markup: bool = True
    split: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def validate_lengths(self) -> "DataConfig":
        """Ensure the configured maximum can contain the minimum."""
        if self.maximum_characters < self.minimum_characters:
            raise ValueError("maximum_characters must be at least minimum_characters.")
        return self


class TokenizerConfig(StrictModel):
    """Immutable SentencePiece training settings."""

    type: Literal["sentencepiece_bpe", "sentencepiece_unigram"] = "sentencepiece_bpe"
    shared: bool = True
    vocabulary_size: int = Field(default=24000, ge=32)
    character_coverage: float = Field(default=1.0, gt=0.0, le=1.0)
    byte_fallback: bool = False
    pad_token: str = "<PAD>"
    unknown_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"


class ModelConfig(StrictModel):
    """Architecture and random parameter initialization settings."""

    embedding_dimension: int = Field(default=384, ge=16)
    attention_heads: int = Field(default=6, ge=1)
    encoder_layers: int = Field(default=4, ge=1)
    decoder_layers: int = Field(default=4, ge=1)
    feedforward_dimension: int = Field(default=1536, ge=16)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    maximum_sequence_length: int = Field(default=256, ge=4)
    tied_target_embeddings: bool = True
    share_source_target_embeddings: bool = True
    initialization: Literal["xavier_uniform", "xavier_normal", "kaiming_uniform"] = (
        "xavier_uniform"
    )

    @model_validator(mode="after")
    def validate_attention_shape(self) -> "ModelConfig":
        """Require each attention head to receive an equal-width projection."""
        if self.embedding_dimension % self.attention_heads:
            raise ValueError(
                "embedding_dimension must be divisible by attention_heads; received "
                f"{self.embedding_dimension} and {self.attention_heads}."
            )
        return self


class TrainingConfig(StrictModel):
    """Optimization, validation, checkpointing, and device settings."""

    batch_tokens: int = Field(default=4096, ge=16)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=4000, ge=0)
    scheduler: Literal["inverse_sqrt", "cosine", "constant"] = "inverse_sqrt"
    epochs: int = Field(default=20, ge=1)
    label_smoothing: float = Field(default=0.1, ge=0, lt=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    mixed_precision: bool = True
    checkpoint_every_steps: int = Field(default=1000, ge=1)
    validate_every_steps: int = Field(default=1000, ge=1)
    keep_periodic_checkpoints: int = Field(default=3, ge=0)
    patience_validations: int = Field(default=8, ge=1)
    seed: int = Field(default=42, ge=0)
    deterministic: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_workers: int = Field(default=0, ge=0)
    log_every_steps: int = Field(default=10, ge=1)
    maximum_steps: int | None = Field(default=None, ge=1)


class FineTuningConfig(StrictModel):
    """Continued-training and historical replay controls."""

    learning_rate: float = Field(default=3e-5, gt=0)
    epochs: int = Field(default=5, ge=1)
    replay_enabled: bool = True
    replay_strategy: Literal[
        "percentage", "fixed_count", "balanced", "weighted", "domain_aware"
    ] = "balanced"
    new_data_ratio: float = Field(default=0.5, ge=0, le=1)
    historical_data_ratio: float = Field(default=0.5, ge=0, le=1)
    historical_example_count: int | None = Field(default=None, ge=1)
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_replay_ratios(self) -> "FineTuningConfig":
        """Validate sampling ratios for replay strategies that use them."""
        if self.replay_enabled and self.new_data_ratio + self.historical_data_ratio <= 0:
            raise ValueError("Replay requires a non-zero new or historical data ratio.")
        if self.replay_strategy == "fixed_count" and self.historical_example_count is None:
            raise ValueError("fixed_count replay requires historical_example_count.")
        return self


class EvaluationConfig(StrictModel):
    """Automatic corpus and preservation metrics."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "bleu",
            "chrf",
            "perplexity",
            "exact_match",
            "number_accuracy",
            "punctuation_accuracy",
            "placeholder_accuracy",
            "markup_accuracy",
        ]
    )
    beam_width: int = Field(default=4, ge=1)
    length_penalty: float = Field(default=0.6, ge=0)
    sample_count: int = Field(default=20, ge=0)
    batch_tokens: int | None = Field(default=None, ge=16)
    maximum_generation_length: int | None = Field(default=None, ge=4)


class PromotionConfig(StrictModel):
    """Required metric deltas for automatic candidate approval."""

    minimum_chrf_change: float = 0.0
    maximum_bleu_regression: float = Field(default=0.3, ge=0)
    maximum_number_accuracy_regression: float = Field(default=0.0, ge=0)
    maximum_placeholder_accuracy_regression: float = Field(default=0.0, ge=0)


class PlatformConfig(StrictModel):
    """Complete validated configuration for one language pair."""

    language_pair: LanguagePairConfig
    data: DataConfig = Field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    fine_tuning: FineTuningConfig = Field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
