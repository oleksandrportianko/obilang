"""Trusted local model loading plus greedy and beam translation APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from nmt.config.schema import ModelConfig, PlatformConfig
from nmt.model.transformer import NMTTransformer, build_model
from nmt.registry.local import LocalModelRegistry, ModelVersion
from nmt.tokenization.sentencepiece import TokenizerBundle, TokenizerCompatibilityError
from nmt.training.checkpoint import load_checkpoint
from nmt.training.device import DeviceSelection, select_device
from nmt.training.trainer import directional_tokenizers
from nmt.utils.paths import ProjectPaths


@dataclass(frozen=True)
class TranslationResult:
    """One decoded sentence with version, token, timing, and confidence metadata."""

    translation: str
    model_version: str
    token_count: int
    inference_time_ms: float
    confidence: float
    source_token_ids: list[int]
    source_pieces: list[str]
    output_token_ids: list[int]


@dataclass
class TranslationRuntime:
    """Loaded directional model, tokenizer bundle, device, and registry metadata."""

    model: NMTTransformer
    tokenizers: TokenizerBundle
    version: ModelVersion
    selection: DeviceSelection

    def translate_batch(
        self,
        texts: list[str],
        *,
        decoding: str = "beam",
        beam_width: int = 4,
        length_penalty: float = 0.6,
        maximum_length: int | None = None,
    ) -> list[TranslationResult]:
        """Translate one non-empty text batch on the loaded local device.

        Args:
            texts: Source sentences. Empty or whitespace-only text is rejected.
            decoding: `greedy` or `beam`.
            beam_width: Active hypotheses per item for beam search.
            length_penalty: Beam score normalization exponent.
            maximum_length: Output cap including BOS; defaults to model maximum.

        Returns:
            Results in input order. Per-item time is total batch time divided by
            batch size because accelerator execution is batched.

        Raises:
            ValueError: For empty input, unknown decoding mode, or overlength source.
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Translation text must contain at least one non-whitespace character.")
        encoded = [self.tokenizers.source.encode(text) for text in texts]
        model_limit = self.model.config.maximum_sequence_length
        overlength = [index for index, ids in enumerate(encoded) if len(ids) > model_limit]
        if overlength:
            raise ValueError(
                f"Source item {overlength[0]} encodes to {len(encoded[overlength[0]])} tokens, "
                f"above model limit {model_limit}. Split the input into shorter sentences."
            )
        source = torch.full(
            (len(encoded), max(map(len, encoded))),
            self.tokenizers.source.pad_id,
            dtype=torch.long,
            device=self.selection.device,
        )
        for index, token_ids in enumerate(encoded):
            source[index, : len(token_ids)] = torch.tensor(token_ids, device=self.selection.device)
        output_limit = maximum_length or model_limit
        started = time.perf_counter()
        if decoding == "greedy":
            generated = self.model.greedy_generate(
                source,
                self.tokenizers.target.bos_id,
                self.tokenizers.target.eos_id,
                output_limit,
            )
        elif decoding == "beam":
            generated = self.model.beam_generate(
                source,
                self.tokenizers.target.bos_id,
                self.tokenizers.target.eos_id,
                output_limit,
                beam_width,
                length_penalty,
            )
        else:
            raise ValueError("decoding must be `greedy` or `beam`.")
        per_item_ms = (time.perf_counter() - started) * 1000 / len(texts)
        return [
            TranslationResult(
                self.tokenizers.target.decode(output_ids),
                self.version.version_id,
                max(0, len(output_ids) - 1),
                per_item_ms,
                float(torch.exp(torch.tensor(score)).item()),
                source_ids,
                self.tokenizers.source.pieces(text),
                output_ids,
            )
            for text, source_ids, output_ids, score in zip(
                texts, encoded, generated.token_ids, generated.scores
            )
        ]


def load_runtime(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    version: str = "production",
    device: str = "auto",
) -> TranslationRuntime:
    """Load a trusted registered model and exact immutable tokenizer.

    Raises:
        RegistryError: If the requested model version cannot be resolved.
        TokenizerCompatibilityError: If registry/checkpoint tokenizer IDs disagree.
        CheckpointError: If the trusted local checkpoint is missing or corrupted.
    """
    reverse = direction != config.language_pair.directions()[0]
    config.language_pair.languages_for_direction(direction)
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    metadata = registry.resolve(version)
    if not metadata.checkpoint_path:
        raise ValueError(f"Model {metadata.version_id} has no completed checkpoint.")
    selection = select_device(device, mixed_precision=False)
    base_bundle = TokenizerBundle.load(
        paths.dataset(config.language_pair.id), metadata.tokenizer_version
    )
    tokenizers = directional_tokenizers(base_bundle, reverse)
    model_config = ModelConfig.model_validate(metadata.model_configuration)
    model = build_model(model_config, tokenizers)
    payload = load_checkpoint(paths.root / metadata.checkpoint_path, map_location=selection.device)
    if payload["tokenizer_version"] != metadata.tokenizer_version:
        raise TokenizerCompatibilityError(
            f"Checkpoint tokenizer {payload['tokenizer_version']} differs from registry "
            f"{metadata.tokenizer_version}."
        )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(selection.device).eval()
    return TranslationRuntime(model, tokenizers, metadata, selection)
