"""Typed local translation REST API with identifier and input-size validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nmt import __version__
from nmt.config.loader import ConfigurationError, load_config
from nmt.inference.service import load_runtime
from nmt.registry.local import LocalModelRegistry, RegistryError
from nmt.utils.paths import ProjectPaths, discover_project_root

IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ApiModel(BaseModel):
    """Strict API payload base that rejects accidental extra keys."""

    model_config = ConfigDict(extra="forbid")


class TranslationRequest(ApiModel):
    """One local model translation request."""

    language_pair: str = Field(pattern=IDENTIFIER_PATTERN)
    direction: str = Field(pattern=IDENTIFIER_PATTERN)
    model_version: str = Field(default="production", pattern=IDENTIFIER_PATTERN)
    text: str = Field(min_length=1, max_length=20_000)
    decoding: Literal["greedy", "beam"] = "beam"
    beam_width: int = Field(default=4, ge=1, le=16)
    length_penalty: float = Field(default=0.6, ge=0, le=3)
    maximum_length: int | None = Field(default=None, ge=2, le=2048)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"


class TranslationResponse(ApiModel):
    """Decoded output and traceable model/performance metadata."""

    translation: str
    model_version: str
    token_count: int
    inference_time_ms: float
    confidence: float
    source_pieces: list[str]
    output_token_ids: list[int]


def create_app(root: Path | None = None) -> FastAPI:
    """Create a local-only-by-launch-policy FastAPI application.

    Args:
        root: Explicit repository root. API payloads never accept filesystem paths.

    Returns:
        Configured FastAPI instance. The CLI binds it to 127.0.0.1 by default.
    """
    paths = ProjectPaths((root or discover_project_root()).resolve())
    application = FastAPI(
        title="From-Scratch NMT API",
        version=__version__,
        description="Local bilingual translation using only registered project models.",
    )

    @lru_cache(maxsize=8)
    def cached_runtime(pair: str, direction: str, version: str, device: str):
        """Cache a bounded number of immutable loaded model versions."""
        config = load_config(pair, root=paths.root)
        return load_runtime(config, paths, direction, version, device)

    @application.get("/api/health")
    def health() -> dict[str, str]:
        """Return process readiness without loading a model."""
        return {"status": "ok", "version": __version__}

    @application.get("/api/models/{language_pair}/{direction}")
    def models(language_pair: str, direction: str) -> dict[str, object]:
        """List local registered versions for a validated pair and direction."""
        try:
            config = load_config(language_pair, root=paths.root)
            config.language_pair.languages_for_direction(direction)
            registry = LocalModelRegistry(paths, language_pair, direction)
            versions = registry.list_versions()
            production = None
            try:
                production = registry.resolve("production").version_id
            except RegistryError:
                pass
            return {
                "language_pair": language_pair,
                "direction": direction,
                "production_version": production,
                "versions": [item.model_dump(mode="json") for item in versions],
            }
        except (ValueError, ConfigurationError, RegistryError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.post("/api/translate", response_model=TranslationResponse)
    def translate(request: TranslationRequest) -> TranslationResponse:
        """Translate one sentence using a trusted registered local checkpoint."""
        try:
            runtime = cached_runtime(
                request.language_pair,
                request.direction,
                request.model_version,
                request.device,
            )
            result = runtime.translate_batch(
                [request.text],
                decoding=request.decoding,
                beam_width=request.beam_width,
                length_penalty=request.length_penalty,
                maximum_length=request.maximum_length,
            )[0]
            return TranslationResponse(
                translation=result.translation,
                model_version=result.model_version,
                token_count=result.token_count,
                inference_time_ms=result.inference_time_ms,
                confidence=result.confidence,
                source_pieces=result.source_pieces,
                output_token_ids=result.output_token_ids,
            )
        except (ValueError, RuntimeError, FileNotFoundError, ConfigurationError, RegistryError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return application
