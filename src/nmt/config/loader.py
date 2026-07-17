"""YAML loading, recursive overlays, and user-facing validation errors."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from nmt.config.schema import PlatformConfig
from nmt.utils.paths import ProjectPaths, discover_project_root


class ConfigurationError(ValueError):
    """Raised when YAML syntax or typed platform configuration is invalid."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one required YAML mapping with actionable parse errors."""
    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    pair: str,
    overlays: list[Path] | None = None,
    root: Path | None = None,
) -> PlatformConfig:
    """Load and validate a pair configuration plus optional YAML overlays.

    Args:
        pair: Language-pair identifier such as ``en-uk``.
        overlays: Model, training, or experiment YAML files applied in order.
        root: Explicit repository root, primarily useful for tests.

    Returns:
        A fully populated, immutable-by-convention ``PlatformConfig``.

    Raises:
        ConfigurationError: If a file is missing, malformed, contains an unknown
            key, or violates a typed configuration constraint.
    """
    paths = ProjectPaths((root or discover_project_root()).resolve())
    document = _read_yaml(paths.pair_config(pair))
    for overlay_path in overlays or []:
        document = _deep_merge(document, _read_yaml(overlay_path.resolve()))
    try:
        config = PlatformConfig.model_validate(document)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration for {pair}:\n{exc}") from exc
    if config.language_pair.id != pair:
        raise ConfigurationError(
            f"Pair configuration ID {config.language_pair.id!r} does not match requested {pair!r}."
        )
    return config
