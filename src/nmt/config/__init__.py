"""Validated YAML configuration loading."""

from nmt.config.loader import load_config
from nmt.config.schema import PlatformConfig

__all__ = ["PlatformConfig", "load_config"]
