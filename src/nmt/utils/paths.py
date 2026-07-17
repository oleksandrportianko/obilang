"""Project layout discovery and safe identifier-to-path resolution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def discover_project_root(start: Path | None = None) -> Path:
    """Find the repository root containing ``pyproject.toml``.

    Args:
        start: Directory from which to search upward. Defaults to the current
            working directory. ``NMT_PROJECT_ROOT`` overrides the search.

    Returns:
        An absolute project-root path.

    Raises:
        FileNotFoundError: If no project marker can be found.
    """
    override = os.getenv("NMT_PROJECT_ROOT")
    candidate = Path(override).expanduser() if override else (start or Path.cwd())
    candidate = candidate.resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError(
        f"Cannot locate pyproject.toml from {candidate}. Run inside the repository "
        "or set NMT_PROJECT_ROOT."
    )


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    """Reject path traversal and shell metacharacters in external identifiers.

    Args:
        value: Pair, direction, version, or experiment identifier.
        field_name: Human-readable field name used in error messages.

    Returns:
        The unchanged validated value.

    Raises:
        ValueError: If the value is empty or contains unsafe characters.
    """
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"Invalid {field_name} {value!r}; use letters, numbers, dots, underscores, "
            "and hyphens only."
        )
    return value


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved platform directories for one repository checkout."""

    root: Path

    @property
    def configs(self) -> Path:
        """Return the configuration root."""
        return self.root / "configs"

    def pair_config(self, pair: str) -> Path:
        """Return the validated language-pair configuration path."""
        return self.configs / "language_pairs" / f"{validate_identifier(pair, 'pair')}.yaml"

    def dataset(self, pair: str) -> Path:
        """Return the isolated dataset directory for ``pair``."""
        return self.root / "datasets" / validate_identifier(pair, "pair")

    def model_direction(self, pair: str, direction: str) -> Path:
        """Return the model artifact root for one pair and direction."""
        return (
            self.root
            / "models"
            / validate_identifier(pair, "pair")
            / validate_identifier(direction, "direction")
        )

    def experiment(self, pair: str, experiment_id: str) -> Path:
        """Return the artifact directory for one experiment."""
        return (
            self.root
            / "experiments"
            / validate_identifier(pair, "pair")
            / validate_identifier(experiment_id, "experiment ID")
        )
