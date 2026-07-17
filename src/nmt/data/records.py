"""Typed records exchanged by dataset ingestion and validation stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParallelRecord:
    """One aligned source-target example with reproducibility provenance."""

    source: str
    target: str
    origin: str
    row_number: int
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record preserving provenance."""
        return asdict(self)


@dataclass(frozen=True)
class RejectedRecord:
    """An input example retained with machine-readable rejection reasons."""

    source: str | None
    target: str | None
    origin: str
    row_number: int
    reasons: tuple[str, ...]
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible rejected-row representation."""
        return asdict(self)


@dataclass
class ValidationReport:
    """Detailed accepted/rejected counters for one dataset build."""

    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    duplicate_rows: int = 0
    empty_rows: int = 0
    missing_value_rows: int = 0
    length_failures: int = 0
    length_ratio_failures: int = 0
    language_mismatch_rows: int = 0
    malformed_markup_rows: int = 0
    conflicting_pairs: int = 0
    exact_source_duplicates: int = 0
    exact_target_duplicates: int = 0
    input_files: list[str] = field(default_factory=list)

    def reject(self, reasons: tuple[str, ...]) -> None:
        """Increment rejection totals once and reason-specific counters.

        Args:
            reasons: Unique symbolic reasons assigned to one row.

        Returns:
            None. This report is updated in place.
        """
        self.rejected_rows += 1
        reason_to_counter = {
            "duplicate_pair": "duplicate_rows",
            "empty": "empty_rows",
            "missing_value": "missing_value_rows",
            "length": "length_failures",
            "length_ratio": "length_ratio_failures",
            "language_mismatch": "language_mismatch_rows",
            "malformed_markup": "malformed_markup_rows",
            "conflicting_translation": "conflicting_pairs",
            "source_duplicate": "exact_source_duplicates",
            "target_duplicate": "exact_target_duplicates",
        }
        for reason in reasons:
            counter = reason_to_counter.get(reason)
            if counter:
                setattr(self, counter, getattr(self, counter) + 1)

    def as_dict(self) -> dict[str, Any]:
        """Return report counters as JSON-compatible metadata."""
        return asdict(self)
