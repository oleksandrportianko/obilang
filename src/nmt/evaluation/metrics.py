"""Transparent non-LLM translation quality and preservation metrics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from statistics import mean
from typing import Any

import sacrebleu

NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d[\d ,.'’]*\d|\d)(?:[.,]\d+)?(?!\w)")
PLACEHOLDER = re.compile(
    r"(?:\{\{[^{}]+\}\}|\{[^{}]+\}|%\([^)]+\)[a-zA-Z]|%[sdif]|\$\{[^{}]+\}|<%[^%]+%>)"
)
MARKUP = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*?)?/?>")


def _multiset_preservation(source: str, candidate: str, pattern: re.Pattern[str]) -> float:
    """Return exact extracted-item recall, counting repeated items."""
    expected = Counter(pattern.findall(source))
    if not expected:
        return 1.0
    actual = Counter(pattern.findall(candidate))
    preserved = sum(min(count, actual[item]) for item, count in expected.items())
    return preserved / sum(expected.values())


def _punctuation(text: str) -> Counter[str]:
    """Extract Unicode punctuation while excluding letters, digits, and spaces."""
    return Counter(character for character in text if unicodedata.category(character).startswith("P"))


def _punctuation_preservation(source: str, candidate: str) -> float:
    """Measure multiset recall for source punctuation in candidate output."""
    expected, actual = _punctuation(source), _punctuation(candidate)
    if not expected:
        return 1.0
    return sum(min(count, actual[item]) for item, count in expected.items()) / sum(expected.values())


def calculate_text_metrics(
    sources: list[str],
    references: list[str],
    candidates: list[str],
    token_loss: float | None = None,
    unknown_token_count: int = 0,
    generated_token_count: int = 0,
) -> dict[str, Any]:
    """Calculate corpus quality, exactness, length, and formatting metrics.

    Args:
        sources: Input sentences in evaluation order.
        references: Gold translations aligned with sources.
        candidates: Model translations aligned with references.
        token_loss: Optional mean non-padding cross-entropy.
        unknown_token_count: Generated or input unknown IDs observed.
        generated_token_count: Denominator for unknown-token rate.

    Returns:
        JSON-compatible metrics. BLEU and chrF use SacreBLEU's standard corpus
        implementations. TER is included when candidates are non-empty.

    Raises:
        ValueError: If the three corpora are empty or differently sized.
    """
    if not sources or len(sources) != len(references) or len(references) != len(candidates):
        raise ValueError(
            "Metric inputs must be non-empty and equally aligned: "
            f"sources={len(sources)}, references={len(references)}, candidates={len(candidates)}."
        )
    bleu = sacrebleu.corpus_bleu(candidates, [references]).score
    chrf = sacrebleu.corpus_chrf(candidates, [references]).score
    ter = sacrebleu.corpus_ter(candidates, [references]).score
    exact = mean(candidate.strip() == reference.strip() for candidate, reference in zip(candidates, references))
    source_lengths = [len(value.split()) for value in sources]
    reference_lengths = [len(value.split()) for value in references]
    candidate_lengths = [len(value.split()) for value in candidates]
    result: dict[str, Any] = {
        "bleu": bleu,
        "chrf": chrf,
        "ter": ter,
        "exact_match": exact,
        "number_accuracy": mean(
            _multiset_preservation(source, candidate, NUMBER)
            for source, candidate in zip(sources, candidates)
        ),
        "punctuation_accuracy": mean(
            _punctuation_preservation(source, candidate)
            for source, candidate in zip(sources, candidates)
        ),
        "placeholder_accuracy": mean(
            _multiset_preservation(source, candidate, PLACEHOLDER)
            for source, candidate in zip(sources, candidates)
        ),
        "markup_accuracy": mean(
            _multiset_preservation(source, candidate, MARKUP)
            for source, candidate in zip(sources, candidates)
        ),
        "unknown_token_rate": unknown_token_count / max(1, generated_token_count),
        "length_statistics": {
            "source_mean_words": mean(source_lengths),
            "reference_mean_words": mean(reference_lengths),
            "candidate_mean_words": mean(candidate_lengths),
            "candidate_reference_ratio": sum(candidate_lengths) / max(1, sum(reference_lengths)),
        },
        "sentence_count": len(candidates),
    }
    if token_loss is not None:
        result["loss"] = token_loss
        result["perplexity"] = math.exp(min(token_loss, 50.0))
    return result


def terminology_accuracy(
    sources: list[str], candidates: list[str], terminology: dict[str, list[str]]
) -> float:
    """Measure required target-term presence for configured source terminology.

    Args:
        sources: Evaluation source sentences.
        candidates: Aligned model outputs.
        terminology: Source term to acceptable target renderings mapping.

    Returns:
        Fraction of triggered source terms with at least one accepted target form,
        or 1.0 when the corpus triggers no configured term.
    """
    matched = 0
    total = 0
    for source, candidate in zip(sources, candidates):
        for source_term, expected_terms in terminology.items():
            if source_term.casefold() in source.casefold():
                total += 1
                matched += any(term.casefold() in candidate.casefold() for term in expected_terms)
    return matched / total if total else 1.0
