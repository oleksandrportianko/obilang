"""Side-by-side model regression reports and configurable promotion gates."""

from __future__ import annotations

from typing import Any

from nmt.config.schema import PlatformConfig
from nmt.evaluation.runner import evaluate_version
from nmt.registry.local import LocalModelRegistry
from nmt.utils.io import atomic_write_json
from nmt.utils.paths import ProjectPaths


def _mapping_difference(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Recursively report only configuration keys whose values differ."""
    result: dict[str, Any] = {}
    for key in sorted(set(first) | set(second)):
        left, right = first.get(key), second.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            nested = _mapping_difference(left, right)
            if nested:
                result[key] = nested
        elif left != right:
            result[key] = {"version_a": left, "version_b": right}
    return result


def promotion_gate_results(
    config: PlatformConfig, baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate candidate deltas against explicit quality-preservation thresholds."""
    requirements = config.promotion
    checks = {
        "minimum_chrf_change": {
            "actual": candidate["chrf"] - baseline["chrf"],
            "required": requirements.minimum_chrf_change,
            "passed": candidate["chrf"] - baseline["chrf"] >= requirements.minimum_chrf_change,
        },
        "maximum_bleu_regression": {
            "actual": baseline["bleu"] - candidate["bleu"],
            "required": requirements.maximum_bleu_regression,
            "passed": baseline["bleu"] - candidate["bleu"] <= requirements.maximum_bleu_regression,
        },
        "maximum_number_accuracy_regression": {
            "actual": baseline["number_accuracy"] - candidate["number_accuracy"],
            "required": requirements.maximum_number_accuracy_regression,
            "passed": baseline["number_accuracy"] - candidate["number_accuracy"]
            <= requirements.maximum_number_accuracy_regression,
        },
        "maximum_placeholder_accuracy_regression": {
            "actual": baseline["placeholder_accuracy"] - candidate["placeholder_accuracy"],
            "required": requirements.maximum_placeholder_accuracy_regression,
            "passed": baseline["placeholder_accuracy"] - candidate["placeholder_accuracy"]
            <= requirements.maximum_placeholder_accuracy_regression,
        },
    }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def compare_versions(
    config: PlatformConfig,
    paths: ProjectPaths,
    direction: str,
    version_a: str,
    version_b: str,
    *,
    dataset_version: str | None = None,
    device: str = "auto",
    update_candidate_status: bool = False,
) -> dict[str, Any]:
    """Compare two models on one common fixed benchmark and apply promotion gates.

    The report includes metrics/deltas, examples with both candidates, changed
    translations, configuration/data/tokenizer differences, speed, memory proxy,
    model size, and an interpretation of overall/domain-only/regression outcomes.
    """
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    metadata_a, metadata_b = registry.resolve(version_a), registry.resolve(version_b)
    common = [item for item in metadata_a.dataset_versions if item in metadata_b.dataset_versions]
    benchmark = dataset_version or (common[-1] if common else metadata_a.dataset_versions[0])
    report_a = evaluate_version(config, paths, direction, metadata_a.version_id, benchmark, device)
    report_b = evaluate_version(config, paths, direction, metadata_b.version_id, benchmark, device)
    metrics_a, metrics_b = report_a["metrics"], report_b["metrics"]
    numeric_deltas = {
        key: metrics_b[key] - metrics_a[key]
        for key in sorted(set(metrics_a) & set(metrics_b))
        if isinstance(metrics_a[key], (int, float)) and isinstance(metrics_b[key], (int, float))
    }
    samples = []
    changed = []
    for first, second in zip(report_a["samples"], report_b["samples"]):
        item = {
            "source": first["source"],
            "reference": first["reference"],
            "version_a_output": first["candidate"],
            "version_b_output": second["candidate"],
            "changed": first["candidate"] != second["candidate"],
        }
        samples.append(item)
        if item["changed"]:
            changed.append(item)
    gates = promotion_gate_results(config, metrics_a, metrics_b)
    interpretation = {
        "improved_overall": numeric_deltas.get("chrf", 0) > 0 and numeric_deltas.get("bleu", 0) >= 0,
        "regressed_on_original_test": not gates["passed"],
        "formatting_regression": numeric_deltas.get("placeholder_accuracy", 0) < 0
        or numeric_deltas.get("markup_accuracy", 0) < 0
        or numeric_deltas.get("punctuation_accuracy", 0) < 0,
        "terminology_regression": None,
        "domain_only_improvement": False,
    }
    comparison: dict[str, Any] = {
        "language_pair": config.language_pair.id,
        "direction": direction,
        "version_a": metadata_a.version_id,
        "version_b": metadata_b.version_id,
        "benchmark_dataset_version": benchmark,
        "metrics": {"version_a": metrics_a, "version_b": metrics_b, "deltas_b_minus_a": numeric_deltas},
        "promotion_gates": gates,
        "interpretation": interpretation,
        "configuration_differences": {
            "model": _mapping_difference(metadata_a.model_configuration, metadata_b.model_configuration),
            "training": _mapping_difference(metadata_a.training_configuration, metadata_b.training_configuration),
        },
        "dataset_differences": {
            "only_version_a": sorted(set(metadata_a.dataset_versions) - set(metadata_b.dataset_versions)),
            "only_version_b": sorted(set(metadata_b.dataset_versions) - set(metadata_a.dataset_versions)),
        },
        "tokenizer_difference": {
            "version_a": metadata_a.tokenizer_version,
            "version_b": metadata_b.tokenizer_version,
            "changed": metadata_a.tokenizer_version != metadata_b.tokenizer_version,
        },
        "performance": {
            "version_a": report_a["performance"],
            "version_b": report_b["performance"],
            "sentences_per_second_change": report_b["performance"]["sentences_per_second"]
            - report_a["performance"]["sentences_per_second"],
            "checkpoint_bytes_change": report_b["performance"]["checkpoint_bytes"]
            - report_a["performance"]["checkpoint_bytes"],
        },
        "translations": samples,
        "changed_translations": changed,
    }
    destination = (
        paths.root
        / "reports"
        / config.language_pair.id
        / direction
        / f"comparison-{metadata_a.version_label}-{metadata_b.version_label}-{benchmark}.json"
    )
    atomic_write_json(destination, comparison)
    comparison["report_path"] = str(destination.relative_to(paths.root))
    if update_candidate_status:
        registry.update(
            metadata_b.version_id,
            status="approved" if gates["passed"] else "candidate",
            regression_metrics={key: float(value) for key, value in numeric_deltas.items()},
        )
    return comparison
