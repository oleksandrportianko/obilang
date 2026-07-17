"""Streamlit local dashboard for real training, datasets, versions, and translation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nmt.config.loader import load_config
from nmt.export.portable import export_model
from nmt.inference.service import load_runtime
from nmt.monitoring.store import list_experiments, read_metric_events
from nmt.registry.local import LocalModelRegistry, RegistryError
from nmt.utils.io import load_json
from nmt.utils.paths import ProjectPaths, discover_project_root
from nmt.versioning.comparison import compare_versions


@st.cache_resource(show_spinner="Loading model weights…")
def cached_runtime(root: str, pair: str, direction: str, version: str, device: str):
    """Cache immutable model runtimes between Streamlit reruns."""
    paths = ProjectPaths(Path(root))
    return load_runtime(load_config(pair, root=paths.root), paths, direction, version, device)


def configured_pairs(paths: ProjectPaths) -> list[str]:
    """Discover pair IDs only from validated configuration filenames."""
    return sorted(path.stem for path in (paths.configs / "language_pairs").glob("*.yaml"))


def launch_cli(paths: ProjectPaths, arguments: list[str]) -> int:
    """Start an authorized long-running CLI workflow without shell interpretation."""
    process = subprocess.Popen(
        [sys.executable, "-m", "nmt.cli", *arguments],
        cwd=paths.root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.pid


def dashboard(paths: ProjectPaths, pair: str, direction: str) -> None:
    """Render language/model/run summary cards and failure list."""
    registry = LocalModelRegistry(paths, pair, direction)
    versions = registry.list_versions()
    experiments = list_experiments(paths, pair)
    try:
        production = registry.resolve("production").version_id
    except RegistryError:
        production = "Not set"
    active = [item for item in experiments if item["status"] == "training"]
    failed = [item for item in experiments if item["status"] == "failed"]
    columns = st.columns(4)
    columns[0].metric("Language pairs", len(configured_pairs(paths)))
    columns[1].metric("Directional versions", len(versions))
    columns[2].metric("Active runs", len(active))
    columns[3].metric("Failed runs", len(failed))
    st.caption(f"Production: {production}")
    if versions:
        st.subheader("Latest versions")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "version": item.version_id,
                        "status": item.status,
                        "parent": item.parent_version,
                        "dataset": item.dataset_versions[-1],
                        "tokenizer": item.tokenizer_version,
                        "best validation": item.best_validation_score,
                    }
                    for item in reversed(versions[-10:])
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    recent_results = [item for item in experiments if item["result"]][:5]
    if recent_results:
        st.subheader("Recent evaluation/run results")
        st.json([item["result"] for item in recent_results], expanded=False)
    if failed:
        st.subheader("Failed runs")
        st.json([item["result"] for item in failed[:5]], expanded=False)


def training_view(paths: ProjectPaths, pair: str) -> None:
    """Render live real metric streams, warnings, checkpoints, and translations."""
    experiments = list_experiments(paths, pair)
    if not experiments:
        st.info("No experiments yet. Start training from the CLI or Model Management tab.")
        return
    selected_id = st.selectbox("Experiment", [item["experiment_id"] for item in experiments])
    selected = next(item for item in experiments if item["experiment_id"] == selected_id)
    directory = paths.root / "experiments" / pair / selected_id
    events = read_metric_events(directory / "metrics.jsonl")
    st.json(selected["manifest"], expanded=False)
    if selected["result"]:
        st.json(selected["result"], expanded=False)
    if not events:
        st.info("The run has not written a metric event yet.")
        return
    frame = pd.DataFrame(events)
    train = frame[frame.get("event", pd.Series(dtype=str)) == "train"]
    validation = frame[frame.get("event", pd.Series(dtype=str)) == "validation"]
    if not train.empty:
        latest = train.iloc[-1]
        columns = st.columns(5)
        columns[0].metric("Epoch", latest.get("epoch"))
        columns[1].metric("Step", latest.get("step"))
        columns[2].metric("Training loss", f"{latest.get('training_loss', 0):.4f}")
        columns[3].metric("Learning rate", f"{latest.get('learning_rate', 0):.3g}")
        columns[4].metric("Progress", f"{100 * latest.get('estimated_progress', 0):.1f}%")
        chart_fields = [field for field in ["training_loss", "learning_rate"] if field in train]
        st.line_chart(train.set_index("step")[chart_fields])
        st.write(
            {
                "tokens_processed": latest.get("tokens_processed"),
                "examples_processed": latest.get("examples_processed"),
                "elapsed_seconds": latest.get("elapsed_seconds"),
                "device_memory": latest.get("device_memory"),
            }
        )
    if not validation.empty:
        chart_fields = [field for field in ["loss", "bleu", "chrf"] if field in validation]
        st.subheader("Validation")
        st.line_chart(validation.set_index("step")[chart_fields])
        step = int(validation.iloc[-1]["step"])
        sample_path = directory / "samples" / f"step-{step:08d}.json"
        samples = load_json(sample_path, [])
        if samples:
            st.subheader("Validation translations")
            st.dataframe(pd.DataFrame(samples), use_container_width=True, hide_index=True)
    checkpoints = [event for event in events if event.get("event") == "checkpoint"]
    if checkpoints:
        st.subheader("Checkpoint events")
        st.dataframe(pd.DataFrame(checkpoints), use_container_width=True, hide_index=True)
    auto_refresh = st.checkbox("Refresh this view every 5 seconds", value=False)
    if auto_refresh:
        st.components.v1.html(
            "<script>setTimeout(() => window.parent.location.reload(), 5000);</script>", height=0
        )


def dataset_view(paths: ProjectPaths, pair: str) -> None:
    """Render versions, row-quality reports, split sizes, inputs, and warnings."""
    root = paths.dataset(pair)
    current = load_json(root / "metadata" / "current.json", {})
    st.write("Current dataset", current or "Not prepared")
    reports = sorted((root / "metadata").glob("report-*.json"), reverse=True)
    if not reports:
        st.info("No dataset report. Run `nmt dataset prepare --pair ...`.")
        return
    report_path = st.selectbox("Dataset report", reports, format_func=lambda path: path.name)
    report = load_json(report_path, {})
    top = st.columns(4)
    top[0].metric("Accepted", report.get("accepted_rows", 0))
    top[1].metric("Rejected", report.get("rejected_rows", 0))
    top[2].metric("Duplicates", report.get("duplicate_rows", 0))
    top[3].metric("Conflicts", report.get("conflicting_pairs", 0))
    st.bar_chart(pd.Series(report.get("split_sizes", {}), name="rows"))
    st.json(report, expanded=False)
    version = report.get("dataset_version")
    if version:
        manifest = load_json(root / "processed" / version / "manifest.json", {})
        st.subheader("Sources and reproducibility")
        st.json(manifest, expanded=False)
        rejected_path = root / "rejected" / f"{version}.jsonl"
        if rejected_path.exists():
            rejected = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()[:100]]
            if rejected:
                st.subheader("Rejected rows (first 100)")
                st.dataframe(pd.DataFrame(rejected), use_container_width=True, hide_index=True)


def comparison_view(paths: ProjectPaths, config: Any, direction: str) -> None:
    """Evaluate and display metric, config, dataset, speed, and output differences."""
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    versions = registry.list_versions()
    if len(versions) < 2:
        st.info("At least two completed versions are needed for comparison.")
        return
    identifiers = [item.version_id for item in versions if item.checkpoint_path]
    first = st.selectbox("Version A", identifiers, index=max(0, len(identifiers) - 2))
    second = st.selectbox("Version B", identifiers, index=len(identifiers) - 1)
    if st.button("Run comparison"):
        with st.spinner("Evaluating both real models on a common fixed test set…"):
            report = compare_versions(config, paths, direction, first, second)
        st.session_state["comparison_report"] = report
    report = st.session_state.get("comparison_report")
    if report and report.get("direction") == direction:
        st.success(f"Promotion gates passed: {report['promotion_gates']['passed']}")
        st.subheader("Metric differences (B − A)")
        st.dataframe(
            pd.DataFrame.from_dict(
                report["metrics"]["deltas_b_minus_a"], orient="index", columns=["delta"]
            ),
            use_container_width=True,
        )
        st.json(report["promotion_gates"], expanded=True)
        st.subheader("Changed translations")
        st.dataframe(pd.DataFrame(report["changed_translations"]), use_container_width=True)
        st.subheader("Configuration, data, tokenizer, and performance")
        st.json(
            {
                "configuration": report["configuration_differences"],
                "datasets": report["dataset_differences"],
                "tokenizer": report["tokenizer_difference"],
                "performance": report["performance"],
                "interpretation": report["interpretation"],
            },
            expanded=False,
        )


def playground(paths: ProjectPaths, config: Any, direction: str) -> None:
    """Translate text with one or multiple versions and inspect tokenization/confidence."""
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    identifiers = [item.version_id for item in registry.list_versions() if item.checkpoint_path]
    if not identifiers:
        st.info("No completed model version is available.")
        return
    selected = st.multiselect("Model versions", identifiers, default=identifiers[-1:])
    text = st.text_area("Source text", height=120)
    columns = st.columns(4)
    decoding = columns[0].selectbox("Decoding", ["beam", "greedy"])
    beam_width = columns[1].number_input("Beam width", 1, 16, 4)
    length_penalty = columns[2].number_input("Length penalty", 0.0, 3.0, 0.6)
    device = columns[3].selectbox("Device", ["auto", "cpu", "cuda", "mps"])
    if st.button("Translate", disabled=not text.strip() or not selected):
        rows = []
        for version in selected:
            runtime = cached_runtime(str(paths.root), config.language_pair.id, direction, version, device)
            result = runtime.translate_batch(
                [text], decoding=decoding, beam_width=int(beam_width), length_penalty=length_penalty
            )[0]
            rows.append(
                {
                    "version": result.model_version,
                    "translation": result.translation,
                    "confidence": result.confidence,
                    "time_ms": result.inference_time_ms,
                    "source_pieces": result.source_pieces,
                    "output_ids": result.output_token_ids,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def model_management(paths: ProjectPaths, config: Any, direction: str) -> None:
    """Promote, roll back, annotate, protect, export, resume, and fine-tune versions."""
    registry = LocalModelRegistry(paths, config.language_pair.id, direction)
    versions = registry.list_versions()
    if not versions:
        st.info("No versions have been allocated.")
        return
    identifiers = [item.version_id for item in versions]
    selected = st.selectbox("Version", identifiers, index=len(identifiers) - 1)
    metadata = registry.resolve(selected)
    st.json(metadata.model_dump(mode="json"), expanded=False)
    action_columns = st.columns(4)
    if action_columns[0].button("Promote"):
        registry.promote(selected)
        st.success(f"Promoted {selected}")
    if action_columns[1].button("Roll back here"):
        registry.rollback(selected)
        st.success(f"Production rolled back to {selected}")
    if action_columns[2].button("Protect" if not metadata.protected else "Unprotect"):
        registry.protect(selected, not metadata.protected)
        st.success("Protection updated")
    if action_columns[3].button("Export model", disabled=not metadata.checkpoint_path):
        destination = export_model(config, paths, direction, selected)
        st.success(f"Exported to {destination.relative_to(paths.root)}")
    notes = st.text_area("Notes", value=metadata.notes)
    if st.button("Save notes"):
        registry.update(selected, notes=notes)
        st.success("Notes saved")
    st.subheader("Lineage")
    st.dataframe(
        pd.DataFrame(
            [{"version": item.version_id, "parent": item.parent_version, "status": item.status} for item in versions]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Start continued training")
    if st.button("Start fine-tuning from selected version", disabled=not metadata.checkpoint_path):
        pid = launch_cli(
            paths,
            ["fine-tune", "--pair", config.language_pair.id, "--direction", direction, "--from-version", selected],
        )
        st.success(f"Fine-tuning process started (PID {pid}). Track it in Training.")
    checkpoints = sorted(paths.model_direction(config.language_pair.id, direction).glob("checkpoints/*/latest.pt"))
    if checkpoints:
        checkpoint = st.selectbox("Resumable checkpoint", checkpoints, format_func=lambda path: str(path.relative_to(paths.root)))
        if st.button("Resume selected checkpoint"):
            pid = launch_cli(paths, ["train", "resume", "--checkpoint", str(checkpoint)])
            st.success(f"Resume process started (PID {pid}).")


def main() -> None:
    """Run the complete local monitoring and model-management application."""
    st.set_page_config(page_title="From-Scratch NMT", layout="wide")
    st.title("From-Scratch Neural Machine Translation")
    paths = ProjectPaths(discover_project_root())
    pairs = configured_pairs(paths)
    if not pairs:
        st.error("No language-pair configuration exists under configs/language_pairs.")
        return
    pair = st.sidebar.selectbox("Language pair", pairs)
    config = load_config(pair, root=paths.root)
    direction = st.sidebar.selectbox("Direction", config.language_pair.directions())
    page = st.sidebar.radio(
        "View",
        ["Dashboard", "Training", "Datasets", "Version Comparison", "Playground", "Model Management"],
    )
    if page == "Dashboard":
        dashboard(paths, pair, direction)
    elif page == "Training":
        training_view(paths, pair)
    elif page == "Datasets":
        dataset_view(paths, pair)
    elif page == "Version Comparison":
        comparison_view(paths, config, direction)
    elif page == "Playground":
        playground(paths, config, direction)
    else:
        model_management(paths, config, direction)


if __name__ == "__main__":
    main()
