"""End-to-end CPU workflow including train, resume, fine-tune, compare, API, and export."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from conftest import create_test_project
from nmt.api.app import create_app
from nmt.config.loader import load_config
from nmt.data.pipeline import prepare_dataset
from nmt.evaluation.runner import evaluate_version
from nmt.export.portable import export_model
from nmt.fine_tuning.replay import fine_tune_model
from nmt.inference.service import load_runtime
from nmt.registry.local import LocalModelRegistry
from nmt.tokenization.sentencepiece import train_tokenizer
from nmt.training.checkpoint import load_checkpoint
from nmt.training.trainer import train_model
from nmt.utils.paths import ProjectPaths


def test_complete_cpu_pipeline(tmp_path: Path) -> None:
    """A tiny corpus crosses every core boundary without asserting translation quality."""
    root = create_test_project(tmp_path)
    paths = ProjectPaths(root)
    config = load_config("xx-yy", root=root)
    first_dataset = prepare_dataset(config, paths)
    train_tokenizer(config, paths)
    trained = train_model(config, paths, "xx-to-yy")
    assert trained.final_checkpoint.is_file()
    checkpoint = load_checkpoint(trained.final_checkpoint)
    assert checkpoint["global_step"] == trained.global_step
    registry = LocalModelRegistry(paths, "xx-yy", "xx-to-yy")
    assert registry.resolve(trained.version_id).status == "candidate"

    runtime = load_runtime(config, paths, "xx-to-yy", trained.version_id, device="cpu")
    translation = runtime.translate_batch(["item 3"], decoding="greedy", maximum_length=8)[0]
    assert translation.model_version == trained.version_id
    evaluation = evaluate_version(config, paths, "xx-to-yy", trained.version_id, device="cpu")
    assert evaluation["metrics"]["sentence_count"] > 0

    client = TestClient(create_app(root))
    response = client.post(
        "/api/translate",
        json={
            "language_pair": "xx-yy",
            "direction": "xx-to-yy",
            "model_version": trained.version_id,
            "text": "item 4",
            "decoding": "greedy",
            "maximum_length": 8,
            "device": "cpu",
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == trained.version_id

    # Exact resume from a mid-run periodic checkpoint keeps the immutable version
    # while restoring and advancing optimizer/scheduler/RNG state.
    periodic = trained.final_checkpoint.parent / "step-00000001.pt"
    resumed = train_model(config, paths, "xx-to-yy", resume_checkpoint=periodic)
    assert resumed.version_id == trained.version_id
    assert resumed.global_step > 1

    incoming = root / "datasets" / "xx-yy" / "incoming" / "new.tsv"
    incoming.write_text(
        "source_text\ttarget_text\tdomain\n"
        + "\n".join(f"fresh {i}\tnew {i}\tnew-domain" for i in range(101, 121))
        + "\n",
        encoding="utf-8",
    )
    second_dataset = prepare_dataset(config, paths)
    assert second_dataset.version != first_dataset.version
    fine_tuned = fine_tune_model(config, paths, "xx-to-yy", trained.version_id)
    assert fine_tuned.training.version_id != trained.version_id
    assert fine_tuned.replay_report["sampled_historical_examples"] > 0
    assert fine_tuned.comparison["version_a"] == trained.version_id
    assert fine_tuned.comparison["version_b"] == fine_tuned.training.version_id

    export_directory = export_model(
        config, paths, "xx-to-yy", fine_tuned.training.version_id
    )
    assert (export_directory / "model.pt2").is_file()
    assert (export_directory / "manifest.json").is_file()
