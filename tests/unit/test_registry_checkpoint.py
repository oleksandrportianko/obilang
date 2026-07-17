"""Checkpoint roundtrip, registry lifecycle, and version-lineage tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from nmt.registry.local import LocalModelRegistry, ModelVersion
from nmt.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
from nmt.utils.paths import ProjectPaths


def checkpoint_metadata() -> dict[str, object]:
    """Return all mandatory resumability fields for serialization tests."""
    return {
        "epoch": 1,
        "global_step": 3,
        "model_configuration": {"width": 2},
        "training_configuration": {"learning_rate": 0.1},
        "tokenizer_version": "tok",
        "dataset_version": "data",
        "best_metric": 1.2,
        "early_stopping_state": {"validations_without_improvement": 1},
    }


def test_checkpoint_restores_model_optimizer_and_scheduler(tmp_path: Path) -> None:
    """Atomic checkpoints contain every state required by exact continuation."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original = model.weight.detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, None, checkpoint_metadata())
    with torch.no_grad():
        model.weight.zero_()
    payload = load_checkpoint(path)
    restore_training_state(payload, model, optimizer, scheduler, None)
    assert torch.allclose(model.weight, original)
    assert payload["global_step"] == 3


def model_version(version_id: str, label: str, parent: str | None = None) -> ModelVersion:
    """Build valid minimal registry metadata."""
    return ModelVersion(
        version_id=version_id,
        version_label=label,
        parent_version=parent,
        created_at=datetime.now(timezone.utc).isoformat(),
        language_pair="aa-bb",
        direction="aa-to-bb",
        experiment_id=f"exp-{label}",
        model_configuration={},
        training_configuration={},
        tokenizer_version="tok",
        dataset_versions=["data"],
        random_seed=42,
    )


def test_registry_lineage_promotion_and_rollback(tmp_path: Path) -> None:
    """Children do not overwrite parents and production rollback preserves both."""
    paths = ProjectPaths(tmp_path)
    registry = LocalModelRegistry(paths, "aa-bb", "aa-to-bb")
    first_id, first_label = registry.allocate_version()
    first = model_version(first_id, first_label)
    registry.add(first)
    registry.update(first_id, status="approved")
    registry.promote(first_id)
    child_id, child_label = registry.allocate_version(first_id)
    child = model_version(child_id, child_label, first_id)
    registry.add(child)
    registry.update(child_id, status="approved")
    registry.promote(child_id)
    assert registry.resolve("production").version_id == child_id
    registry.rollback(first_id)
    assert registry.resolve("production").version_id == first_id
    assert registry.resolve(child_id).status == "approved"
