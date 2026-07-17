"""Raw-to-splits, tokenizer training, and CLI discovery integration tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from conftest import create_test_project
from nmt.cli import app
from nmt.config.loader import load_config
from nmt.data.pipeline import prepare_dataset
from nmt.tokenization.sentencepiece import TokenizerBundle, train_tokenizer
from nmt.utils.paths import ProjectPaths


def test_raw_dataset_to_immutable_tokenizer(tmp_path: Path) -> None:
    """The data/tokenizer pipeline creates traceable usable artifacts from local text."""
    root = create_test_project(tmp_path)
    paths = ProjectPaths(root)
    config = load_config("xx-yy", root=root)
    prepared = prepare_dataset(config, paths)
    assert prepared.report["accepted_rows"] == 60
    assert all(prepared.report["split_sizes"][name] > 0 for name in ("train", "validation", "test"))
    bundle = train_tokenizer(config, paths)
    assert bundle.dataset_version == prepared.version
    assert bundle.source.encode("item 1")[0] == bundle.source.bos_id
    assert bundle.source.decode(bundle.source.encode("item 1")) == "item 1"
    loaded = TokenizerBundle.load(paths.dataset("xx-yy"), bundle.version)
    assert loaded.source.vocabulary_size == bundle.source.vocabulary_size


def test_cli_help_exposes_complete_workflows() -> None:
    """The single CLI keeps top-level workflows discoverable."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("dataset", "tokenizer", "train", "fine-tune", "evaluate", "compare", "translate", "ui"):
        assert command in result.stdout
