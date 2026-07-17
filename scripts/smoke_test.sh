#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in an activated development environment.
cp tests/smoke/fixtures/en-uk.tsv datasets/en-uk/raw/smoke.tsv
python -m nmt.cli dataset prepare --pair en-uk --config configs/smoke.yaml
python -m nmt.cli tokenizer train --pair en-uk --config configs/smoke.yaml
python -m nmt.cli train --pair en-uk --direction en-to-uk --config configs/smoke.yaml
python -m nmt.cli train --pair en-uk --direction uk-to-en --config configs/smoke.yaml
python -m pytest tests/unit tests/integration tests/smoke -q
