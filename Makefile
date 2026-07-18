VENV ?= .venv

ifeq ($(OS),Windows_NT)
PYTHON ?= python
VENV_PYTHON := $(VENV)/Scripts/python.exe
VENV_NMT := $(VENV)/Scripts/nmt.exe
else
PYTHON ?= python3
VENV_PYTHON := $(VENV)/bin/python
VENV_NMT := $(VENV)/bin/nmt
endif

.PHONY: help venv install install-gpu lint test unit integration smoke api ui clean-generated

help:
	@$(PYTHON) -c "print('Targets: venv install lint test unit integration smoke api ui clean-generated')"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[api,ui,dev]"

install-gpu: venv
	@$(PYTHON) -c "print('Install the matching CUDA PyTorch wheel first; see README.md, then run make install.')"

lint:
	$(VENV_PYTHON) -m ruff check src tests ui
	$(VENV_PYTHON) -m compileall -q src ui

test:
	$(VENV_PYTHON) -m pytest -q

unit:
	$(VENV_PYTHON) -m pytest tests/unit -q

integration:
	$(VENV_PYTHON) -m pytest tests/integration -q

smoke:
	$(VENV_PYTHON) -c "from pathlib import Path; import shutil; shutil.copyfile(Path('tests/smoke/fixtures/en-uk.tsv'), Path('datasets/en-uk/raw/smoke.tsv'))"
	$(VENV_PYTHON) -m nmt.cli dataset prepare --pair en-uk --config configs/smoke.yaml
	$(VENV_PYTHON) -m nmt.cli tokenizer train --pair en-uk --config configs/smoke.yaml
	$(VENV_PYTHON) -m nmt.cli train --pair en-uk --direction en-to-uk --config configs/smoke.yaml
	$(VENV_PYTHON) -m nmt.cli train --pair en-uk --direction uk-to-en --config configs/smoke.yaml
	$(VENV_PYTHON) -m pytest -q

api:
	$(VENV_NMT) api --host 127.0.0.1 --port 8000

ui:
	$(VENV_NMT) ui --host 127.0.0.1 --port 8501

clean-generated:
	@$(PYTHON) -c "print('Generated data/model deletion is intentionally not automated. Remove exact version directories only after checking registry protection.')"
