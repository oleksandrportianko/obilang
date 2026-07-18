# Training Quick Guide

## macOS and Linux (Bash/zsh)

```bash
# Run all commands from the repository root.
cd /path/to/obilang

# Activate the local Python environment on macOS or Linux.
source .venv/bin/activate

# Confirm that Apple MPS acceleration is available.
# Apple Silicon normally prints: True
python -c "import torch; print(torch.backends.mps.is_available())"

# Check that aligned source and target files contain the same number of rows.
wc -l datasets/en-uk/raw/source.txt datasets/en-uk/raw/target.txt

# Validate raw and incoming data without creating a new dataset version.
nmt dataset validate --pair en-uk

# Clean, filter, version, and split the dataset.
nmt dataset prepare --pair en-uk

# Inspect accepted, rejected, training, validation, and test row counts.
nmt dataset report --pair en-uk

# Train a new SentencePiece tokenizer from the current training split.
# Run this before the first model; do not retrain it during normal fine-tuning.
nmt tokenizer train --pair en-uk

# Train a small English-to-Ukrainian model.
# Device selection is automatic: MPS on supported Macs, otherwise CPU.
nmt train \
  --pair en-uk \
  --direction en-to-uk \
  --config configs/models/small.yaml \
  --config configs/training/low_vram.yaml

# Apply a personal ignored override last, for example a higher epoch count.
nmt train \
  --pair en-uk \
  --direction en-to-uk \
  --config configs/models/small.yaml \
  --config configs/training/low_vram.yaml \
  --config configs/training/custom/more_epochs.yaml

# Train the independent Ukrainian-to-English model.
nmt train \
  --pair en-uk \
  --direction uk-to-en \
  --config configs/models/small.yaml \
  --config configs/training/low_vram.yaml

# Run a very short two-step CPU workflow instead of a normal training run.
nmt train \
  --pair en-uk \
  --direction en-to-uk \
  --config configs/smoke.yaml

# Open the local training dashboard at http://127.0.0.1:8501.
nmt ui

# List model versions, statuses, parents, datasets, and checkpoints.
nmt versions list --pair en-uk

# Resume an interrupted run from its atomic latest checkpoint.
# Replace the example version ID with the version shown by `nmt versions list`.
NMT_CHECKPOINT_PATH="models/en-uk/en-to-uk/checkpoints/en-uk-en-to-uk-v1.0.0/latest.pt"
nmt train resume --checkpoint "$NMT_CHECKPOINT_PATH" --device mps

# After adding new parallel data under datasets/en-uk/incoming/, validate it.
nmt dataset validate --pair en-uk

# Create a new complete dataset version containing historical and incoming data.
nmt dataset prepare --pair en-uk

# Fine-tune from an immutable parent with historical replay enabled by default.
NMT_PARENT_VERSION="en-uk-en-to-uk-v1.0.0"
nmt fine-tune \
  --pair en-uk \
  --direction en-to-uk \
  --from-version "$NMT_PARENT_VERSION" \
  --config configs/training/low_vram.yaml

# Evaluate a candidate on its fixed test set.
NMT_CANDIDATE_VERSION="en-uk-en-to-uk-v1.1.0"
nmt evaluate \
  --pair en-uk \
  --direction en-to-uk \
  --version "$NMT_CANDIDATE_VERSION" \
  --device mps

# Compare the candidate against its parent and inspect promotion gates.
nmt compare \
  --version-a "$NMT_PARENT_VERSION" \
  --version-b "$NMT_CANDIDATE_VERSION" \
  --device mps

# Mark a reviewed candidate as approved.
nmt versions set-status \
  --version "$NMT_CANDIDATE_VERSION" \
  --status approved

# Promote the approved candidate to the production alias.
nmt versions promote --version "$NMT_CANDIDATE_VERSION"
```

## Windows (PowerShell)

Run these commands from the repository root. PowerShell uses a backtick (`` ` ``)
for line continuation, not a backslash. The backtick must be the final character
on its line.

```powershell
# Activate the local Python environment.
.\.venv\Scripts\Activate.ps1

# Check CUDA availability. Device selection remains automatic during training.
python -c "import torch; print(torch.cuda.is_available())"

# Validate, prepare, and inspect the dataset.
nmt dataset validate --pair en-uk
nmt dataset prepare --pair en-uk
nmt dataset report --pair en-uk

# Train the tokenizer after preparing a new dataset version.
nmt tokenizer train --pair en-uk

# Train a small English-to-Ukrainian model.
nmt train `
  --pair en-uk `
  --direction en-to-uk `
  --config configs/models/small.yaml `
  --config configs/training/low_vram.yaml

# Add a personal ignored override last, for example a higher epoch count.
nmt train `
  --pair en-uk `
  --direction en-to-uk `
  --config configs/models/small.yaml `
  --config configs/training/low_vram.yaml `
  --config configs/training/custom/more_epochs.yaml

# Train the independent Ukrainian-to-English model.
nmt train `
  --pair en-uk `
  --direction uk-to-en `
  --config configs/models/small.yaml `
  --config configs/training/low_vram.yaml

# Use the short CPU smoke configuration instead of a normal run.
nmt train `
  --pair en-uk `
  --direction en-to-uk `
  --config configs/smoke.yaml

# Resume an interrupted run.
$checkpointPath = "models/en-uk/en-to-uk/checkpoints/en-uk-en-to-uk-v1.0.0/latest.pt"
nmt train resume --checkpoint $checkpointPath --device auto

# Fine-tune from an immutable parent with historical replay.
$parentVersion = "en-uk-en-to-uk-v1.0.0"
nmt fine-tune `
  --pair en-uk `
  --direction en-to-uk `
  --from-version $parentVersion `
  --config configs/training/low_vram.yaml

# Evaluate and compare a candidate against its parent.
$candidateVersion = "en-uk-en-to-uk-v1.1.0"
nmt evaluate `
  --pair en-uk `
  --direction en-to-uk `
  --version $candidateVersion `
  --device auto
nmt compare `
  --version-a $parentVersion `
  --version-b $candidateVersion `
  --device auto

# Approve and promote a reviewed candidate.
nmt versions set-status `
  --version $candidateVersion `
  --status approved
nmt versions promote --version $candidateVersion
```
