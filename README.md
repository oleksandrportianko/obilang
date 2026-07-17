# From-Scratch Neural Machine Translation Platform

A local-first, production-oriented platform for training dedicated bilingual
encoder-decoder Transformers with random weights. The initial namespace contains
two independent models:

- `en-uk/en-to-uk`
- `en-uk/uk-to-en`

No pretrained model, tokenizer, embedding, hosted AI service, or base-model
weight is used. SentencePiece tokenizers are trained from the selected local
training split. PyTorch provides tensor primitives and low-level attention; the
repository explicitly implements encoder/decoder layers, masks, residual flow,
loss preparation, optimization, greedy decoding, and beam search.

The detailed decisions and Mermaid diagrams are in
[docs/architecture.md](docs/architecture.md).

## What is included

- strict UTF-8 readers for aligned text, TSV, CSV, and JSONL;
- Unicode/whitespace normalization, configurable filters, conflicts and duplicate
  detection, retained rejected rows, and detailed reports;
- content-derived dataset and tokenizer versions with stable, leakage-aware splits;
- shared or separate from-scratch SentencePiece BPE/Unigram tokenizers;
- explicit pre-normalized encoder-decoder Transformer with causal/padding masks;
- token-budget batches, AdamW, warmup schedules, label smoothing, gradient
  accumulation/clipping, CUDA AMP, CPU/CUDA/MPS device selection, and early stop;
- atomic latest/best/periodic/final checkpoints with optimizer, scheduler, scaler,
  sampler progress, early-stop state, configuration, and all RNG states;
- immutable model lineage, production promotion/rollback, protection, and statuses;
- replay fine-tuning, parent regression evaluation, and configurable promotion gates;
- BLEU, chrF, TER, loss, perplexity, exact match, length, unknown token, number,
  punctuation, placeholder, and markup preservation metrics;
- CLI, localhost FastAPI, Streamlit monitoring/management UI, and PT2 graph export;
- local experiment streams and complete reproducibility-manifest export;
- unit, integration, and end-to-end CPU smoke tests.

COMET is intentionally not a required metric: common COMET packages download a
pretrained evaluator, which would weaken offline reproducibility. It can be added
as an explicitly optional evaluation adapter without affecting training.

## Repository layout

```text
configs/                 validated model, training, pair, and smoke YAML
datasets/<pair>/         raw, incoming, processed, splits, rejected, metadata, tokenizer
models/<pair>/<direction>/ registries, checkpoints, immutable versions, exports
experiments/<pair>/      run manifest, JSONL metrics, samples, result
reports/                 evaluations, comparisons, reproducibility exports
src/nmt/                 reusable platform package
ui/app.py                Streamlit application
tests/                   unit, integration, smoke, and synthetic corpus
docker/                  optional local container setup
```

Generated dataset/model/experiment directories are local artifacts. Back them up
according to corpus licensing and operational requirements.

## Installation

Python 3.10–3.12 is supported. CPU works everywhere. A virtual environment is
strongly recommended.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[api,ui,dev]'
nmt --help
```

On zsh, keep the extras expression quoted as shown. On Apple Silicon, the normal
PyPI PyTorch wheel supports MPS when the installed macOS/PyTorch combination does.
Verify with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

MPS is automatically preferred after CUDA; use `--device cpu` for a portable
baseline. Mixed precision currently falls back to FP32 on MPS because autocast
support differs among operating-system and PyTorch versions.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[api,ui,dev]"
nmt --help
```

If script execution is disabled, use a process-scoped policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

All platform paths use `pathlib`; workers default to zero for reliable Windows
spawn behavior. Increase `training.num_workers` only after verifying the dataset
and environment.

### NVIDIA CUDA

Install the CUDA wheel matching the local driver using the command generated at
<https://pytorch.org/get-started/locally/>, then install this project. Confirm:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The platform never assumes CUDA. A manual unavailable request fails with a clear
message; `device: auto` falls back to MPS or CPU. `configs/training/low_vram.yaml`
uses smaller token microbatches plus more accumulation for roughly 8 GB VRAM.

### Optional Docker

Docker is not required. To run the local API with persistent local artifacts:

```bash
docker compose -f docker/compose.yaml up --build
```

The compose file publishes only loopback ports. GPU containers require NVIDIA
Container Toolkit and a local compose override; the standard image remains CPU
portable.

## Dataset formats and preparation

Place files in `datasets/en-uk/raw`. New additions normally go in
`datasets/en-uk/incoming`; preparation reads both. See
[datasets/en-uk/README.md](datasets/en-uk/README.md) for schemas and alignment.

Examples:

```text
# source.txt / target.txt: equal line counts
Hello.                  Привіт.
```

```text
# TSV
source_text<TAB>target_text<TAB>optional_domain
```

```csv
source_text,target_text,domain
"Hello","Привіт","general"
```

```json
{"source":"Hello","target":"Привіт","domain":"general"}
```

Validate without writes, then create an immutable processed version:

```bash
nmt dataset validate --pair en-uk
nmt dataset prepare --pair en-uk
nmt dataset report --pair en-uk
```

Malformed and filtered rows are stored in `rejected/<version>.jsonl` with all
reasons. Exact inputs and filtering/split configuration determine the version
fingerprint. Stable seeded hashing keeps old examples in their prior split after
data is added. Fixed test/validation examples therefore remain regression
benchmarks. The simple near-duplicate signature catches punctuation, case, and
whitespace variants, not semantic paraphrases.

Optional script checking is configured per pair with `source_scripts` and
`target_scripts`, so core ingestion has no language-specific assumption. It is a
conservative heuristic and is disabled by default for the initial pair.

## Train a tokenizer from scratch

```bash
nmt tokenizer train --pair en-uk
```

The trainer reads only the current processed **training** split. Models, vocab,
configuration, corpus dataset version, statistics, hashes, and special-token IDs
are stored under `datasets/en-uk/tokenizer/<version>`. IDs are fixed as PAD=0,
UNK=1, BOS=2, and EOS=3.

Normal fine-tuning loads its parent's tokenizer. It never retrains it. To replace
a tokenizer, train a new artifact and start a fresh major model line; learned
embedding rows are otherwise incompatible. No automatic vocabulary migration is
claimed or attempted.

## Train independent directional models

The 384-dimensional 4+4-layer medium architecture is the approximately 8 GB
consumer-GPU default:

```bash
nmt train --pair en-uk --direction en-to-uk \
  --config configs/models/medium.yaml \
  --config configs/training/low_vram.yaml

nmt train --pair en-uk --direction uk-to-en \
  --config configs/models/medium.yaml \
  --config configs/training/low_vram.yaml
```

PowerShell uses the same arguments with a backtick for multiline commands:

```powershell
nmt train --pair en-uk --direction en-to-uk `
  --config configs/models/medium.yaml `
  --config configs/training/low_vram.yaml
```

Each fresh run creates a major version in `training` status, then becomes a
`candidate` only after its final checkpoint and fixed-test evaluation succeed.
The console explicitly reports device and effective precision. Structured events
also go to `experiments/<pair>/<experiment>/metrics.jsonl`.

### Stop and resume

Ctrl+C saves `interrupted.pt` and updates `latest.pt` atomically. Resume only
trusted local PyTorch checkpoints:

```bash
nmt train resume \
  --checkpoint models/en-uk/en-to-uk/checkpoints/<version>/latest.pt
```

Resume restores model, optimizer, scheduler, CUDA scaler, epoch/batch/step,
best metric, early stopping, counters, and Python/NumPy/PyTorch/CUDA RNG state.
Exact bitwise identity is still limited by hardware kernels and worker scheduling;
enable `training.deterministic` when its speed/operation constraints are acceptable.

PyTorch `.pt` uses pickle internally and can execute code while loading. Never
load an untrusted checkpoint. Dynamic-shape PT2 graph exports are provided for inference.

## Fine-tuning and catastrophic forgetting

Add data under `incoming`, run preparation, then continue from any immutable
parent:

```bash
nmt dataset prepare --pair en-uk
nmt fine-tune --pair en-uk --direction en-to-uk \
  --from-version en-uk-en-to-uk-v1.0.0
```

Replay is the recommended default. The platform detects train examples absent
from the parent's full processed data and mixes deterministic historical samples.
Supported strategies are `percentage`, `fixed_count`, `balanced`, `weighted`, and
`domain_aware`. Configure ratios/counts, a lower learning rate, embedding freeze,
or first-N encoder-layer freeze under `fine_tuning`.

New-data-only mode is explicit:

```bash
nmt fine-tune --pair en-uk --direction en-to-uk \
  --from-version en-uk-en-to-uk-v1.0.0 --new-data-only
```

It can cause catastrophic forgetting and makes no guarantee that old behavior is
preserved. Every child is new and immutable. After training, parent and child are
evaluated on the parent's original test set. Required gates decide whether the
child becomes `approved`; a failure leaves it a `candidate` for review.

## Evaluation, comparison, promotion, and rollback

```bash
nmt evaluate --pair en-uk --direction en-to-uk \
  --version en-uk-en-to-uk-v1.0.0

nmt compare \
  --version-a en-uk-en-to-uk-v1.0.0 \
  --version-b en-uk-en-to-uk-v1.1.0

nmt versions list --pair en-uk
nmt versions inspect --version en-uk-en-to-uk-v1.1.0
nmt versions promote --version en-uk-en-to-uk-v1.1.0
nmt versions rollback --pair en-uk --to en-uk-en-to-uk-v1.0.0
```

Comparison reports contain metric deltas, gate results, original/new dataset IDs,
tokenizer/config changes, checkpoint size, parameter count, speed, changed output,
and source/reference/A/B examples. Promotion protects the version and atomically
moves the directional production pointer. The prior production artifact remains
approved and available for rollback. A manual gate bypass requires `--override`.

## Translation CLI

```bash
nmt translate \
  --pair en-uk \
  --direction en-to-uk \
  --version production \
  --text "Hello, how are you?"

nmt translate-file \
  --pair en-uk \
  --direction en-to-uk \
  --version en-uk-en-to-uk-v1.2.0 \
  --input input.txt \
  --output translated.txt
```

Both commands support CPU, CUDA, and MPS plus greedy/beam decoding. Beam width,
length penalty, and maximum output length are bounded. Output includes exact model
version, token IDs, token pieces, mean-sequence confidence, and inference time.

## Local REST API

```bash
nmt api
```

OpenAPI is at `http://127.0.0.1:8000/docs`. Example:

```bash
curl -X POST http://127.0.0.1:8000/api/translate \
  -H 'Content-Type: application/json' \
  -d '{
    "language_pair":"en-uk",
    "direction":"en-to-uk",
    "model_version":"production",
    "text":"Hello",
    "decoding":"beam",
    "beam_width":4
  }'
```

The API accepts identifiers, not filesystem paths, validates field sizes, rejects
unknown keys, and resolves artifacts beneath the project root. It binds to
`127.0.0.1` by default. Authentication/TLS is not included; do not expose it to
an untrusted network without a secured reverse proxy.

## Local monitoring UI

```bash
nmt ui
```

The Streamlit application at `http://127.0.0.1:8501` provides:

- dashboard pair/run/version/production/failure summary;
- live real loss, learning-rate, BLEU, chrF, progress, token/example, memory,
  checkpoint, warning, and validation-translation events;
- data reports, splits, sources, rejections, and reproducibility manifests;
- model comparisons with changed translations and promotion gates;
- multi-version translation playground with decoding controls and token inspection;
- lineage, notes, protection, promotion, rollback, export, resume, and fine-tune.

Long-running actions launch the same CLI in an argument-array subprocess without
a shell. The UI does not simulate progress; it reads the experiment JSONL stream.

## Export and reproducibility

```bash
nmt export model --pair en-uk --direction en-to-uk --version production
nmt experiment export-manifest --pair en-uk --experiment exp-20260717-120000-abcd1234
```

An inference export contains a traced forward graph, exact source/target
SentencePiece models, special IDs, architecture, direction, and model identity.
The experiment export combines run/result, model metadata, input hashes,
preprocessing, tokenizer hashes, environment, Git commit, seeds, and dependency
versions.

## Add another language pair

No core code change is needed:

1. Create `datasets/<pair>/{raw,incoming,processed,splits,rejected,metadata,tokenizer}`.
2. Add source/target files under `raw`.
3. Copy `configs/language_pairs/en-uk.yaml` to `<pair>.yaml`, then change the pair
   ID, source/target language codes, optional scripts, tokenizer, and policies.
4. Run the same dataset, tokenizer, and directional training commands.

Every pair and direction receives isolated data, tokenizer artifacts, registry,
versions, checkpoints, metrics, experiments, and production pointer.

## Tests and CPU smoke workflow

The committed synthetic corpus is mechanics-only. On a clean checkout:

```bash
cp tests/smoke/fixtures/en-uk.tsv datasets/en-uk/raw/smoke.tsv
nmt dataset prepare --pair en-uk --config configs/smoke.yaml
nmt tokenizer train --pair en-uk --config configs/smoke.yaml
nmt train --pair en-uk --direction en-to-uk --config configs/smoke.yaml
pytest -q
```

Or run `bash scripts/smoke_test.sh`; Windows uses
`powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1`. The tiny model
trains only a few CPU steps and verifies the workflow, not translation quality.

## Troubleshooting

**Alignment failed.** The message gives both line counts. Repair the source/target
files; the platform will not guess missing translations.

**Unsupported encoding.** Convert every input to UTF-8 (without relying on the
system locale). Decode errors include the file and byte offset.

**Empty validation/test split.** Stable hash assignment needs enough distinct
examples. Add data or change split configuration, then prepare a new version.

**SentencePiece cannot reach the requested vocabulary.** `hard_vocab_limit` is
disabled, so small corpora normally produce a smaller valid vocabulary. Byte
fallback requires enough vocabulary slots for byte meta-pieces. Reduce vocabulary
or disable byte fallback for toy data.

**CUDA out of memory.** Resume from `latest.pt` after reducing `batch_tokens`,
maximum sequence length, embedding/FFN dimensions, or beam width. Increase
gradient accumulation to preserve the effective batch. Do not evaluate with a
wide beam during constrained training.

**NaN/infinite loss or gradients.** Inspect rejected/accepted data, lower learning
rate, temporarily disable mixed precision, and resume from the most recent valid
checkpoint. Non-finite values stop the run instead of corrupting later weights.

**MPS operation unsupported.** Select CPU for that run or update macOS/PyTorch.
The platform never silently honors a manual unavailable-device request.

**Corrupted checkpoint.** Use `best.pt`, another periodic `step-*.pt`, or a
protected version checkpoint. Atomic replacement prevents partial named anchors,
but disk/hardware corruption still requires backup.

**Insufficient disk.** Check free space before a long run. AdamW checkpoints can
be several times inference parameter size because they include optimizer state.
Retention removes only old periodic checkpoints, never best/final/latest or
promoted versions.

## Limits

- Local JSON registries are file-locked for one workstation, not distributed.
- Token batches reduce padding but do not implement attention key/value caching.
- Near-duplicate detection is lexical, not semantic.
- Script mismatch detection is not a full language classifier.
- Domain-only interpretation requires domain-specific benchmark files and is
  reported conservatively when unavailable.
- PT2 export contains the dynamic-shape forward graph; consumers implement the documented
  autoregressive loop. The platform CLI/API already provide both decoders.
- Translation quality depends entirely on owned corpus size/quality, compute, and
  tuning. The synthetic smoke corpus cannot produce a useful translator.

## License

The platform source code and bundled documentation are licensed under the
[MIT License](LICENSE). Third-party or user-supplied datasets, trained model
weights, and other generated artifacts may be subject to separate terms; the MIT
License does not grant rights to material you do not own or have permission to use.
