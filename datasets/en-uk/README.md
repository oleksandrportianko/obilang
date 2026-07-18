# English–Ukrainian dataset namespace

This directory contains data shared by the two independent directional models
`en-to-uk` and `uk-to-en`. Text must be UTF-8. Do not place tokenizer models or
model checkpoints in `raw` or `incoming`.

## Accepted inputs

- Aligned text: `raw/source.txt` and `raw/target.txt`. Line N in source is the
  translation of line N in target; counts must match exactly.
- TSV: first two columns are source and target. An optional third column is the
  domain. A `source_text<TAB>target_text` header is accepted.
- CSV: required headers are `source_text,target_text`; optional `domain` and
  extra metadata columns are preserved.
- JSONL: one object per line, for example
  `{"source":"Hello","target":"Привіт","domain":"general"}`.

`raw` is historical input and `incoming` is newly acquired data. Preparation
reads both, so a new version contains the complete visible corpus. The platform
normalizes Unicode/whitespace but preserves punctuation and numbers. Exact pair
duplicates and conflicting repeated source or target sentences are rejected.
Nothing is silently deleted: rejected values and symbolic reasons are written
to `rejected/<dataset-version>.jsonl`, with counters under `metadata`.

## Multi-sentence examples

Multi-sentence text is supported: place each complete source paragraph on one
line and its complete translation on the matching target line. The small model
configuration supports up to 512 SentencePiece tokens per side, which is often
enough for a roughly 2,000-character paragraph. Keep source and target within
that limit; the exact token count depends on vocabulary and punctuation. The
validation decoder uses the same 512-token limit, so these examples are
evaluated without an artificial shorter output cap.

## Alignment and versions

Only parallel text relies on line alignment. Tabular rows carry both sentences.
A dataset version is deterministic SHA-256 over ordered input paths and bytes
plus normalization, filtering, and split configuration. `processed/<version>`
contains accepted rows and a manifest with every input hash. `splits/<version>`
contains stable hash-based train, validation, and test files. Existing examples
keep their split when incoming data is added, and punctuation/case variants of
the same source are kept together to reduce leakage.

Add data, then run:

```bash
nmt dataset validate --pair en-uk
nmt dataset prepare --pair en-uk
```

Keep a copy of licensed original data. Dataset artifacts can reproduce platform
processing, but this repository does not grant redistribution rights for a real
corpus.
