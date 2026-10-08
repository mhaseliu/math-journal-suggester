# Run and reproduce

## What can be reproduced directly

Run `python -m journal_suggester.experiment verify-results` from a clone to recompute test accuracy from the three saved prediction files. It needs only Python. Split identities are in `data/splits/`; original content hashes and revisions are frozen. Recovering the exact historical corpus also requires the original cleaned text: fresh API responses may differ. The public release does not claim that a new crawl recreates those bytes.

## Data

Supply UTF-8 JSONL records containing `paper_id`, `title`, `abstract`, `journal_id`, `year`, `doi`, `arxiv_id`, and `url`. Journal IDs must match `data/journals.csv`. Use journal publication year. Keep source provenance outside Git.

A small real-data collection can start with `journal-suggester collect --config configs/pilot.json --output data/processed/pilot`. `journal_suggester.collect`, `corpus`, `quality`, and `counts` contain the collection and cleaning functions. Do not treat a pilot corpus as the full benchmark.

For a fresh metadata collection, `python -m journal_suggester.counts --output data/processed/counts` builds an eligible-record ledger; reconcile incomplete series using the publisher/series modules before collecting abstracts with `python -m journal_suggester.corpus --config configs/evaluation.json --output data/processed/corpus`. Inspect the coverage report and retain the provenance. New source responses may change coverage.

For a new full experiment, clean source records with `quality.clean_paper`, quarantine rejected records, then freeze duplicate-safe splits:

```python
from journal_suggester.io import read_json
from journal_suggester.splits import split
split("data/processed/papers.jsonl", "artifacts/source/splits",
      train_size=8000, val_size=1000, test_size=1000, seed=42,
      counts=read_json("data/publication-counts.json"), evaluation_floor=1)
```

The split fails on insufficient journal coverage. Preserve the generated manifest. To reconstruct the published split, match recovered records to `data/splits/` identifiers and verify every `record_sha256`; do not resplit and call it the original test.

## GPU environment

Run neural commands on your GPU host. The tested stacks are GB10 ARM64 (PyTorch 2.8.0/CUDA 12.9) and B300 (PyTorch 2.12.1/CUDA 13.2). Requirements and GB10's uv lock are in `environments/`. The B300 stack intentionally overrides pinned Kev's `torch<2.9` constraint; the experiment checked numerical agreement before using it. Other hardware is unverified.

On GB10 with Python 3.12 and uv: `uv sync --project environments/gb10-speed --frozen`; use that environment's Python. Set `JOURNAL_EXECUTION_BACKEND=gb10`, `JOURNAL_CUDA_MEMORY_GIB=32`, and, only with the pinned CUDA 13 ptxas/Triton 3.4 combination, `JOURNAL_TRITON_CUDA13=1`.

For a separately configured CUDA machine, set `CUDA_VISIBLE_DEVICES=0`, `JOURNAL_EXECUTION_BACKEND=cuda`, and `JOURNAL_ALLOW_MODEL_EXECUTION=1`. See `environments/README.md` for the B300 stack. This never selects an SSH host or allocates a cloud GPU for you. Use the three-update pilot and compare predictions before running a full experiment on a new stack.

## Standalone final recipe

These commands use only your frozen source splits and pinned models, with no earlier experiment directories:

```bash
python -m journal_suggester.experiment select --splits artifacts/source/splits --output artifacts/experiment
python -m journal_suggester.experiment embed --root artifacts/experiment
python -m journal_suggester.experiment prepare --root artifacts/experiment
python -m journal_suggester.experiment train --root artifacts/experiment --pilot
python -m journal_suggester.experiment train --root artifacts/experiment
```

The supported training recipe is the published 8,000/1,000-paper, maximum-five-epoch run. Conditional loss uses the naturally retrieved validation targets for your corpus (845 in the recorded run). `prepare` independently audits candidate/evidence order and native tokenizer retention. The pilot performs three updates and verifies checkpoint reload. Training refuses to run without it. Outputs are immutable; failures should be inspected before using a new directory. Preparation excludes test scoring.

Choose the checkpoint from `artifacts/experiment/checkpoints/full/progress.json` using validation. For a new run, evaluate that fixed checkpoint and the released baseline on test only after selection:

```bash
python -m journal_suggester.experiment embed --root artifacts/experiment --partitions test
python -m journal_suggester.experiment evaluate --root artifacts/experiment --partition test --checkpoint PATH_TO_SELECTED_CHECKPOINT
```

## Inference

On a configured GPU host, use your own reference records/vectors and the verified published adapter:

```bash
journal-suggester serve --split artifacts/experiment/splits --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

The preview binds to loopback. Use an SSH port-forward for personal access. The `public_app`/`web_rpc` modules retain isolated-worker and request-filtering logic; running an internet service additionally requires your own proxy, service isolation and host configuration. Private host addresses and service installers are deliberately excluded.
