# Run and reproduce

## What can be reproduced directly

Run `python -m journal_suggester.experiment verify-results` to recompute test accuracy from the three saved prediction files. This needs only Python. `data/splits/` records the paper identities and original content hashes. Recreating the exact corpus requires the original cleaned records, because collecting the data again may return different content.

## Data

Supply UTF-8 JSONL records containing `paper_id`, `title`, `abstract`, `journal_id`, `year`, `doi`, `arxiv_id`, and `url`. Journal IDs must match `data/journals.csv`, and `year` is the journal publication year. Keep source records and collection history outside Git.

For a small trial collection, run `journal-suggester collect --config configs/pilot.json --output data/processed/pilot`. The collection and cleaning functions are in `journal_suggester.collect`, `corpus`, `quality`, and `counts`. This trial dataset is too small to reproduce the full benchmark.

To collect a full dataset, run `python -m journal_suggester.counts --output data/processed/counts` to list eligible records. Resolve incomplete series using the publisher and series modules, then collect abstracts with `python -m journal_suggester.corpus --config configs/evaluation.json --output data/processed/corpus`. Check the coverage report and retain the source records. Coverage may change as sources are updated.

Clean records with `quality.clean_paper`, set rejected records aside, then freeze splits that keep duplicate versions together:

```python
from journal_suggester.io import read_json
from journal_suggester.splits import split
split("data/processed/papers.jsonl", "artifacts/source/splits",
      train_size=8000, val_size=1000, test_size=1000, seed=42,
      counts=read_json("data/publication-counts.json"), evaluation_floor=1)
```

Splitting fails if journal coverage is insufficient. Keep the generated manifest. Reconstructing the published split requires matching its `data/splits/` identifiers and every `record_sha256`. Creating a new split produces a different benchmark.

## GPU environment

Run Qwen and Kev on your GPU host. The tested environments are GB10 ARM64 (PyTorch 2.8.0/CUDA 12.9) and B300 (PyTorch 2.12.1/CUDA 13.2). Dependencies and GB10's uv lock are in `environments/`. The B300 setup overrides Kev's `torch<2.9` requirement after numerical checks confirmed agreement. Other hardware has not been verified.

On GB10, use Python 3.12 and run `uv sync --project environments/gb10-speed --frozen`. Use the resulting environment's Python. Set `JOURNAL_EXECUTION_BACKEND=gb10` and `JOURNAL_CUDA_MEMORY_GIB=32`. Set `JOURNAL_TRITON_CUDA13=1` only with the pinned CUDA 13 ptxas/Triton 3.4 combination.

On another configured CUDA machine, set `CUDA_VISIBLE_DEVICES=0`, `JOURNAL_EXECUTION_BACKEND=cuda`, and `JOURNAL_ALLOW_MODEL_EXECUTION=1`. See `environments/README.md` for the B300 setup. You must connect to the GPU machine yourself. Run the three-update pilot and compare predictions before starting a full experiment in a new environment.

## Standalone final recipe

These commands need your frozen splits and pinned models:

```bash
python -m journal_suggester.experiment select --splits artifacts/source/splits --output artifacts/experiment
python -m journal_suggester.experiment embed --root artifacts/experiment
python -m journal_suggester.experiment prepare --root artifacts/experiment
python -m journal_suggester.experiment train --root artifacts/experiment --pilot
python -m journal_suggester.experiment train --root artifacts/experiment
```

This recipe uses 8,000 training and 1,000 validation papers for up to five epochs. Validation loss covers papers whose actual journal was retrieved (845 in the recorded run). `prepare` checks candidate order, supporting evidence, and whether tokenization retains each input. Training requires a successful three-update pilot that also checks checkpoint reloading. Existing outputs cannot be overwritten, so inspect failures before starting in a new directory. Preparation does not score the test set.

Select a checkpoint using the validation results in `artifacts/experiment/checkpoints/full/progress.json`. Once it is fixed, evaluate it and the released baseline on test:

```bash
python -m journal_suggester.experiment embed --root artifacts/experiment --partitions test
python -m journal_suggester.experiment evaluate --root artifacts/experiment --partition test --checkpoint PATH_TO_SELECTED_CHECKPOINT
```

## Inference

On your GPU host, supply your reference records, vectors, and the verified fine-tuned adapter:

```bash
journal-suggester serve --split artifacts/experiment/splits --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

The preview is accessible only on the machine running it. Use SSH port forwarding to access it from your laptop. The `public_app` and `web_rpc` modules include worker isolation and request filtering. Hosting a public service also requires a proxy, isolated services, and host configuration. Private host addresses and deployment scripts are excluded.
