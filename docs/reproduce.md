# Reproduce the experiment

Start with the [installation instructions](../README.md#quickstart). Run commands from the repository root with your Python environment active. Checking saved results needs no GPU. Running Qwen or Kev does.

## Check the reported results

This recomputes test accuracy from the three saved prediction files:

```bash
python -m journal_suggester.experiment verify-results
```

## Prepare the data

Each paper needs these fields: `paper_id`, `title`, `abstract`, `journal_id`, `year`, `doi`, `arxiv_id`, and `url`. Store one record per line in UTF-8 JSONL. Use journal IDs from `data/journals.csv` and the journal publication year.

To collect papers, first list eligible publication records:

```bash
python -m journal_suggester.counts --output data/processed/counts
```

Resolve incomplete journal series using the publisher and series modules, then collect abstracts:

```bash
python -m journal_suggester.corpus --config configs/evaluation.json --output data/processed/corpus
```

Check coverage and keep the source records outside Git. For a small trial collection, use `journal-suggester collect --config configs/pilot.json --output data/processed/pilot`.

Apply `quality.clean_paper` to each record and set rejected records aside. Save accepted records to `data/processed/papers.jsonl`, then create splits that keep duplicate versions together:

```python
from journal_suggester.io import read_json
from journal_suggester.splits import split
split("data/processed/papers.jsonl", "artifacts/source/splits",
      train_size=8000, val_size=1000, test_size=1000, seed=42,
      counts=read_json("data/publication-counts.json"), evaluation_floor=1)
```

Keep the generated manifest. Splitting fails if journal coverage is insufficient.

To reproduce the original benchmark, match the paper IDs and `record_sha256` checksums in `data/splits/`. The original cleaned text is not distributed, and collecting it again may return different content. A new split is a new benchmark.

## Set up the GPU machine

Connect to your GPU machine before continuing. The tested setups are:

| Hardware | PyTorch | CUDA |
|---|---|---|
| GB10 ARM64 | 2.8.0 | 12.9 |
| B300 | 2.12.1 | 13.2 |

On **GB10**, use Python 3.12 and run `uv sync --project environments/gb10-speed --frozen`. Activate that environment and set `JOURNAL_EXECUTION_BACKEND=gb10` and `JOURNAL_CUDA_MEMORY_GIB=32`. Use `JOURNAL_TRITON_CUDA13=1` only with the pinned CUDA 13 ptxas/Triton 3.4 combination.

On **another CUDA machine**, follow [GPU setup](../environments/README.md) and set `CUDA_VISIBLE_DEVICES=0`, `JOURNAL_EXECUTION_BACKEND=cuda`, and `JOURNAL_ALLOW_MODEL_EXECUTION=1`. The tested B300 setup overrides Kev's `torch<2.9` requirement after numerical checks confirmed agreement. Other hardware is unverified.

## Train the model

With the data and GPU environment ready, run:

```bash
python -m journal_suggester.experiment select --splits artifacts/source/splits --output artifacts/experiment
python -m journal_suggester.experiment embed --root artifacts/experiment
python -m journal_suggester.experiment prepare --root artifacts/experiment
python -m journal_suggester.experiment train --root artifacts/experiment --pilot
python -m journal_suggester.experiment train --root artifacts/experiment
```

This uses 8,000 training and 1,000 validation papers for up to five epochs. Preparation checks candidates, supporting evidence, and tokenization. The pilot runs three updates and checks that saved weights give the same predictions after reloading. It must pass before training starts.

Validation loss covers papers whose actual journal was retrieved (845 in the recorded run). The test set stays unused. Existing outputs cannot be overwritten, so investigate failures before starting in a new directory.

## Evaluate the selected checkpoint

Select a checkpoint using the validation results in `artifacts/experiment/checkpoints/full/progress.json`. Once it is fixed, evaluate it and the released baseline on test:

```bash
python -m journal_suggester.experiment embed --root artifacts/experiment --partitions test
python -m journal_suggester.experiment evaluate --root artifacts/experiment --partition test --checkpoint PATH_TO_SELECTED_CHECKPOINT
```

## Run the website

On your GPU host, supply your reference records, vectors, and the verified fine-tuned adapter:

```bash
journal-suggester serve --split artifacts/experiment/splits --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

Open `http://127.0.0.1:8765` on the GPU machine, or use SSH port forwarding to access it from your laptop. For public hosting, configure a proxy and isolated services around `public_app` and `web_rpc`. Private host addresses and deployment scripts are excluded.
