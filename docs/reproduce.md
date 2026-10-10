# Reproduce the experiment

Start with the [installation instructions](../README.md#quickstart). Run commands from the repository root with the Python environment active. Checking saved results does not need a GPU but running Kev does.

## Check saved results

```bash
python -m journal_suggester.experiment verify-results
```

This checks both models' predictions for all 1,000 test papers, their journal ordering and probabilities, and the reported accuracy. It also verifies that both models received identical requests.

## Run recommendations

Kev uses the Qwen3.5-4B-Base backbone with its trained adapter and decision head. It scores all 95 journal names directly. No separate embedding model or reference corpus is needed for ranking.

Use a GPU machine with the pinned environment. The tested setups are:

| Hardware | PyTorch | CUDA |
|---|---|---|
| ASUS Ascent GX10 (NVIDIA GB10, ARM64) | 2.8.0 | 12.9 |
| B300 | 2.12.1 | 13.2 |

On ASUS Ascent GX10, run `uv sync --project environments/gb10-speed --frozen` and activate that environment. Set `JOURNAL_EXECUTION_BACKEND=gb10` and `JOURNAL_CUDA_MEMORY_GIB=32`. The tested fast-kernel setup uses `JOURNAL_TRITON_CUDA13=1` with Triton 3.4 and `TRITON_PTXAS_PATH` pointing to the CUDA 13.0 `ptxas` binary.

For a CUDA host using the B300 stack, follow [environment setup](../environments/README.md). Set `CUDA_VISIBLE_DEVICES=0`, `JOURNAL_EXECUTION_BACKEND=cuda`, and `JOURNAL_ALLOW_MODEL_EXECUTION=1`. The package versions are checked before training. Other hardware is unverified.

Download the adapter package using the revision in `configs/release.json`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from journal_suggester.io import read_json
release = read_json('configs/release.json')
snapshot_download(release['model_repository'], revision=release['model_revision'],
                  local_dir='checkpoints/kev-math-journal-suggester')
PY
journal-suggester serve --kev-run checkpoints/kev-math-journal-suggester
```

The pinned backbone is downloaded on first load. Published test scores use B300. A separate [GB10 validation check](../results/deployment-validation.json) scored 76.1% Top 3 versus 76.2% on B300. The cleaned export and original checkpoint produced identical GB10 predictions. Inference pins the validated GB10 kernel settings so rebuilding the GPU cache does not change predictions. Open `http://127.0.0.1:8765` on the GPU machine, or use SSH port forwarding from your laptop.

### Optional similar papers

Similar-paper lookup uses a separate Qwen3-Embedding-8B model after Kev ranks the journals. These examples do not affect ranking. Supply a directory containing `reference.jsonl` and its vectors using `--split` and `--vectors`. Generate the vectors with `journal_suggester.gpu.embed_split` and `configs/models.json`. Allow additional GPU memory when loading both models.

## Recover the original data

Each cleaned paper is a JSONL record containing `paper_id`, `group_id`, `title`, `abstract`, `journal_id`, `year`, and available identifiers such as `doi` and `arxiv_id`. Preserve duplicate-group metadata from preprocessing.

The original collected text is not distributed. Recover records using the identifiers in `data/splits/` and match their `record_sha256` checksums. The source files are `reference.jsonl`, `validation.jsonl`, and `test.jsonl`. The selection command checks their full hashes and reconstructs the exact training and learning-rate-screen selections. Changed or missing records fail verification.

Collection and cleaning utilities remain in `counts`, `corpus`, `quality`, `records`, and `splits`. Recovering metadata from upstream services may return different content, so exact data reconstruction is not guaranteed.

## Prepare and train

Select the records on a CPU machine:

```bash
python -m journal_suggester.experiment select --splits data/recovered --output artifacts/experiment
```

On the GPU machine, prepare requests and run the separate training stages:

```bash
python -m journal_suggester.experiment prepare --root artifacts/experiment
python -m journal_suggester.experiment train --root artifacts/experiment --mode pilot
python -m journal_suggester.experiment train --root artifacts/experiment --mode screen
python -m journal_suggester.experiment train --root artifacts/experiment --mode full
```

The pilot checks three updates and checkpoint reloads. The screen compares three rates on 1,000 papers for two epochs. The full run starts from released Kev on 8,000 papers, using the selected rate and up to five epochs. Validation runs every half epoch. The stopping rule and checkpoint selection use overall validation Top 3.

Existing outputs cannot be overwritten. Reports, predictions, and saved training state stay inside the experiment directory. The code saves optimizer, scheduler, and RNG state but does not implement exact resume.

## Evaluate once the checkpoint is fixed

```bash
python -m journal_suggester.experiment evaluate --root artifacts/experiment --test-papers data/recovered/test.jsonl
```

This loads the checkpoint already selected by validation, checks its predictions after reload, and scores the original 1,000 test papers without training or selecting another checkpoint.

Add `--released` to the evaluation command to score released Kev before journal fine-tuning. It uses the same test papers, journal ordering, precision, and scoring code, with results saved separately.
