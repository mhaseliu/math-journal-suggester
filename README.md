# Math journal suggester

Recommend math journals from a title and abstract, with similar published papers as evidence. Qwen3-Embedding-8B retrieves 20 journals, then a fine-tuned Kev-4B reranks them.

| Held-out test · 1,000 papers | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Qwen retrieval | 21.1% | 40.6% | 51.4% |
| Qwen + released Kev | 22.3% | 39.6% | 49.7% |
| **Qwen + fine-tuned Kev** | **51.3%** | **72.4%** | **78.2%** |

The selected model improves Top 3 by **31.8 percentage points** over retrieval. The checkpoint was fixed using validation before scoring test. [Experiment](docs/experiment.md) · [Saved results](results/test.json) · [Model card](model/README.md).

## Quickstart

Use Python 3.11 or newer. Check `python3 --version` and use a newer Python executable below if needed. Cloning requires GitHub access while this repository is private. These commands install the code and verify the saved results without a GPU:

```bash
git clone https://github.com/mhaseliu/math-journal-suggester.git
cd math-journal-suggester
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,web]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m journal_suggester.experiment verify-results
```

## Running Qwen and Kev

[Reproduction guide](docs/reproduce.md) covers data preparation, GPU setup, generating recommendations, fine-tuning, and evaluation. Running Qwen and Kev requires a configured GPU machine with the model weights and reference data.

After that setup, start the website from the GPU environment:

```bash
journal-suggester serve --split artifacts/experiment/splits \
  --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

Keep the terminal running and open [http://127.0.0.1:8765](http://127.0.0.1:8765) on that machine, or use SSH port forwarding to access it from your laptop. Add `--port 8766` if the port is occupied. Press Ctrl+C to stop the server. Pasting text works offline, but importing arXiv metadata requires internet access.

The repository includes the code, records identifying which papers belong to each dataset split, and checksums for verifying the original paper records. It also includes overall results and each method's predictions for individual test papers, so the reported accuracy can be checked without rerunning the models.

**Collected titles/abstracts, reference vectors, upstream weights, credentials, and private deployment settings are excluded.** The fine-tuned adapter is packaged for a separate Hugging Face release. See the model card for availability.

## Scope

The experiment covers 95 journals/series and pooled publication years 2016 through 2025. Observed publication venue is a weak label: several journals can suit a paper. Results do not estimate acceptance probability. Abstract availability biases the sample, and highly selective journals have sparse training coverage.

Code: [Apache 2.0](LICENSE). [Upstream credits](NOTICE) · [Data and model distribution](docs/distribution.md).
