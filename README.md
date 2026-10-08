# Math journal suggester

Recommend mathematics journals from a title and abstract, with similar published papers as evidence. Qwen3-Embedding-8B retrieves 20 journals; a fine-tuned Kev-4B reranks them.

| Held-out test · 1,000 papers | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Qwen retrieval | 21.1% | 40.6% | 51.4% |
| Qwen + released Kev | 22.3% | 39.6% | 49.7% |
| **Qwen + fine-tuned Kev** | **51.3%** | **72.4%** | **78.2%** |

The selected model improves Top 3 by **31.8 percentage points** over retrieval. The checkpoint was fixed using validation before scoring test. [Experiment](docs/experiment.md) · [Saved results](results/test.json) · [Model card](model/README.md).

## Quickstart

Python 3.11 or newer. These commands use no GPU, credentials, or network APIs after installation:

```bash
git clone https://github.com/mhaseliu/math-journal-suggester.git
cd math-journal-suggester
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,web]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m journal_suggester.experiment verify-results
.venv/bin/journal-suggester demo
```

Open `http://127.0.0.1:8765`. This is a clearly labelled **synthetic, lexical demo**. Pasting text works offline; importing arXiv metadata requires internet access.

## Real inference and training

[Reproduction guide](docs/reproduce.md) covers data, GPU environments, inference, preparation, the three-update pilot, full training, and final evaluation. Neural execution requires an explicitly configured GPU machine; the demo never silently loads a model.

The public release contains code, split identifiers and content hashes, aggregate reports, and per-paper test predictions. **Collected titles/abstracts, reference vectors, upstream weights, credentials, and private deployment settings are excluded.** The fine-tuned adapter is packaged for a separate Hugging Face release; see the model card for availability.

## Scope

The experiment covers 95 journals/series and pooled publication years 2016 through 2025. Observed publication venue is a weak label: several journals can suit a paper. Results do not estimate acceptance probability. Abstract availability biases the sample, and highly selective journals have sparse training coverage. The test is a random held-out split, not a future-year benchmark; upstream model pretraining overlap is unknown.

Code: [Apache 2.0](LICENSE). [Upstream credits](NOTICE) · [Data and model distribution](docs/distribution.md).
