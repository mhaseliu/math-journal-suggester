# Math journal suggester

Recommend math journals from a paper’s title and abstract using a fine-tuned Kev-4B model. Kev scores all 95 journal names directly.

| Test papers | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| 1,000 | **57.2%** | **78.9%** | **85.3%** |

The epoch-2.5 checkpoint was selected using validation. Test papers were excluded from training and model selection. This benchmark had been examined during earlier project development.

[Website](https://journal-suggester.tail34e410.ts.net) · [Model](https://huggingface.co/mhaseliu/kev-math-journal-suggester) · [Experiment](docs/experiment.md) · [Model card](model/README.md)

## Quickstart

Use Python 3.11 or newer. These commands install the code and check the saved results without a GPU:

```bash
git clone https://github.com/mhaseliu/math-journal-suggester.git
cd math-journal-suggester
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,web]'
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m journal_suggester.experiment verify-results
```

## Run the model

Follow the [GPU setup and model download instructions](docs/reproduce.md), then run:

```bash
journal-suggester serve --kev-run checkpoints/kev-math-journal-suggester
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) on the GPU machine, or use SSH port forwarding from your laptop. Pasting text works offline, but importing arXiv metadata requires internet access.

Journal recommendations need the Kev adapter and its pinned Qwen3.5-4B backbone. Similar-paper lookup is optional and uses separate reference records and Qwen3-Embedding-8B vectors. It does not influence journal ranking.

The repository includes code, dataset split identifiers and record checksums, aggregate results, and predictions for each test paper. **Collected titles and abstracts, reference vectors, model weights, credentials, and private deployment settings are excluded.** The adapter and decision head are distributed through Hugging Face.

## Scope

The experiment covers 95 math journals and series published from 2016 through 2025. Several journals can suit a paper. Recommendations do not estimate acceptance probability or mathematical quality. Abstract availability biases the sample, and highly selective journals have limited training coverage.

Code and adapter: [Apache 2.0](LICENSE). [Credits](NOTICE) · [Distribution](docs/distribution.md).
