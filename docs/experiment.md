# Experiment

Kev-4B ranks all 95 journal names from a manuscript’s title and abstract. Names appear in a fixed alphabetical order. The model receives no supporting papers or journal descriptions. The title and abstract are capped at 768 tokens using the pinned tokenizer.

## Data

Training uses 8,000 papers. Validation and test each contain 1,000 papers, with at least one per journal. Paper counts approximately follow journal publication counts. When a journal lacked enough usable abstracts, the remaining places went to other journals.

The papers were published from 2016 through 2025. Records and abstracts came from Crossref, OpenAlex, publishers, and arXiv. Duplicate identities, titles, groups, and abstracts were checked across splits. The published identifiers and checksums allow recovered records to be matched to the original data. Collected paper text is not distributed.

## Learning-rate comparison

Three runs continued training released Kev’s LoRA adapter and decision head on the same 1,000 papers for two epochs. All used seed 42 and the same 1,000 validation papers. The screen and full training selections share 994 papers.

| Peak rate | Epoch 1 validation Top 3 | Epoch 2 validation Top 3 |
|---|---:|---:|
| `1e-5` | 48.0% | 49.7% |
| `2e-5` | 49.8% | 51.0% |
| `4e-5` | **51.9%** | 51.4% |

The screen selected `4e-5`. The full run started again from released Kev. These comparisons use one seed and do not establish an optimal learning rate.

## Full training

The full run used AdamW, a peak rate of `4e-5`, and a five-epoch OneCycle schedule. Each optimizer update accumulated eight single-paper batches. Validation ran every half epoch. After at least two epochs, three checks without a new best Top 3 score triggered early stopping. Ties favored the earlier checkpoint.

Training stopped at epoch 4 and selected epoch 2.5. Its validation Top 1, Top 3, and Top 5 were 54.5%, 76.2%, and 82.6%. [Full validation history](../results/training.json).

Training ran on a B300 with FP32 stored weights, BF16 autocast, and verified fast kernels. A three-update pilot checked gradients and checkpoint reloads. No training examples were rejected or truncated. The selected checkpoint reproduced its saved predictions exactly in the eight-paper reload check.

## Test

| Model | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Released Kev | 15.7% | 29.8% | 37.3% |
| Fine-tuned Kev | **57.2%** | **78.9%** | **85.3%** |

Both models scored the same 1,000 test papers with identical requests, all 95 journal names, precision, and B300 scoring code. Released Kev is the pretrained model before our journal fine-tuning. Fine-tuning improved Top 3 accuracy by **49.1 percentage points**.

Top 1, Top 3, and Top 5 measure whether the observed publication journal appears among the first one, three, or five recommendations. All 95 journals are candidates for every paper. Evaluation used the fixed epoch-2.5 checkpoint, with no additional training or test-based selection.

Test papers were excluded from training and validation. The benchmark had been examined during earlier project development, so this is not a fresh external evaluation. Results use one split and do not establish performance on future publications.

[Fine-tuned predictions](../results/predictions/kev_only.jsonl) · [Baseline predictions](../results/predictions/released_kev.jsonl) · [Baseline checks](../results/released-baseline.json) · [Test results](../results/test.json) · [Tuning results](../results/tuning.json).
