---
language: en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
library_name: kev
tags:
- mathematics
- journal-recommendation
- lora
- kev
---

# Kev for math journal recommendations

Fine-tuned from [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) to recommend math journals from a paper’s title and abstract. The model scores all 95 journal names in alphabetical order, without supporting papers or journal descriptions.

This package contains the **epoch-2.5 LoRA adapter, decision head, and tokenizer**. It requires the pinned Qwen3.5-4B-Base backbone. Journal ranking does not require an embedding model or reference corpus.

[Code and setup](https://github.com/mhaseliu/math-journal-suggester) · [Experiment](https://github.com/mhaseliu/math-journal-suggester/blob/main/docs/experiment.md)

## Results

| Test papers | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| 1,000 | **57.2%** | **78.9%** | **85.3%** |

The checkpoint was fixed using validation before this evaluation. Test papers were excluded from training and checkpoint selection. The benchmark had been examined during earlier project development. Saved per-paper predictions and verification code are available in the code repository.

## Training

Training used 8,000 papers from 95 math journals and series published from 2016 through 2025. Validation and test each contained 1,000 papers. The title and abstract are capped at 768 tokens. All 95 journal names remain in every request.

We continued training released Kev’s LoRA adapter and decision head using AdamW, a peak learning rate of `4e-5`, and a five-epoch OneCycle schedule. Each update accumulated eight single-paper batches. Validation ran every half epoch. Training stopped after three checks without a new best validation Top 3 score, following a minimum of two epochs. It stopped at epoch 4 and selected epoch 2.5, preferring the earlier checkpoint on ties.

Training and test evaluation ran on a B300 with FP32 stored weights and BF16 autocast. TF32 and reduced-precision accumulation were disabled. The experiment used one seed and one split.

## Use

Install the code and pinned GPU environment from the reproduction guide. Download this complete model package and run:

```bash
journal-suggester serve --kev-run checkpoints/kev-math-journal-suggester
```

For a JSON file containing `title` and `abstract`:

```bash
journal-suggester suggest --kev-run checkpoints/kev-math-journal-suggester --paper paper.json
```

The loader verifies the model checksums and uses the evaluated request format and precision. Load through the project or Kev so the decision head is included. Loading only the LoRA adapter through a text-generation interface does not reproduce these recommendations.

## Limitations

The model predicts observed publication venues. It does not estimate acceptance probability or mathematical quality. Several journals can suit the same paper. Abstract availability and journal proportions bias the sample. Highly selective journals have limited training coverage. Performance on future publications and overlap with upstream pretraining data remain untested.

## Files and license

The adapter uses Apache 2.0. Upstream credits are in `NOTICE`. Download the backbone separately under its license.

`manifest.json` records the input format and inference-file checksums. `SHA256SUMS` covers the package. Private training arguments were removed from `head.pt`, with head tensors and adapter weights verified unchanged. Optimizer state and collected paper text are excluded.
