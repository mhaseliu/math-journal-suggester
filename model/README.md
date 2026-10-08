---
language: en
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
base_model_relation: adapter
library_name: kev
tags:
- mathematics
- journal-recommendation
- reranking
- lora
- kev
---

# Kev for mathematics journal recommendation

A warm-start fine-tune of [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b), using the pinned Qwen3.5-4B-Base backbone. It ranks a supplied list of candidate journals from a manuscript title/abstract and retrieved supporting excerpts. It does not generate arbitrary journal names.

This package contains the **epoch-2 LoRA adapter, decision head and tokenizer**. It is not a complete standalone language model and does not include the Qwen embedding model or reference corpus.

[Code and reproduction](https://github.com/mhaseliu/math-journal-suggester) · [Experiment](https://github.com/mhaseliu/math-journal-suggester/blob/main/docs/experiment.md)

## Results

All methods were evaluated on the same 1,000 held-out papers and natural Top 20 candidates. Epoch 2 was selected on validation before test evaluation.

| Method | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Qwen retrieval | 21.1% | 40.6% | 51.4% |
| Qwen + released Kev | 22.3% | 39.6% | 49.7% |
| **Qwen + this adapter** | **51.3%** | **72.4%** | **78.2%** |

All papers count, including 150 whose true journal was not retrieved. Saved per-paper predictions and verification code are in the code repository.

## Training

8,000 papers from 95 mathematics journals/series, publication years 2016 through 2025. Training and reference papers are disjoint from the 1,000 validation and 1,000 test papers and their duplicate groups. Metadata/abstracts were collected from Crossref, publisher records, OpenAlex and arXiv. Abstract text is not redistributed with this model.

Only released Kev's LoRA and decision head were updated. Peak learning rate `4e-5`, AdamW/OneCycle, microbatch one, accumulation eight; maximum five epochs. Half-epoch validation triggered early stopping at epoch 3.5; epoch 2 was selected. FP32 stored weights, BF16 autocast, strict arithmetic and verified fast kernels. The full experiment used a B300; GB10 was used for final inference. One seed/split was tested.

## Use

Install the pinned GPU environment from the code repository on your GPU host. Download this whole model directory to `checkpoints/kev-math-epoch2`, then supply your own reference corpus and Qwen3-Embedding-8B vectors:

```bash
journal-suggester serve --split artifacts/experiment/splits --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

The loader verifies the public manifest before loading. It uses the original full-forward FP32-weight/BF16-autocast inference path. Standard Transformers text generation or a PEFT-only load omits Kev's decision head and is not the intended use.

## Limitations

Predicts observed publication venue as a proxy for journal fit; it does not estimate acceptance or judge mathematical significance. Abstract availability and journal proportions bias the sample. Rare/selective journals have limited coverage. The test pools 2016–2025 publications rather than measuring future-year generalization; possible pretraining overlap is unknown. Several journals may be appropriate even when the original venue is not recommended.

## Files and license

Apache 2.0; upstream Kev/Qwen attribution is in `NOTICE`. Base/embedding weights retain their upstream licenses and are downloaded separately. `manifest.json` covers the inference files. `SHA256SUMS` covers the package. Training paths and arguments were removed from `head.pt`; adapter bytes and head tensors were checked unchanged. Optimizer state and collected paper text are excluded.
