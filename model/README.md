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

# Kev for math journal recommendations

Fine-tuned from [jaredpalmer/kev-4b](https://huggingface.co/jaredpalmer/kev-4b) with the pinned Qwen3.5-4B-Base model. Given a manuscript's title, abstract, and supporting excerpts, it ranks the supplied candidate journals.

This package contains the **epoch-2 LoRA adapter, decision head, and tokenizer**. You also need the Qwen base model, embedding model, and reference corpus.

[Code and reproduction](https://github.com/mhaseliu/math-journal-suggester) · [Experiment](https://github.com/mhaseliu/math-journal-suggester/blob/main/docs/experiment.md)

## Results

All methods used the same 1,000 test papers and Top 20 retrieved candidates. Epoch 2 was selected using validation before testing.

| Method | Top 1 | Top 3 | Top 5 |
|---|---:|---:|---:|
| Qwen retrieval | 21.1% | 40.6% | 51.4% |
| Qwen + released Kev | 22.3% | 39.6% | 49.7% |
| **Qwen + this adapter** | **51.3%** | **72.4%** | **78.2%** |

Accuracy includes all papers, including 150 whose actual journal was not retrieved. The code repository contains predictions for each paper and code to verify the results.

## Training

Training used 8,000 papers from 95 math journals and series published from 2016 through 2025. Validation and test each contained 1,000 papers, with all versions excluded from training and references. Metadata and abstracts came from Crossref, publishers, OpenAlex, and arXiv.

We updated Kev's LoRA adapter and decision head with AdamW, a peak learning rate of `4e-5`, and a five-epoch OneCycle schedule. Each update accumulated gradients from eight single-paper batches. Validation ran every half epoch. Training stopped early at epoch 3.5, and epoch 2 was selected.

Training used a B300 with FP32 weights, BF16 autocast, and verified fast kernels. TF32 and reduced-precision accumulation were disabled. Final inference ran on an ASUS Ascent GX10 with an NVIDIA GB10 chip. The experiment used one seed and one split.

## Use

Set up the GPU environment described in the code repository. Place the complete model package in `checkpoints/kev-math-epoch2`, then supply your reference corpus and Qwen3-Embedding-8B vectors:

```bash
journal-suggester serve --split artifacts/experiment/splits --vectors artifacts/experiment/vectors --kev-run checkpoints/kev-math-epoch2
```

The Kev loader verifies the inference-file manifest and runs full-forward inference with FP32 weights and BF16 autocast. Load the adapter through Kev so its decision head is included.

## Limitations

The model matches papers to their observed publication venues. It does not predict acceptance or assess mathematical significance. Abstract availability and journal proportions bias the sample, and selective journals have limited coverage. The test mixes publication years from 2016 through 2025. Performance on future publications and overlap with pretraining data remain untested. Several journals can suit the same paper.

## Files and license

The adapter uses Apache 2.0. Upstream credits are in `NOTICE`. Base and embedding models are downloaded separately under their own licenses.

`manifest.json` lists checksums for the inference files. `SHA256SUMS` covers the whole package. Private training paths and arguments were removed from `head.pt`, with adapter weights and head tensors verified unchanged. Optimizer state and collected paper text are excluded.
