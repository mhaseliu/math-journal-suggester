# Experiment

The audited corpus contains 10,061 reference papers, 1,000 validation papers, and 1,000 test papers. Training used 8,000 reference papers. Versions stay in one split. Held-out papers are excluded from references, and training papers cannot retrieve themselves as evidence.

We sampled papers in proportion to each journal's publication count. If a journal had too few eligible papers, we filled the remaining places from other journals. Validation and test include at least one paper per journal. Training has no minimum. Publication metadata came from Crossref and publishers, with abstracts also collected from OpenAlex and arXiv. Abstract availability limited coverage.

Qwen3-Embedding-8B was used without fine-tuning. Each journal is scored by its closest eligible paper's cosine similarity. Kev receives 20 journals in retrieval order, with two supporting excerpts each. Training inserts the actual journal if missing. Validation and test use unchanged candidates. Accuracy counts every paper, while validation loss counts only papers whose journal was retrieved.

Three trials used the same 1,000 training and 1,000 validation papers for two epochs. Peak learning rates of `1e-5`, `2e-5`, and `4e-5` reached Top 3 validation accuracy of 49.5%, 50.3%, and 50.8%, respectively. We selected `4e-5` from these three settings. [Tuning results](../results/tuning.json).

The final run updated released Kev's LoRA adapter and decision head, keeping base weights frozen. It used 8,000 papers, AdamW, peak `4e-5`, and a five-epoch OneCycle schedule. Each update accumulated eight single-paper batches. Weights were stored in FP32 with BF16 autocast. TF32 and reduced-precision accumulation were disabled. Fast kernels were checked before use.

The model was validated and saved every half epoch. After at least two epochs, training stopped after three checks without a loss improvement of 0.01 or a new best Top 3 accuracy. Small loss improvements accumulated. Training ended at epoch 3.5. Epoch 2 had the highest Top 3 accuracy (70.7%) and was selected before test scoring, with earlier checkpoints preferred on ties. [Configuration](../configs/training-8000.json) · [History](../results/training.json).

On test, retrieval found the actual journal for 850 of 1,000 papers. Fine-tuned Kev placed it in the Top 3 for 724 papers, compared with 406 for retrieval and 396 for released Kev. All methods shared candidate lists. Saved predictions in `results/` let you recompute accuracy without running models.

IHÉS, Annals, Acta Mathematica, Inventiones, and JAMS contribute 105 training and 18 validation papers. For those 18 validation papers, retrieval found the actual journal six times and fine-tuned Kev ranked it among the first three once. Limited training coverage may contribute to weaker performance.

Earlier experiments explored shuffled candidates, wider shortlists, smaller learning-rate comparisons, and a 3,000-paper run (51.0% Top 3 validation). These informed the final design. Only the selected fine-tuned checkpoint was tested. We used one seed and one pooled split. Future decisions informed by these results need a new independent benchmark.

The code was reorganized for this release without repeating training. See [code and data versions](../results/provenance.json) for the original source commit and checksums of the evaluation code and dataset split. The benchmark measures journal matching on published papers.
