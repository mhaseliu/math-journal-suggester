# Experiment

The audited corpus contains 10,061 reference papers, 1,000 validation papers and 1,000 test papers. Training uses 8,000 distinct papers from the reference pool. Duplicate identities/groups remain together; held-out papers and their versions are excluded from references. Training queries exclude themselves as evidence.

Publication-count-proportional sampling redistributes shortages. Validation and test guarantee at least one paper per journal; training has no floor. Publication metadata came from Crossref and publisher records; abstract sources also included OpenAlex and arXiv. Available abstracts, rather than complete journal coverage, determine usable records.

Qwen3-Embedding-8B is frozen. A journal's retrieval score is its closest eligible paper's cosine similarity. Kev receives 20 journals in retrieval order and two supporting excerpts per journal. Training inserts a missing target journal; validation/test never do. All papers count in accuracy, including retrieval misses. Conditional validation loss uses only retrieved targets.

Three two-epoch trials used the same 1,000 training and 1,000 validation papers. Peak rates `1e-5`, `2e-5`, `4e-5` reached Top 3 validation accuracy of 49.5%, 50.3%, 50.8%. The selected rate is a useful setting, not a demonstrated global optimum. [Tuning results](../results/tuning.json).

The final run warm-started released Kev's LoRA adapter and decision head; base weights stayed frozen. It used 8,000 papers, AdamW, peak `4e-5`, a five-epoch OneCycle schedule, batch one, accumulation eight, FP32 stored weights and BF16 autocast. Strict arithmetic disabled TF32 and reduced-precision accumulation. Fast kernels were checked before adoption.

Validation and saving occurred every half epoch. After at least two epochs, three checks without a loss improvement of 0.01 or a new Top 3 best triggered stopping. Small loss improvements accumulated. Training stopped at epoch 3.5; epoch 2 had the highest Top 3 accuracy (70.7%), with earlier checkpoints preferred on ties. The model was fixed before test scoring. [Configuration](../configs/training-8000.json) · [History](../results/training.json).

On test, the actual journal reached the candidate list for 850/1,000 papers. Top 3 was 724/1,000 after fine-tuning versus 406 for retrieval and 396 for released Kev. All three methods used identical natural candidates. Per-journal metrics and probabilities are available in `results/`; no model execution is needed to recompute the headline table.

IHÉS, Annals, Acta Mathematica, Inventiones, and JAMS contribute 105 training and 18 validation papers. For those 18 validation papers, retrieval found the actual journal six times and fine-tuned Kev ranked it among the first three once. Limited training coverage may contribute to weaker performance.

Earlier exploratory experiments included shuffled candidates, wider shortlists, small learning-rate screens, and an audited 3,000-paper run (51.0% Top 3 validation). They informed the final design; only the final fixed model was scored on this test. One seed and one pooled split were used. A new independent benchmark is needed for future decisions informed by these test results.

This repository is a portable release of the recorded experiment, not a new training run. Original executed-code hashes and split fingerprints are preserved in [provenance](../results/provenance.json). Website code is a demonstration interface; the research benchmark evaluates published venue matching.
