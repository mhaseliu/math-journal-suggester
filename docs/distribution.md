# Distribution

Project code and the fine-tuned adapter use Apache 2.0. Kev's code and weights, its Qwen backbone, and the optional Qwen embedding model declare Apache 2.0. Their attribution and original licenses are retained in `NOTICE` and `licenses/`. Dependencies retain their own licenses.

`data/splits/` contains factual paper identifiers, journal assignments and hashes of the original cleaned records. `results/predictions/` contains model outputs without manuscript text. These make split membership and reported metrics inspectable without redistributing collected abstracts.

The repository does not grant rights to publisher or arXiv content. Retrieve texts from their sources under the applicable terms, retain provenance, and check redistribution permissions before sharing a rebuilt corpus. Abstracts collected through metadata APIs can have different rights from bibliographic metadata.

The model package contains the selected LoRA adapter, decision head, tokenizer, model card, license and checksums. It excludes the base model, reference corpus/vectors, optimizer state and private training paths. Removing operational metadata does not change adapter bytes or head tensors. The export verification records the original weight hashes and the released manifest hash.

The adapter is hosted on Hugging Face. Downloading the public model does not require an account. No dataset of collected abstracts is included in that upload.
