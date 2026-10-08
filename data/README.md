# Data included here

`journals.csv` is the 95-journal catalog; `publication-counts.json` gives audited indexed counts, not a complete census. `splits/*.jsonl` contains original split membership, public identifiers and record hashes, without paper titles or abstracts. These files are manifests, not inputs to neural inference. `record_sha256` hashes the original complete cleaned record using `journal_suggester.io.digest`.

Training is a subset of references. Validation and test are disjoint from references and from each other. See `docs/reproduce.md` for recovering or preparing complete records under source terms.
