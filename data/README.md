# Data included here

`journals.csv` lists the 95 journals. `publication-counts.json` contains audited indexed publication counts. Split files contain paper identifiers, journal assignments, and checksums of the complete original records. They contain no titles or abstracts.

`train.jsonl` identifies the 8,000 full-training papers. `screen.jsonl` identifies the 1,000 learning-rate-screen papers, of which 994 also appear in full training. Both selections come from the reference pool. Validation and test are disjoint from that pool and from each other.

`record_sha256` uses `journal_suggester.io.digest`. See the [reproduction guide](../docs/reproduce.md) for recovering complete records.
