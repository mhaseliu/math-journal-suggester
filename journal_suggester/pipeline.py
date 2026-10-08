"""Offline candidate generation, model requests, and retrieval evaluation."""
from pathlib import Path

from .evaluation import metrics
from .examples import build_request
from .io import journals, read_json, read_jsonl, request_digest, write_json, write_jsonl
from .retrieval import LexicalIndex, shortlist


def prepare(split_dir, output, vectors=None, config_path="configs/models.json", partitions=("train", "validation", "test"), candidate_order="shuffle"):
    split_dir, output = Path(split_dir), Path(output)
    config = read_json(config_path)
    names = {j["journal_id"]: j["journal_name"] for j in journals()}
    references = read_jsonl(split_dir / "reference.jsonl")
    if not references:
        raise ValueError("Empty reference pool")
    tokenizer = None
    if vectors:
        from .gpu import load_vectors, require_gb10
        require_gb10()
        from kev.checkpoint import Checkpoint
        from kev.model import load_tokenizer
        ck = Checkpoint(config["kev_model"] + "@" + config["kev_revision"])
        tokenizer = load_tokenizer(ck.meta.base, ck.meta.base_revision)
        reference_vectors = load_vectors(vectors, "reference", references, config)
    else:
        index = LexicalIndex(references)
    report = {"backend": "qwen3-embedding-8b" if vectors else "lexical-smoke", "reference_size": len(references),
              "covered_journals": sorted({p["journal_id"] for p in references}),
              "missing_catalog_journals": sorted(set(names) - {p["journal_id"] for p in references}),
              "split_fingerprint": read_json(split_dir / "manifest.json")["fingerprint"], "candidate_order": candidate_order}
    for partition in partitions:
        queries = read_jsonl(split_dir / f"{partition}.jsonl")
        if vectors and queries:
            query_vectors = load_vectors(vectors, partition, queries, config)
        requests, sidecar, evaluations = [], [], []
        for i, query in enumerate(queries):
            scores = (reference_vectors @ query_vectors[i]).tolist() if vectors else index.scores(query)
            candidates, info = shortlist(query, references, scores, target=query["journal_id"] if partition == "train" else None)
            request, lengths = build_request(query, candidates, names, tokenizer=tokenizer,
                                            label=query["journal_id"] if partition == "train" else None,
                                            query_tokens=config["query_tokens"], reference_tokens=config["reference_tokens"],
                                            max_state=config["max_state"], max_request=config["max_request"], candidate_order=candidate_order)
            requests.append(request)
            sidecar.append({"paper_id": query["paper_id"], "group_id": query["group_id"], "target": query["journal_id"],
                            "candidates": [c["journal_id"] for c in candidates], "evidence_ids": {c["journal_id"]: [p["paper_id"] for p in c["references"]] for c in candidates},
                            "request_hash": request_digest(request), "lengths": lengths, **info})
            if partition != "train":
                evaluations.append({"paper_id": query["paper_id"], "target": query["journal_id"], "ranking": info["natural_journals"],
                                    "candidates": info["natural_journals"], "request_hash": request_digest(request)})
            if (i + 1) % 100 == 0:
                print(f"Prepared {partition}: {i + 1}/{len(queries)}", flush=True)
        write_jsonl(output / f"{partition}.jsonl", requests)
        write_jsonl(output / f"{partition}.meta.jsonl", sidecar)
        report[partition] = {"n": len(queries), "insertions": sum(r["inserted"] for r in sidecar),
                             "insufficient_evidence_queries": sum(bool(r["missing_evidence"]) for r in sidecar)}
        # Do not score the held-out test during pilot development.
        if partition == "validation" and evaluations:
            write_jsonl(output / "retrieval.validation.predictions.jsonl", evaluations)
            report["validation"]["metrics"] = metrics(evaluations)
    write_json(output / "report.json", report)
    return report
