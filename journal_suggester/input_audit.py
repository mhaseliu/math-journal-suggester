"""Recompute natural journal/evidence ranks independently from saved vectors."""
from collections import defaultdict
from pathlib import Path

from .gpu import load_vectors
from .io import digest, journals, read_json, read_jsonl, request_digest, write_json
from .records import identities


def audit_inputs(source, query_vectors, partition, queries, directory, models, tokenizer, training=False):
    import numpy as np
    source, directory = Path(source), Path(directory)
    refs = read_jsonl(source / "splits/reference.jsonl")
    rv = load_vectors(source / "vectors", "reference", refs, models)
    qv = load_vectors(query_vectors, partition, queries, models)
    for vector_dir, name, rows, values, role in ((source / "vectors", "reference", refs, rv, "document"),
            (Path(query_vectors), partition, queries, qv, "query")):
        saved = read_json(vector_dir / f"{name}.json")
        assert saved["paper_ids"] == [p["paper_id"] for p in rows] and saved["role"] == role
        assert values.shape == (len(rows), 4096) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=.001)
    names = {j["journal_id"]: j["journal_name"] for j in journals()}
    keys, groups, by_journal = defaultdict(set), defaultdict(set), defaultdict(list)
    for i, ref in enumerate(refs):
        for key in identities(ref):
            keys[key].add(i)
        groups[ref["group_id"]].add(i)
        by_journal[ref["journal_id"]].append(i)
    pid_order = {pid: rank for rank, pid in enumerate(sorted(p["paper_id"] for p in refs))}
    requests = read_jsonl(directory / f"{partition}.jsonl")
    metadata = read_jsonl(directory / f"{partition}.meta.jsonl")
    assert len(requests) == len(metadata) == len(queries)
    def truncate(text, limit):
        return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:limit], skip_special_tokens=True)
    for n, (query, request, meta) in enumerate(zip(queries, requests, metadata)):
        assert query["paper_id"] == meta["paper_id"] and query["journal_id"] == meta["target"]
        assert request_digest(request) == meta["request_hash"]
        scores = rv @ qv[n]
        excluded = set(groups.get(query["group_id"], set()))
        for key in identities(query):
            excluded.update(keys.get(key, set()))
        evidence = {}
        for jid, indices in by_journal.items():
            eligible = [i for i in indices if i not in excluded]
            evidence[jid] = sorted(eligible, key=lambda i: (-float(scores[i]), pid_order[refs[i]["paper_id"]]))[:2]
            assert len(evidence[jid]) == 2
        natural = sorted(evidence, key=lambda j: (-float(scores[evidence[j][0]]), j))[:20]
        assert natural == meta["natural_journals"]
        order = list(natural)
        inserted = training and query["journal_id"] not in order
        if inserted:
            order[-1] = query["journal_id"]
        assert meta["inserted"] == inserted and meta["candidates"] == order
        assert list(request["questions"]["journal"]["criteria"]) == order
        expected = ["Manuscript:\n" + truncate(query["title"] + "\n" + query["abstract"], models["query_tokens"])]
        for jid in order:
            assert meta["evidence_ids"][jid] == [refs[i]["paper_id"] for i in evidence[jid]]
            expected.append("Candidate journal: " + names[jid])
            for i in evidence[jid]:
                expected.append("Reference: " + truncate(refs[i]["title"] + "\n" + refs[i]["abstract"], models["reference_tokens"]))
        assert request["state"] == "\n\n".join(expected)
        if training:
            assert request["questions"]["journal"]["label"] == query["journal_id"]
        else:
            assert "label" not in request["questions"]["journal"]
        if (n + 1) % 500 == 0:
            print(f"Independently audited {partition}: {n + 1}/{len(queries)}", flush=True)
    report = {"checks_pass": True, "records": len(queries), "vector_rows_roles_finiteness_norms_verified": True,
        "natural_candidates_and_all_evidence_recomputed": True, "exact_query_and_reference_rendering_verified": True,
        "self_and_duplicate_groups_excluded": True, "metadata_hash": digest(metadata), "test_scored": False}
    write_json(directory / f"{partition}.retrieval-audit.json", report)
    return report
