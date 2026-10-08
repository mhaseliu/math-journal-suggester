"""End-to-end metrics keep every query, including targets outside the shortlist."""
import random
from collections import defaultdict


def metrics(rows, candidate_k=10):
    if not rows:
        raise ValueError("No evaluation queries")
    per = defaultdict(list)
    present, hit1, hit3, hit5, reciprocal = [], [], [], [], []
    for row in rows:
        target, ranking = row["target"], row["ranking"]
        if len(set(ranking)) != len(ranking) or set(ranking) != set(row["candidates"]):
            raise ValueError("Ranking must contain exactly the natural candidate set")
        rank = ranking.index(target) + 1 if target in ranking else float("inf")
        present.append(target in row["candidates"])
        hit1.append(rank <= 1)
        hit3.append(rank <= 3)
        hit5.append(rank <= 5)
        reciprocal.append(1 / rank)
        per[target].append(rank <= 3)
    n = len(rows)
    return {"n": n, f"candidate_recall_at_{candidate_k}": sum(present) / n, "top1": sum(hit1) / n,
            "top3": sum(hit3) / n, "top5": sum(hit5) / n, "mrr": sum(reciprocal) / n,
            "conditional_top1": sum(hit1) / sum(present) if any(present) else None,
            "macro_top3": sum(sum(v) / len(v) for v in per.values()) / len(per),
            "per_journal": {k: {"support": len(v), "top3": sum(v) / len(v)} for k, v in sorted(per.items())}}


def paired_top3(before, after, seed=42, samples=2000, *, allow_changed_inputs=False):
    left = {r["paper_id"]: r for r in before}
    right = {r["paper_id"]: r for r in after}
    if not left or len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError("Paired bootstrap requires identical unique query IDs")
    delta = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if a["target"] != b["target"] or (not allow_changed_inputs and
                (a["candidates"] != b["candidates"] or a.get("request_hash") != b.get("request_hash"))):
            raise ValueError("Paired inputs differ")
        delta.append(int(b["target"] in b["ranking"][:3]) - int(a["target"] in a["ranking"][:3]))
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(samples))
    return {"top3_delta": sum(delta) / len(delta), "ci95": [draws[int(samples * .025)], draws[min(samples - 1, int(samples * .975))]],
            "samples": samples, "seed": seed}
