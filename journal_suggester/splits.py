"""Frozen, stratified, duplicate-group-disjoint random splits (seed 42)."""
import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .records import deduplicate


def quotas(total, weights, capacities):
    """Capped largest-remainder apportionment; redistribute shortages deterministically."""
    result = {k: 0 for k in sorted(capacities)}
    left = min(total, sum(capacities.values()))
    while left:
        active = [k for k in result if result[k] < capacities[k]]
        denominator = sum(weights.get(k, 0) for k in active)
        targets = {k: left * (weights.get(k, 0) / denominator if denominator else 1 / len(active)) for k in active}
        assigned = 0
        for k in active:
            n = min(capacities[k] - result[k], int(targets[k]))
            result[k] += n
            assigned += n
        left -= assigned
        for k in sorted(active, key=lambda k: (-(targets[k] % 1), k)):
            if left and result[k] < capacities[k]:
                result[k] += 1
                left -= 1
    return result


def floor_quotas(total, weights, minimum=1):
    """Fixed publication-weighted quotas; availability must not silently change them."""
    if minimum < 0 or total < minimum * len(weights) or any(v < 0 for v in weights.values()):
        raise ValueError("Invalid total, journal floor, or publication weights")
    extra = quotas(total - minimum * len(weights), weights, {k: total for k in weights})
    return {k: v + minimum for k, v in extra.items()}


def split(input_path, output, train_size=10000, val_size=1000, test_size=1000, seed=42, counts=None,
          evaluation_floor=None, reference_minimum=3):
    if min(train_size, val_size, test_size) < 0:
        raise ValueError("Split sizes must be nonnegative")
    papers, quarantine = deduplicate(read_jsonl(input_path))
    specification = {"papers": papers, "sizes": [train_size, val_size, test_size], "seed": seed, "counts": counts}
    if evaluation_floor is not None:
        specification.update(evaluation_floor=evaluation_floor, reference_minimum=reference_minimum)
    fingerprint = digest(specification)
    output = Path(output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Frozen split differs; use a new experiment directory")
        for name, expected in manifest.get("partition_hashes", {}).items():
            if digest(read_jsonl(output / f"{name}.jsonl")) != expected:
                raise ValueError(f"Frozen {name} partition was altered")
        return manifest
    groups = defaultdict(list)
    for paper in papers:
        groups[paper["journal_id"]].append(paper)
    for jid, group in groups.items():
        random.Random(f"{seed}:{jid}").shuffle(group)
    usable = {k: len(v) for k, v in groups.items()}
    # Keep at least three distinct works per journal in the reference pool when possible.
    budget = {k: max(0, v - 3) for k, v in usable.items()}
    if evaluation_floor is None:
        val_q = quotas(val_size, usable, budget)
        test_q = quotas(test_size, usable, {k: budget[k] - val_q[k] for k in budget})
    else:
        if not counts or set(usable) - set(counts):
            raise ValueError("Audited publication counts required for every collected journal")
        val_q = floor_quotas(val_size, counts, evaluation_floor)
        test_q = floor_quotas(test_size, counts, evaluation_floor)
        shortages = {k: val_q[k] + test_q[k] + reference_minimum - usable.get(k, 0)
                     for k in counts if usable.get(k, 0) < val_q[k] + test_q[k] + reference_minimum}
        if shortages:
            raise ValueError(f"Insufficient usable papers; split not frozen: {shortages}")
    pool, val, test = [], [], []
    for jid in sorted(groups):
        v, t = val_q[jid], test_q[jid]
        val.extend(groups[jid][:v])
        test.extend(groups[jid][v:v + t])
        pool.extend(groups[jid][v + t:])
    pool_counts = Counter(p["journal_id"] for p in pool)
    train_q = quotas(train_size, counts or usable, pool_counts)
    train, used = [], Counter()
    for p in pool:
        if used[p["journal_id"]] < train_q[p["journal_id"]]:
            train.append(p)
            used[p["journal_id"]] += 1
    manifest = {"fingerprint": fingerprint, "seed": seed, "usable": usable,
                "reference_size": len(pool), "train_size": len(train), "validation_size": len(val), "test_size": len(test),
                "requested_sizes": {"train": train_size, "validation": val_size, "test": test_size},
                "train_quotas": train_q, "validation_quotas": val_q, "test_quotas": test_q,
                "weight_source": "eligible_publication_counts" if counts else "usable_pilot_counts_not_publication_census",
                "rare_journals": [k for k, n in usable.items() if n < 5], "quarantined_groups": len(quarantine)}
    if evaluation_floor is not None:
        manifest.update(evaluation_floor=evaluation_floor, reference_minimum=reference_minimum,
                        evaluation_weights="eligible_publication_counts")
    partitions = {"reference": pool, "train": train, "validation": val, "test": test, "quarantine": quarantine}
    manifest["partition_hashes"] = {name: digest(rows) for name, rows in partitions.items()}
    for name, rows in partitions.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_json(manifest_path, manifest)
    return manifest
