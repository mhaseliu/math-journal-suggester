"""Frozen data and strict requests for the all-journal Kev experiment."""

import hashlib
from pathlib import Path
import unicodedata

from journal_suggester.io import (
    digest,
    read_json,
    read_jsonl,
    request_digest,
    write_json,
    write_jsonl,
)
from journal_suggester.records import identities
from journal_suggester.journal_choice import ordered_journals, build_request


PARTS = ("train", "screen", "validation")
SOURCE_FILES = {f"source/{name}.jsonl" for name in PARTS} | {
    "source/test-identities.jsonl",
    "source/journals.json",
    "configs/experiment.json",
    "configs/models.json",
}


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _file_record(path):
    return {"sha256": file_sha(path), "bytes": Path(path).stat().st_size}


def _identity_record(paper):
    keys = set(identities(paper))
    keys.update(
        ("paper", pid) for pid in [paper["paper_id"], *paper.get("versions", [])]
    )
    abstract = " ".join(
        unicodedata.normalize("NFKC", paper["abstract"]).casefold().split()
    )
    if not paper["title"].strip() or not abstract or not paper.get("group_id"):
        raise ValueError("Missing text or duplicate-group identity")
    return {
        "paper_id": paper["paper_id"],
        "group_id": paper["group_id"],
        "identity_hashes": sorted(digest(key) for key in keys),
        "abstract_hash": digest(abstract),
        "record_hash": digest(paper),
    }


def _unique_identity_sets(rows, name):
    keys, groups, abstracts, pids = set(), set(), set(), set()
    for row in rows:
        identity = _identity_record(row) if "abstract" in row else row
        new_keys = set(identity["identity_hashes"])
        if (
            identity["paper_id"] in pids
            or identity["group_id"] in groups
            or identity["abstract_hash"] in abstracts
            or keys & new_keys
        ):
            raise ValueError(
                f"Duplicate paper, identity, group or abstract within {name}"
            )
        pids.add(identity["paper_id"])
        groups.add(identity["group_id"])
        abstracts.add(identity["abstract_hash"])
        keys.update(new_keys)
    return pids, groups, keys, abstracts


def _check_isolation(parts):
    sets = {name: _unique_identity_sets(rows, name) for name, rows in parts.items()}
    for held in ("validation", "test"):
        for name in parts:
            if name == held or (held == "test" and name == "validation"):
                continue
            if any(a & b for a, b in zip(sets[held], sets[name])):
                raise ValueError(
                    f"Held-out identity/group/abstract overlap: {name} and {held}"
                )


def load_selection(root):
    """Verify the portable selection without reading original source directories."""
    root = Path(root)
    saved = read_json(root / "selection.json")
    if (
        set(saved["files"]) != SOURCE_FILES
        or not saved["checks_pass"]
        or saved["test_text_included"]
        or saved["test_scored"]
    ):
        raise ValueError("Unexpected selection file allowlist or test status")
    for name, expected in saved["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or root.resolve() not in path.resolve().parents
            or _file_record(path) != expected
        ):
            raise ValueError(f"Selected input was altered: {name}")
    if (
        read_json(root / "configs/experiment.json") != saved["config"]
        or read_json(root / "configs/models.json") != saved["models"]
    ):
        raise ValueError("Selected configuration changed")
    catalog = ordered_journals(read_json(root / "source/journals.json"))
    if [r["journal_id"] for r in catalog] != saved["candidate_ids"] or [
        r["journal_name"] for r in catalog
    ] != saved["choice_keys"]:
        raise ValueError("Selected journal ordering changed")
    parts = {name: read_jsonl(root / f"source/{name}.jsonl") for name in PARTS}
    test = read_jsonl(root / "source/test-identities.jsonl")
    allowed_test = {
        "paper_id",
        "group_id",
        "identity_hashes",
        "abstract_hash",
        "record_hash",
    }
    if any(set(row) != allowed_test for row in test):
        raise ValueError("Test identities contain unexpected fields")
    if {name: len(rows) for name, rows in parts.items()} != saved["counts"] or len(
        test
    ) != saved["test_identities"]:
        raise ValueError("Selected record count changed")
    _check_isolation(parts | {"test": test})
    return saved, parts


def _spread_indices(metadata, count):
    if not 1 <= count <= len(metadata):
        raise ValueError("Pilot size must fit the selected inputs")
    order = sorted(
        range(len(metadata)),
        key=lambda i: (metadata[i]["lengths"]["request_tokens"], i),
    )
    if count == 1:
        return [order[-1]]
    chosen = [order[round(i * (len(order) - 1) / (count - 1))] for i in range(count)]
    return [order[-1], *[i for i in chosen if i != order[-1]]]


def prepare(root, tokenizer=None):
    """Create immutable full/screen/pilot inputs. No weights are loaded here."""
    root = Path(root)
    if (root / "prepared.json").exists() or (root / "inputs").exists():
        raise ValueError("Preserve existing prepared inputs")
    if tokenizer is None:
        raise ValueError("Supply the pinned Kev tokenizer on the GPU host")
    selection, parts = load_selection(root)
    config, models = selection["config"], selection["models"]
    catalog = read_json(root / "source/journals.json")
    cache, rendered = {}, {}
    for name, papers in parts.items():
        requests, metadata = [], []
        for paper in papers:
            training = name != "validation"
            key = (digest(paper), training)
            if key not in cache:
                cache[key] = build_request(
                    paper,
                    catalog,
                    tokenizer,
                    label=paper["journal_id"] if training else None,
                    config=config,
                )
            request, lengths = cache[key]
            requests.append(request)
            metadata.append(
                {
                    "paper_id": paper["paper_id"],
                    "group_id": paper["group_id"],
                    "target": paper["journal_id"],
                    "candidates": selection["candidate_ids"],
                    "choice_keys": selection["choice_keys"],
                    "request_hash": request_digest(request),
                    "lengths": lengths,
                    "source_record_hash": digest(paper),
                }
            )
        rendered[name] = requests, metadata
    train_ix = _spread_indices(rendered["train"][1], config["pilot"]["training_papers"])
    val_ix = _spread_indices(
        rendered["validation"][1], config["pilot"]["validation_papers"]
    )
    modes = {
        "full": {"train": rendered["train"], "validation": rendered["validation"]},
        "screen": {"train": rendered["screen"], "validation": rendered["validation"]},
        "pilot": {
            "train": tuple([[rows[i] for i in train_ix] for rows in rendered["train"]]),
            "validation": tuple(
                [[rows[i] for i in val_ix] for rows in rendered["validation"]]
            ),
        },
    }
    files, counts, hashes = {}, {}, {}
    for mode, partitions in modes.items():
        counts[mode], hashes[mode] = {}, {}
        for partition, (requests, metadata) in partitions.items():
            counts[mode][partition] = len(requests)
            hashes[mode][partition] = [request_digest(request) for request in requests]
            for suffix, rows in (("", requests), (".meta", metadata)):
                relative = f"inputs/{mode}/{partition}{suffix}.jsonl"
                write_jsonl(root / relative, rows)
                files[relative] = _file_record(root / relative)
    prepared = {
        "checks_pass": True,
        "config": config,
        "models": models,
        "candidate_ids": selection["candidate_ids"],
        "choice_keys": selection["choice_keys"],
        "selection_hash": digest(selection),
        "files": files,
        "counts": counts,
        "request_hashes": hashes,
        "pilot_indices": {"train": train_ix, "validation": val_ix},
        "all_choices_retained": True,
        "validation_labels_in_requests": False,
        "reference_text_in_requests": False,
        "optimizer_updates": 0,
        "test_scored": False,
    }
    write_json(root / "prepared.json", prepared)
    return prepared
