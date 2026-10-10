"""Prepare, train, and evaluate the published 95-journal Kev recipe."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .io import (
    digest,
    journals,
    read_json,
    read_jsonl,
    request_digest,
    write_json,
    write_jsonl,
)
from .journal_choice import ordered_journals, build_request
from .training.data import (
    SOURCE_FILES,
    _file_record,
    _identity_record,
    _check_isolation,
    load_selection,
)


def select(
    split_dir, output, identifiers="data/splits", config="configs/kev-only.json"
):
    source, root, identifiers = Path(split_dir), Path(output), Path(identifiers)
    if root.exists():
        raise ValueError("Use a fresh experiment directory")
    spec, models = read_json(config), read_json("configs/models.json")
    originals = {
        part: read_jsonl(source / f"{part}.jsonl")
        for part in ("reference", "validation", "test")
    }
    manifest = read_json(identifiers / "manifest.json")
    for part, rows in originals.items():
        if digest(rows) != manifest["partition_hashes"][part]:
            raise ValueError(f"{part} records differ from the published benchmark")
    available = {p["paper_id"]: p for rows in originals.values() for p in rows}
    parts = {}
    for part in ("train", "screen", "validation", "test"):
        records = read_jsonl(identifiers / f"{part}.jsonl")
        rows = []
        for identity in records:
            paper = available.get(identity["paper_id"])
            if paper is None or digest(paper) != identity["record_sha256"]:
                raise ValueError(
                    f"Original record is missing or changed: {identity['paper_id']}"
                )
            rows.append(paper)
        parts[part] = rows
    catalog = ordered_journals(journals())
    candidate_ids = [j["journal_id"] for j in catalog]
    for part, count in (
        ("train", spec["full"]["training_papers"]),
        ("screen", spec["screen"]["training_papers"]),
        ("validation", spec["validation_papers"]),
        ("test", 1000),
    ):
        if len(parts[part]) != count or {p["journal_id"] for p in parts[part]} != set(
            candidate_ids
        ):
            raise ValueError(f"Unexpected {part} count or journal coverage")
    _check_isolation(parts)
    test = [_identity_record(p) for p in parts.pop("test")]
    for part, rows in parts.items():
        write_jsonl(root / f"source/{part}.jsonl", rows)
    write_jsonl(root / "source/test-identities.jsonl", test)
    write_json(root / "source/journals.json", catalog)
    write_json(root / "configs/experiment.json", spec)
    write_json(root / "configs/models.json", models)
    train_ids = {p["paper_id"] for p in parts["train"]}
    selection = {
        "checks_pass": True,
        "config": spec,
        "models": models,
        "fingerprint": manifest["fingerprint"],
        "candidate_ids": candidate_ids,
        "choice_keys": [j["journal_name"] for j in catalog],
        "files": {name: _file_record(root / name) for name in sorted(SOURCE_FILES)},
        "counts": {part: len(rows) for part, rows in parts.items()},
        "journal_counts": {
            part: dict(Counter(p["journal_id"] for p in rows))
            for part, rows in parts.items()
        },
        "screen_overlap_with_full": sum(
            p["paper_id"] in train_ids for p in parts["screen"]
        ),
        "test_identities": len(test),
        "test_text_included": False,
        "test_scored": False,
    }
    write_json(root / "selection.json", selection)
    load_selection(root)
    return {
        "checks_pass": True,
        "counts": selection["counts"],
        "test_text_included": False,
    }


def prepare(root):
    from .cloud_runtime import require_training_gpu

    require_training_gpu()
    from kev.model import load_tokenizer
    from .training.data import prepare as prepare_inputs

    selected, _ = load_selection(root)
    models = selected["models"]
    return prepare_inputs(
        root, load_tokenizer(models["base_model"], models["base_revision"])
    )


def evaluate(root, test_papers):
    """Score the validation-selected checkpoint once. Test never selects weights."""
    from .training.train import (
        load_inputs,
        select_checkpoint,
        apply_precision,
        verify_environment,
        score_model,
    )

    root = Path(root)
    output = root / "evaluation/test"
    if output.exists():
        raise ValueError("Preserve the existing test evaluation")
    prepared, inputs = load_inputs(root, "full")
    result = read_json(root / "full-result.json")
    if (
        result["status"] != "complete"
        or not result["checks_pass"]
        or result["test_scored"]
        or result["prepared_hash"] != digest(prepared)
        or result["selected"] != select_checkpoint(result["history"])
    ):
        raise ValueError("A completed validation-selected full run is required")
    papers = read_jsonl(test_papers)
    if [_identity_record(p) for p in papers] != read_jsonl(
        root / "source/test-identities.jsonl"
    ):
        raise ValueError("Test records differ from the excluded identities")
    checkpoint = Path(result["selected"]["checkpoint"])
    checkpoint_hashes = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in ("adapter_model.safetensors", "adapter_config.json", "head.pt")
    }
    write_json(
        output / "selection.json",
        {
            "checkpoint_hashes": checkpoint_hashes,
            "selected_epoch": result["selected"]["epoch"],
            "test_hash": digest(papers),
            "optimizer_updates": 0,
        },
    )
    torch = apply_precision()
    verify_environment(prepared["models"])
    from kev.checkpoint import Checkpoint, LoadOptions

    tokenizer, model = Checkpoint(str(checkpoint)).load(
        "cuda",
        LoadOptions(dtype=torch.float32, merge=False, fused=False, cuda_graphs=False),
    )
    sample = tuple(rows[:8] for rows in inputs["validation"])
    score_model(
        model,
        tokenizer,
        sample,
        prepared["candidate_ids"],
        prepared["config"],
        output / "parity",
        torch,
    )
    before = read_jsonl(checkpoint / "validation.predictions.jsonl")[:8]
    after = read_jsonl(output / "parity.predictions.jsonl")
    if (
        len(before) != 8
        or len(after) != 8
        or any(
            a["ranking"] != b["ranking"]
            or a["paper_id"] != b["paper_id"]
            or max(
                abs(a["probabilities"][j] - b["probabilities"][j])
                for j in prepared["candidate_ids"]
            )
            > 1e-6
            for a, b in zip(before, after)
        )
    ):
        raise ValueError("Selected-checkpoint reload parity failed")
    requests, metadata = [], []
    catalog = read_json(root / "source/journals.json")
    for paper in papers:
        request, lengths = build_request(
            paper,
            catalog,
            tokenizer,
            config=prepared["config"],
        )
        requests.append(request)
        metadata.append(
            {
                "paper_id": paper["paper_id"],
                "group_id": paper["group_id"],
                "target": paper["journal_id"],
                "candidates": prepared["candidate_ids"],
                "choice_keys": prepared["choice_keys"],
                "request_hash": request_digest(request),
                "lengths": lengths,
            }
        )
    report = score_model(
        model,
        tokenizer,
        (requests, metadata),
        prepared["candidate_ids"],
        prepared["config"],
        output / "kev",
        torch,
        partition="test",
    )
    if any(hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != value
           for name, value in checkpoint_hashes.items()):
        raise ValueError("Selected checkpoint changed during evaluation")
    write_json(output / "result.json", report)
    return report["metrics"]


def verify_results(directory="results", identifiers="data/splits"):
    """Recompute accuracy without a GPU, network, or manuscript text."""
    from .evaluation import metrics

    directory = Path(directory)
    rows = read_jsonl(directory / "predictions/kev_only.jsonl")
    expected = read_json(directory / "test.json")
    identities = read_jsonl(Path(identifiers) / "test.jsonl")
    candidates = [j["journal_id"] for j in ordered_journals(journals())]
    if (
        len(rows) != 1000
        or len({r["paper_id"] for r in rows}) != 1000
        or [(r["paper_id"], r["target"]) for r in rows]
        != [(p["paper_id"], p["journal_id"]) for p in identities]
    ):
        raise ValueError("Test predictions differ from published identities")
    for row in rows:
        probs = row["probabilities"]
        if (
            row["candidates"] != candidates
            or set(probs) != set(candidates)
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=1e-5)
            or row["ranking"] != sorted(probs, key=lambda j: (-probs[j], j))
        ):
            raise ValueError("Invalid probabilities, journal order, or ranking")
    measured = metrics(rows, 95)
    for key in ("top1", "top3", "top5", "macro_top3"):
        if not math.isclose(measured[key], expected["metrics"][key], abs_tol=1e-12):
            raise ValueError(f"Reported {key} differs from saved predictions")
    verification = read_json(directory / "verification.json")
    sha = hashlib.sha256(
        (directory / "predictions/kev_only.jsonl").read_bytes()
    ).hexdigest()
    if sha != verification["test_predictions_sha256"]:
        raise ValueError("Saved prediction file changed")
    return {
        "checks_pass": True,
        "test_papers": len(rows),
        "model_calls": 0,
        "metrics": {k: measured[k] for k in ("top1", "top3", "top5")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select")
    p.add_argument("--splits", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--identifiers", default="data/splits")
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p = sub.add_parser("train")
    p.add_argument("--root", required=True)
    p.add_argument("--mode", choices=["pilot", "screen", "full"], required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--root", required=True)
    p.add_argument("--test-papers", required=True)
    p = sub.add_parser("verify-results")
    p.add_argument("--directory", default="results")
    p.add_argument("--identifiers", default="data/splits")
    args = parser.parse_args()
    if args.command == "select":
        result = select(args.splits, args.output, args.identifiers)
    elif args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "train":
        from .training.train import run

        result = run(args.root, args.mode)
        result = {key: result[key] for key in ("status", "checks_pass")}
    elif args.command == "evaluate":
        result = evaluate(args.root, args.test_papers)
    else:
        result = verify_results(args.directory, args.identifiers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
