"""Pinned native trainer, fixed candidate order, and epoch checkpoint/evaluation hooks."""
import argparse
from dataclasses import replace
import hashlib
import inspect
import math
from pathlib import Path
import sys
import time

from .b300_precision import apply_precision
from .evaluation import metrics
from .io import read_json, read_jsonl, request_digest, write_json, write_jsonl
from .ordered_train import ordered_variants, verify_kev_revision


def score_model(model, tok, directory, output, *, limit=None, partition="validation"):
    """Natural candidates, FP32 stored weights/BF16 autocast, no cache/merge/labels."""
    torch = apply_precision()
    from kev.api import SystemOneRequest, to_record
    from kev.model import training_context
    requests = read_jsonl(Path(directory) / f"{partition}.jsonl")
    metadata = read_jsonl(Path(directory) / f"{partition}.meta.jsonl")
    assert len(requests) == len(metadata) and requests
    if limit is not None:
        requests, metadata = requests[:limit], metadata[:limit]
    was_training = model.training
    model.eval()
    predictions, times = [], []
    try:
        for request, meta in zip(requests, metadata):
            assert request_digest(request) == meta["request_hash"] and not meta["inserted"]
            assert "label" not in request["questions"]["journal"]
            record, info = to_record(SystemOneRequest.model_validate(request))
            enc = model.encode(tok, record, max_state=5504, max_branch=training_context(5504)["max_branch"], strict=True)
            assert len(enc["ids"]) <= 6144
            start = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model.forward_batch([enc])[0][0].float().softmax(-1).cpu().tolist()
            times.append(time.monotonic() - start)
            probs = dict(zip(info[0]["keys"], probabilities))
            assert list(probs) == meta["candidates"] == meta["natural_journals"]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities)
            assert math.isclose(sum(probabilities), 1, abs_tol=1e-5)
            predictions.append({"paper_id": meta["paper_id"], "target": meta["target"],
                "candidates": meta["candidates"], "request_hash": meta["request_hash"],
                "probabilities": probs, "ranking": sorted(probs, key=lambda j: (-probs[j], j))})
            if len(predictions) % 100 == 0:
                print(f"Validation {len(predictions)}/{len(requests)} ({sum(times):.1f}s)", flush=True)
                write_jsonl(str(output) + ".partial.jsonl", predictions)
    finally:
        model.train(was_training)
    write_jsonl(str(output) + ".predictions.jsonl", predictions)
    result = {"metrics": metrics(predictions, candidate_k=20), "seconds": sum(times),
              "query_seconds": times, "test_scored": partition == "test",
              "inference": "FP32 weights, BF16 autocast, strict arithmetic, unmerged, full forward; no prefix cache"}
    write_json(str(output) + ".json", result)
    Path(str(output) + ".partial.jsonl").unlink(missing_ok=True)
    return predictions


def patched_main(source):
    """Only change arithmetic flags and insert an epoch hook; fail on source drift."""
    flags = '        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True'
    anchor = '                if step == steps: break'
    if source.count(flags) != 1 or source.count(anchor) != 1:
        raise RuntimeError("Pinned native training loop changed; refusing to patch")
    source = source.replace(flags, '        journal_apply_precision()')
    return source.replace(anchor,
        '                if journal_epoch_hook(model, meta, tok, a, ep, mb, plan, step, seen, init_source): last = time.time()\n' + anchor)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--journal-input", required=True)
    parser.add_argument("--journal-pilot", action="store_true")
    options, native_args = parser.parse_known_args()
    torch = apply_precision()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(0.8, 160 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    verify_kev_revision(read_json("configs/models.json")["kev_code_revision"])
    import kev.train as trainer
    from kev.checkpoint import Checkpoint
    from safetensors.torch import load_file
    trainer.record_variants = ordered_variants
    expected_n = 24 if options.journal_pilot else 1000
    expected_epochs = 1 if options.journal_pilot else 2
    expected_steps = expected_n * expected_epochs // 8

    def epoch_hook(model, meta, tok, a, ep, mb, plan, step, seen, init_source):
        if mb + 1 != len(plan):
            return
        assert a.epochs == expected_epochs and a.batch == 1 and a.accum == 8 and not a.full_ft
        assert step == (ep + 1) * expected_n // 8 and seen == (ep + 1) * expected_n
        directory = Path(a.out) / f"epoch-{ep + 1}"
        directory.mkdir(parents=True, exist_ok=False)
        model.lm.save_pretrained(directory)
        head = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        saved = replace(meta, head=head, extra={"args": vars(a), "init_source": init_source,
                                             "epoch": ep + 1, "optimizer_steps": step})
        trainer.finish_checkpoint(str(directory), saved, tok)
        adapter = load_file(str(directory / "adapter_model.safetensors"))
        assert len(adapter) == 496 and all(bool(torch.isfinite(v).all()) for v in adapter.values())
        assert len(head) == 4 and all(bool(torch.isfinite(v).all()) for v in head.values())
        write_json(directory / "snapshot.json", {"epoch": ep + 1, "records_seen": int(seen),
            "optimizer_steps": step, "scheduler_steps_total": expected_steps, "checks_pass": True,
            "precision": "strict_fast", "test_scored": False})
        print(f"Saved epoch {ep + 1}, step {step}/{expected_steps}; validating", flush=True)
        before = score_model(model, tok, options.journal_input, directory / "validation",
                             limit=8 if options.journal_pilot else None)
        if options.journal_pilot:
            from kev.checkpoint import LoadOptions
            ck = Checkpoint(str(directory))
            reloaded_tok, reloaded = ck.load("cuda", LoadOptions(dtype=torch.float32, merge=False, fused=False, cuda_graphs=False))
            after = score_model(reloaded, reloaded_tok, options.journal_input, directory / "reload", limit=8)
            delta = max(abs(x["probabilities"][k] - y["probabilities"][k])
                        for x, y in zip(before, after) for k in x["probabilities"])
            assert delta <= 1e-6, f"Checkpoint reload changed predictions: {delta}"
            write_json(directory / "reload-check.json", {"checks_pass": True, "n": 8, "max_probability_delta": delta})
            del reloaded
            torch.cuda.empty_cache()
        return True

    source = inspect.getsource(trainer.main)
    patched = patched_main(source)
    trainer.__dict__.update(journal_apply_precision=apply_precision, journal_epoch_hook=epoch_hook)
    exec(compile(patched, "<pinned-kev-epoch-hook>", "exec"), trainer.__dict__)
    sys.argv = ["kev.train"] + native_args
    parsed = trainer.parse_args()
    assert parsed.epochs == expected_epochs and parsed.batch == 1 and parsed.accum == 8
    assert parsed.max_steps == (3 if options.journal_pilot else 0) and parsed.dtype == "bf16"
    assert parsed.weights_dtype == "fp32" and parsed.checkpointing and not parsed.full_ft
    assert parsed.max_state == 5504 and parsed.lora_targets == "all"
    if Path(parsed.out).exists():
        raise ValueError("Checkpoint output already exists")
    # Do not create out itself: the native trainer enforces exclusive creation.
    write_json(Path(parsed.out).parent / (Path(parsed.out).name + "-wrapper.json"), {
        "upstream_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_main_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "changes": ["strict arithmetic flags", "fixed candidate order", "save/evaluate epoch boundaries"],
        "optimizer_loss_scheduler": "unchanged pinned native trainer; one continuous OneCycle over both epochs"})
    trainer.main()


if __name__ == "__main__":
    main()
