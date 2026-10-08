"""Native Kev optimization with half-epoch checkpoints and predeclared early stopping."""
import argparse
from dataclasses import replace
import hashlib
import inspect
from pathlib import Path
import sys

from .b300_precision import apply_precision
from .b300_train import score_model
from .early_stopping import EarlyStopping, conditional_cross_entropy
from .io import digest, read_json, read_jsonl, request_digest, write_json
from .ordered_train import ordered_variants, verify_kev_revision


def patched_main(source):
    flags = '        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True'
    anchor = '                if step == steps: break'
    if source.count(flags) != 1 or source.count(anchor) != 1:
        raise RuntimeError("Pinned native training loop changed; refusing to patch")
    source = source.replace(flags, '        journal_apply_precision()')
    return source.replace(anchor,
        '                journal_checked = journal_step_hook(locals())\n'
        '                if journal_checked is not None: last = time.time()\n'
        '                if journal_checked is True: steps = step\n' + anchor)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--journal-root", required=True)
    parser.add_argument("--journal-pilot", action="store_true")
    args, native = parser.parse_known_args()
    root = Path(args.journal_root)
    spec = read_json("configs/training-8000.json")
    assert (spec["training_papers"], spec["epochs"], spec["batch"], spec["accum"], spec["validation_every_steps"]) == (8000, 5, 1, 8, 500)
    assert spec["early_stopping"]["reset_patience_on_new_top3_best"]
    prepared = read_json(root / "prepared.json")
    assert spec == prepared["config"]
    assert read_json(root / "local-audit.json")["prepared_hash"] == digest(prepared)
    assert read_json(root / "local-audit.json")["checks_pass"]
    directory = root / ("pilot-input" if args.journal_pilot else "qwen")
    rows = read_jsonl(directory / "train.jsonl")
    n, epochs, interval = (24, 1, 1) if args.journal_pilot else (8000, 5, 500)
    total = n*epochs//8
    assert len(rows) == n
    if not args.journal_pilot:
        assert [request_digest(r) for r in rows] == prepared["train_request_hashes"]
    assert [request_digest(r) for r in read_jsonl(directory / "validation.jsonl")] == prepared["validation_request_hashes"]
    assert digest(read_jsonl(directory / "validation.meta.jsonl")) == prepared["validation_metadata_hash"]
    assert digest(read_jsonl(directory / "test.meta.jsonl")) == prepared["test_identity_hash"]
    torch = apply_precision()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(0.8, 160*2**30/torch.cuda.get_device_properties(0).total_memory))
    verify_kev_revision(read_json("configs/models.json")["kev_code_revision"])
    import kev.train as trainer
    from kev.checkpoint import Checkpoint, LoadOptions
    from safetensors.torch import load_file
    trainer.record_variants = ordered_variants
    validation_meta = read_jsonl(directory / "validation.meta.jsonl")
    expected_loss_n = sum(m["target"] in m["natural_journals"] for m in validation_meta)
    policy = EarlyStopping(spec["early_stopping"])
    history = []

    def cpu_state(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_state(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu_state(v) for v in value)
        return value

    def step_hook(v):
        step, seen, ep, mb = v["step"], v["seen"], v["ep"], v["mb"]
        if step % interval:
            return None
        a, model, meta, tok = v["a"], v["model"], v["meta"], v["tok"]
        assert seen == step*8 and v["sched"].total_steps == total
        assert len(v["reqs"]) == n and a.epochs == epochs
        epoch = seen/n
        path = Path(a.out)/f"step-{step:05d}"
        path.mkdir(parents=True, exist_ok=False)
        model.lm.save_pretrained(path)
        head = cpu_state(model.head.state_dict())
        saved = replace(meta, head=head, extra={"args": vars(a), "init_source": v["init_source"],
                                              "epoch": epoch, "optimizer_steps": step})
        trainer.finish_checkpoint(str(path), saved, tok)
        adapter = load_file(str(path/"adapter_model.safetensors"))
        assert len(adapter) == 496 and all(bool(torch.isfinite(x).all()) for x in adapter.values())
        assert len(head) == 4 and all(bool(torch.isfinite(x).all()) for x in head.values())
        print(f"Saved step {step}/{total}, epoch {epoch:g}; validating", flush=True)
        predictions = score_model(model, tok, directory, path/"validation", limit=8 if args.journal_pilot else None)
        validation = read_json(path/"validation.json")
        loss = conditional_cross_entropy(predictions)
        assert args.journal_pilot or loss["n"] == expected_loss_n
        state = policy.update(epoch, loss["cross_entropy"], validation["metrics"]["top3"])
        stop = state["stop"] and not args.journal_pilot
        entry = {"epoch": epoch, "optimizer_steps": step, "records_seen": seen, "checkpoint": str(path),
                 "metrics": validation["metrics"], "validation_loss": loss, "stopping": state, "test_scored": False}
        history.append(entry)
        write_json(path/"snapshot.json", {"checks_pass": True, "epoch": epoch, "optimizer_steps": step,
            "records_seen": seen, "scheduler_steps_total": total, "precision": "strict_fast", "test_scored": False})
        # Preserve optimizer, scheduler and RNG state for recovery tooling; never confuse a weight-only checkpoint with a resume point.
        optimizer = cpu_state(v["opt"].state_dict())
        assert len(optimizer["state"]) == 500
        payload = {"optimizer": optimizer, "scheduler": v["sched"].state_dict(),
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                   "python_rng": v["rng"].getstate(), "epoch_index": ep, "next_microbatch": mb+1,
                   "request_order": [request_digest(r) for r in v["reqs"]], "stopping": dict(policy.__dict__),
                   "optimizer_steps": step, "records_seen": seen, "specification": spec,
                   "prepared_hash": digest(prepared), "history": history}
        temporary = path/"training-state.pt.tmp"
        torch.save(payload, temporary)
        temporary.rename(path/"training-state.pt")
        write_json(path/"validation-loss.json", loss)
        best = min(history, key=lambda e: (-e["metrics"]["top3"], e["optimizer_steps"]))
        write_json(Path(a.out)/"progress.json", {"status": "early_stopped" if stop else "training",
            "planned_optimizer_steps": total, "completed_optimizer_steps": step, "history": history,
            "selected": best, "prepared_hash": digest(prepared), "test_scored": False})
        print(f"Validation top3={validation['metrics']['top3']:.3f}, loss={loss['cross_entropy']:.5f}, patience={state['bad_checks']}/3, stop={stop}", flush=True)
        if args.journal_pilot and step == total:
            ck = Checkpoint(str(path))
            loaded_tok, loaded = ck.load("cuda", LoadOptions(dtype=torch.float32, merge=False, fused=False, cuda_graphs=False))
            after = score_model(loaded, loaded_tok, directory, path/"reload", limit=8)
            delta = max(abs(x["probabilities"][k]-y["probabilities"][k]) for x,y in zip(predictions,after) for k in x["probabilities"])
            assert delta <= 1e-6
            saved_state = torch.load(path/"training-state.pt", map_location="cpu", weights_only=False)
            assert saved_state["optimizer_steps"] == 3 and saved_state["scheduler"]["total_steps"] == 3
            assert len(saved_state["optimizer"]["state"]) == 500
            write_json(Path(a.out)/"reload-check.json", {"checks_pass": True, "max_probability_delta": delta,
                "optimizer_scheduler_state_readable": True, "test_scored": False})
            del loaded
            torch.cuda.empty_cache()
        return stop

    source = inspect.getsource(trainer.main)
    patched = patched_main(source)
    trainer.__dict__.update(journal_apply_precision=apply_precision, journal_step_hook=step_hook)
    exec(compile(patched, "<pinned-kev-half-epoch-hook>", "exec"), trainer.__dict__)
    sys.argv = ["kev.train"] + native
    a = trainer.parse_args()
    assert a.epochs == epochs and a.batch == 1 and a.accum == 8 and a.lr == spec["learning_rate"]
    assert a.max_steps == 0 and a.weights_dtype == "fp32" and a.dtype == "bf16" and a.checkpointing
    assert not a.full_ft and a.max_state == 5504 and a.lora_targets == "all"
    assert not Path(a.out).exists()
    write_json(Path(a.out).parent/(Path(a.out).name+"-wrapper.json"), {
        "upstream_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_main_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "changes": ["strict arithmetic", "fixed candidate order", "half-epoch evaluation/checkpoint/optimizer state", "bounded early stop"],
        "optimizer_loss_scheduler": "unchanged native AdamW/cross-entropy/OneCycle; five-epoch schedule in full run"})
    trainer.main()
    progress = read_json(Path(a.out)/"progress.json")
    stats = read_json(Path(a.out)/"training_metrics.json")
    assert stats["optimizer_steps"] == progress["completed_optimizer_steps"]
    assert stats["records_seen"] == 8*stats["optimizer_steps"] and stats["rejected_records"] == stats["truncated_records"] == 0
    assert stats["requested_records"] == n*epochs
    progress.update(status="complete", stopped_early=stats["optimizer_steps"] < total)
    write_json(Path(a.out)/"progress.json", progress)


if __name__ == "__main__":
    main()
