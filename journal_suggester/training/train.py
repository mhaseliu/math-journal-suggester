"""Remote native Kev fine-tuning with all 95 journals and no retrieval."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from journal_suggester.evaluation import metrics
from journal_suggester.io import digest, read_json, read_jsonl, request_digest, write_json, write_jsonl
from journal_suggester.kev_runtime import released, training_command
from journal_suggester.ordered_train import ordered_variants, verify_kev_revision


MODES = ("pilot", "screen", "full")
PACKAGES = {"torch": "2.8.0+cu129", "triton": "3.4.0", "transformers": "5.17.0",
            "peft": "0.21.0", "fla-core": "0.5.2", "causal-conv1d": "1.7.0"}
B300_PACKAGES = {**PACKAGES, "torch": "2.12.1+cu132", "triton": "3.7.1"}


def execution_backend():
    backend = os.environ.get("JOURNAL_EXECUTION_BACKEND", "gb10")
    if backend not in ("gb10", "modal-b300", "cuda"):
        raise RuntimeError(f"Unknown execution backend: {backend}")
    return backend


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class Top3Stopping:
    """Count checks without a strict Top 3 improvement; ties retain earlier weights."""
    def __init__(self, minimum_epochs=2, patience=3):
        self.minimum_epochs, self.patience = minimum_epochs, patience
        self.best = None
        self.bad_checks = 0
        self.last_epoch = 0

    def update(self, epoch, top3):
        if not math.isfinite(epoch) or epoch <= self.last_epoch or not math.isfinite(top3) or not 0 <= top3 <= 1:
            raise ValueError("Invalid or out-of-order validation measurement")
        improved = self.best is None or top3 > self.best
        if improved:
            self.best = top3
        self.bad_checks = 0 if improved else self.bad_checks + 1
        self.last_epoch = epoch
        return {"stop": epoch >= self.minimum_epochs and self.bad_checks >= self.patience,
                "improved": improved, "bad_checks": self.bad_checks, "best_top3": self.best}


def select_checkpoint(history):
    if not history:
        raise ValueError("No evaluated checkpoints")
    return min(history, key=lambda item: (-item["metrics"]["top3"], item["epoch"]))


def select_rate(trials, expected_rates):
    if len(trials) != len(expected_rates) or {t["learning_rate"] for t in trials} != set(expected_rates):
        raise ValueError("The complete configured learning-rate screen is required")
    if any(not t.get("checks_pass") or t.get("mode") != "screen" for t in trials):
        raise ValueError("Every screen trial must have passed")
    return min(trials, key=lambda t: (-t["selected"]["metrics"]["top3"], t["learning_rate"], t["selected"]["epoch"]))


def plan(spec, mode):
    if mode not in MODES or spec["candidate_count"] != 95 or spec["candidate_order"] != "alphabetical" or spec["choice_format"] != "journal_name_only":
        raise ValueError("Expected the fixed 95-journal Kev-only experiment")
    if spec["batch"] != 1 or spec["accum"] != 8 or spec["test_evaluation"]:
        raise ValueError("Expected batch 1, accumulation 8, and no test evaluation")
    n, epochs = spec[mode]["training_papers"], spec[mode]["epochs"]
    if type(n) is not int or type(epochs) is not int or n <= 0 or n % 8 or epochs < 1:
        raise ValueError("Training must contain complete optimizer updates")
    interval = n // 8 if mode == "pilot" else n / 8 * spec[mode]["validation_every_epochs"]
    if interval < 1 or interval != int(interval) or n * epochs // 8 % int(interval):
        raise ValueError("Validation must land on complete updates including the final update")
    validation_n = spec["pilot"]["validation_papers"] if mode == "pilot" else spec["validation_papers"]
    if mode == "pilot" and (n, epochs, validation_n) != (24, 1, 8):
        raise ValueError("The pilot requires 24 training records, three updates and eight validation records")
    return {"training_papers": n, "epochs": epochs, "validation_papers": validation_n,
            "validation_every_steps": int(interval), "optimizer_steps": n * epochs // 8}


def validate_rows(requests, metadata, candidate_ids, training, choice_keys=None):
    if len(candidate_ids) != 95 or len(set(candidate_ids)) != 95:
        raise ValueError("Exactly 95 distinct, fixed journal candidates are required")
    if not requests or len(requests) != len(metadata):
        raise ValueError("Request/metadata count mismatch")
    choice_keys = choice_keys or metadata[0].get("choice_keys")
    if not choice_keys or len(choice_keys) != 95 or len(set(choice_keys)) != 95:
        raise ValueError("Exactly 95 distinct journal names are required")
    if len({m["paper_id"] for m in metadata}) != len(metadata) or len({m["group_id"] for m in metadata}) != len(metadata):
        raise ValueError("Duplicate query identity or group")
    for request, meta in zip(requests, metadata):
        if set(request["questions"]) != {"journal"}:
            raise ValueError("One journal-choice question is required")
        question = request["questions"]["journal"]
        if (list(question["criteria"]) != choice_keys or meta["candidates"] != candidate_ids
                or meta.get("choice_keys") != choice_keys or any(value is not None for value in question["criteria"].values())):
            raise ValueError("Journal candidates were dropped, reordered or changed")
        if meta["target"] not in candidate_ids or request_digest(request) != meta["request_hash"]:
            raise ValueError("Request hash or target mismatch")
        if training:
            if question.get("label") != choice_keys[candidate_ids.index(meta["target"])]:
                raise ValueError("Incorrect training label")
        elif "label" in question:
            raise ValueError("Validation requests must not contain labels")


def load_inputs(root, mode):
    from journal_suggester.training.data import load_selection
    root = Path(root)
    prepared = read_json(root / "prepared.json")
    if not prepared.get("checks_pass"):
        raise ValueError("Prepared-input checks must pass before model execution")
    selection, _ = load_selection(root)
    if digest(selection) != prepared["selection_hash"]:
        raise ValueError("Selection manifest changed after preparation")
    if any(prepared[key] != selection[key] for key in ("config", "models", "candidate_ids", "choice_keys")):
        raise ValueError("Prepared settings differ from the frozen selection")
    expected = plan(prepared["config"], mode)
    names = {f"inputs/{m}/{f}" for m in MODES for f in
             ("train.jsonl", "train.meta.jsonl", "validation.jsonl", "validation.meta.jsonl")}
    if names != set(prepared["files"]):
        raise ValueError("Missing prepared input file hashes")
    for name, saved in prepared["files"].items():
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe prepared input path")
        if path.stat().st_size != saved["bytes"] or file_hash(path) != saved["sha256"]:
            raise ValueError(f"Prepared input changed: {name}")
    parts = {}
    for partition in ("train", "validation"):
        requests = read_jsonl(root / f"inputs/{mode}/{partition}.jsonl")
        metadata = read_jsonl(root / f"inputs/{mode}/{partition}.meta.jsonl")
        validate_rows(requests, metadata, prepared["candidate_ids"], partition == "train", prepared["choice_keys"])
        required_n = expected["training_papers"] if partition == "train" else expected["validation_papers"]
        if len(requests) != required_n:
            raise ValueError("Input count does not match the experiment specification")
        parts[partition] = (requests, metadata)
    if {m["group_id"] for m in parts["train"][1]} & {m["group_id"] for m in parts["validation"][1]}:
        raise ValueError("Training/validation group overlap")
    return prepared, parts


def checked_encoding(encoded, candidates, max_request):
    if encoded.get("state_truncated") or len(encoded["ids"]) > max_request:
        raise ValueError("The strict context budget was exceeded")
    if len(encoded["opt_idx"]) != 1 or len(encoded["opt_idx"][0]) != len(candidates) or len(candidates) != 95:
        raise ValueError("Not all 95 options survived encoding")


def patched_main(source):
    replacements = {
        '        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True':
        '        journal_apply_precision()',
        '                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_':
        '                journal_gradient_hook(locals())\n                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_',
        '                if step == steps: break':
        '                journal_checked = journal_step_hook(locals())\n'
        '                if journal_checked is not None: last = time.time()\n'
        '                if journal_checked is True: steps = step\n'
        '                if step == steps: break',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise RuntimeError("Pinned native training source changed; refusing to patch")
        source = source.replace(before, after)
    return source


def apply_precision():
    from journal_suggester.cloud_runtime import require_training_gpu
    os.environ["JOURNAL_CUDA_MEMORY_GIB"] = "32"
    if execution_backend() == "gb10":
        os.environ.setdefault("JOURNAL_TRITON_CUDA13", "1")
    torch = require_training_gpu(40)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(32 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    return torch


def verify_environment(models):
    verify_kev_revision(models["kev_code_revision"])
    backend = execution_backend()
    expected_packages = B300_PACKAGES if backend in ("modal-b300", "cuda") else PACKAGES
    versions = {name: importlib.metadata.version(name) for name in expected_packages}
    if versions != expected_packages:
        raise RuntimeError(f"Unexpected {backend} package versions: {versions}")
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from causal_conv1d import causal_conv1d_fn
    for wrapper, expected in ((qwen.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
                              (qwen.causal_conv1d_fn, causal_conv1d_fn)):
        if inspect.getclosurevars(wrapper).nonlocals.get("implementation") is not expected:
            raise RuntimeError("The verified fast kernels must be active")
    return {"backend": backend, "versions": versions, "fast_kernels_verified": True, "cuda_memory_gib": 32}


def score_model(model, tokenizer, inputs, candidate_ids, spec, output, torch, *, partition="validation"):
    if partition not in ("validation", "test"):
        raise ValueError("Expected validation or test scoring")
    from kev.api import SystemOneRequest, to_record
    from kev.model import training_context
    requests, metadata = inputs
    validate_rows(requests, metadata, candidate_ids, False)
    was_training = model.training
    model.eval()
    predictions, timings = [], []
    try:
        for request, meta in zip(requests, metadata):
            record, info = to_record(SystemOneRequest.model_validate(request))
            if info[0]["keys"] != meta["choice_keys"]:
                raise ValueError("Native API changed candidate order")
            encoded = model.encode(tokenizer, record, max_state=spec["max_state"],
                                   max_branch=training_context(spec["max_state"])["max_branch"], strict=True)
            checked_encoding(encoded, candidate_ids, spec["max_request"])
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_batch([encoded])[0][0].float()
                if logits.numel() != 95 or not bool(torch.isfinite(logits).all()):
                    raise ValueError("Invalid 95-journal model output")
                probabilities = logits.softmax(-1).cpu().tolist()
                target_log_probability = float(logits.log_softmax(-1)[candidate_ids.index(meta["target"])])
            timings.append(time.monotonic() - start)
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-5):
                raise ValueError("Invalid normalized probabilities")
            probs = dict(zip(candidate_ids, probabilities))
            predictions.append({"paper_id": meta["paper_id"], "target": meta["target"],
                                "request_hash": meta["request_hash"], "candidates": candidate_ids,
                                "probabilities": probs, "target_log_probability": target_log_probability,
                                "ranking": sorted(probs, key=lambda j: (-probs[j], j))})
            if len(predictions) % 100 == 0:
                print(f"{partition.capitalize()} {len(predictions)}/{len(requests)}", flush=True)
    finally:
        model.train(was_training)
    write_jsonl(str(output) + ".predictions.jsonl", predictions)
    report = {"metrics": metrics(predictions, 95),
              "cross_entropy": -math.fsum(p["target_log_probability"] for p in predictions) / len(predictions),
              "query_seconds": timings, "seconds": sum(timings), "test_scored": partition == "test"}
    write_json(str(output) + ".json", report)
    return report


def _train(root, mode, learning_rate, output):
    prepared, inputs = load_inputs(root, mode)
    spec, models = prepared["config"], prepared["models"]
    schedule, prepared_hash = plan(spec, mode), digest(prepared)
    check_trial_policy(root, mode, learning_rate, prepared)
    launch = read_json(output / "launch.json")
    if any(launch[key] != value for key, value in
           (("mode", mode), ("learning_rate", learning_rate), ("prepared_hash", prepared_hash))):
        raise ValueError("Worker launch differs from the immutable parent trial")
    torch = apply_precision()
    environment = verify_environment(models)
    from kev.checkpoint import Checkpoint
    from kev.data import materialize
    from safetensors.torch import load_file
    import kev.train as trainer
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    checkpoint = Checkpoint(released(models))
    if checkpoint.meta.base != models["base_model"] or checkpoint.meta.base_revision != models["base_revision"]:
        raise ValueError("Released checkpoint backbone/revision differs from the pinned specification")
    source_hashes = {"weights_sha256": checkpoint.weights_sha256(), "head_sha256": file_hash(checkpoint.file("head.pt"))}
    config = {**models, "seed": spec["seed"], "max_state": spec["max_state"], "max_request": spec["max_request"]}
    destination = output / "checkpoints"
    if destination.exists():
        raise ValueError("Use a fresh immutable checkpoint directory")
    native = training_command(Path(root) / f"inputs/{mode}", destination, checkpoint, config,
                              max_steps=0, learning_rate=learning_rate, batch=1, accum=8,
                              checkpointing=True, preserve_candidate_order=True)[4:]
    native[native.index("--epochs") + 1] = str(schedule["epochs"])
    history, gradient_checks = [], []
    policy = Top3Stopping(**{k: spec["full"]["early_stopping"][k] for k in ("minimum_epochs", "patience")})
    startup = {}

    def cpu_state(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_state(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(cpu_state(item) for item in value)
        return value

    original_loader, original_encoder = trainer.training_requests, trainer.encode_batch

    def checked_loader(*args, **kwargs):
        rows = original_loader(*args, **kwargs)
        if len(rows) != len(inputs["train"][0]):
            raise ValueError("Native Kev silently dropped training requests")
        for row, meta in zip(rows, inputs["train"][1]):
            record = materialize(row)
            question = record["questions"][0]
            if (len(record["questions"]) != 1 or question["keys"] != prepared["choice_keys"]
                    or prepared["candidate_ids"][question["label"]] != meta["target"]):
                raise ValueError("Native loader changed the labels or candidate set")
        startup["model_and_input_setup_seconds"] = time.monotonic() - start
        return rows

    def checked_encoder(*args, **kwargs):
        variants = original_encoder(*args, **kwargs)
        for variant in variants:
            checked_encoding(variant.enc, prepared["candidate_ids"], spec["max_request"])
            if variant.rec["questions"][0]["keys"] != prepared["choice_keys"] or variant.permuted or variant.share != 1:
                raise ValueError("Native trainer altered the fixed candidate order")
        return variants

    def gradient_hook(values):
        if mode != "pilot" and values["step"]:
            return
        params = [(name, p) for name, p in values["model"].named_parameters() if p.requires_grad]
        if len(params) != 500 or any("lora_" not in name and not name.startswith("head.") for name, _ in params):
            raise ValueError("Unexpected trainable parameters")
        if any(p.dtype != torch.float32 or p.grad is None for _, p in params):
            raise ValueError("Expected FP32 trainable weights and populated gradients")
        norms = {}
        for group in ("adapter", "head"):
            selected = [p.grad for name, p in params if ("lora_" in name if group == "adapter" else name.startswith("head."))]
            finite = bool(torch.stack([torch.isfinite(g).all() for g in selected]).all())
            norm = float(torch.stack([g.float().norm() for g in selected]).norm())
            if not finite or not math.isfinite(norm) or norm <= 0:
                raise ValueError("Missing, zero or nonfinite gradient group")
            norms[group] = norm
        gradient_checks.append({"optimizer_step": values["step"] + 1, "norms_before_native_clipping": norms})

    def step_hook(values):
        step, seen = values["step"], values["seen"]
        if step % schedule["validation_every_steps"]:
            return None
        if seen != step * 8 or values["sched"].total_steps != schedule["optimizer_steps"]:
            raise ValueError("Native scheduler or record accounting changed")
        model, tokenizer = values["model"], values["tok"]
        directory = destination / f"step-{step:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        model.lm.save_pretrained(directory)
        metadata = replace(values["meta"], head=cpu_state(model.head.state_dict()),
                           extra={"args": vars(values["a"]), "init_source": values["init_source"],
                                  "epoch": seen / schedule["training_papers"], "optimizer_steps": step,
                                  "experiment": spec["experiment_id"]})
        trainer.finish_checkpoint(str(directory), metadata, tokenizer)
        validation = score_model(model, tokenizer, inputs["validation"], prepared["candidate_ids"],
                                 spec, directory / "validation", torch)
        epoch = seen / schedule["training_papers"]
        stopping = policy.update(epoch, validation["metrics"]["top3"])
        stop = mode == "full" and stopping["stop"]
        entry = {"epoch": epoch, "optimizer_steps": step, "checkpoint": str(directory),
                 "metrics": validation["metrics"], "cross_entropy": validation["cross_entropy"],
                 "stopping": stopping, "test_scored": False}
        history.append(entry)
        state = {"optimizer": cpu_state(values["opt"].state_dict()), "scheduler": values["sched"].state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                 "python_rng": values["rng"].getstate(), "epoch_index": values["ep"],
                 "next_microbatch": values["mb"] + 1, "request_order": [request_digest(r) for r in values["reqs"]],
                 "optimizer_steps": step, "records_seen": seen, "stopping": dict(policy.__dict__),
                 "specification": spec, "learning_rate": learning_rate, "prepared_hash": prepared_hash,
                 "history": history}
        if len(state["optimizer"]["state"]) != 500:
            raise ValueError("Optimizer state is incomplete")
        temporary = directory / "training-state.pt.tmp"
        torch.save(state, temporary)
        temporary.rename(directory / "training-state.pt")
        write_json(directory / "snapshot.json", {"checks_pass": True, "epoch": epoch, "optimizer_steps": step,
                   "scheduler_steps_total": schedule["optimizer_steps"], "prepared_hash": prepared_hash,
                   "training_state_saved": True, "exact_resume_implemented": False})
        write_json(output / "progress.json", {"mode": mode, "history": history, "selected": select_checkpoint(history),
                   "prepared_hash": prepared_hash, "stopped_early": stop, "test_scored": False})
        print(f"Update {step}/{schedule['optimizer_steps']}, epoch {epoch:g}: Top 3 {validation['metrics']['top3']:.4f}", flush=True)
        return stop

    source = inspect.getsource(trainer.main)
    patched = patched_main(source)
    trainer.record_variants, trainer.training_requests, trainer.encode_batch = ordered_variants, checked_loader, checked_encoder
    trainer.__dict__.update(journal_apply_precision=apply_precision, journal_gradient_hook=gradient_hook, journal_step_hook=step_hook)
    exec(compile(patched, "<pinned-kev-only-hooks>", "exec"), trainer.__dict__)
    sys.argv = ["kev.train"] + native
    args = trainer.parse_args()
    if args.full_ft or args.weights_dtype != "fp32" or args.dtype != "bf16" or not args.checkpointing or args.lora_targets != "all":
        raise ValueError("Native training settings differ from the pinned recipe")
    if args.init_from != released(models) or args.batch != 1 or args.accum != 8 or args.lr != learning_rate or args.epochs != schedule["epochs"]:
        raise ValueError("Warm start or native schedule differs")
    write_json(output / "native-command.json", {"argv": native, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(), "optimizer_loss_scheduler": "unchanged native AdamW/cross-entropy/OneCycle"})
    trainer.main()
    stats = read_json(destination / "training_metrics.json")
    if stats["optimizer_steps"] != history[-1]["optimizer_steps"] or stats["records_seen"] != stats["optimizer_steps"] * 8:
        raise ValueError("Training accounting differs from saved checkpoints")
    if stats["requested_records"] != schedule["training_papers"] * schedule["epochs"] or stats["rejected_records"] or stats["truncated_records"]:
        raise ValueError("Training dropped or truncated records")
    if not gradient_checks or (mode == "pilot" and stats["optimizer_steps"] != 3):
        raise ValueError("Gradient checks or pilot update count failed")
    init_source = read_json(destination / "training_config.json")["init_source"]
    if any(init_source[key] != value for key, value in source_hashes.items()):
        raise ValueError("Warm start did not use the released adapter and head")
    trained = Checkpoint(str(destination))
    changes = {}
    for kind, old, new in (("adapter", load_file(str(checkpoint.file("adapter_model.safetensors"))), load_file(str(trained.file("adapter_model.safetensors")))),
                           ("head", checkpoint.meta.head, trained.meta.head)):
        if old.keys() != new.keys() or not all(bool(torch.isfinite(v).all()) for v in new.values()):
            raise ValueError("Checkpoint tensor shapes or finiteness changed")
        changed = sum(not torch.equal(old[k], new[k]) for k in old)
        if not changed:
            raise ValueError(f"No {kind} weights were updated")
        changes[kind] = {"tensors": len(new), "changed": changed}
    if checkpoint.weights_sha256() != source_hashes["weights_sha256"] or file_hash(checkpoint.file("head.pt")) != source_hashes["head_sha256"]:
        raise ValueError("Released source checkpoint was modified")
    report = {"status": "trained", "checks_pass": True, "mode": mode, "learning_rate": learning_rate,
              "prepared_hash": prepared_hash, "history": history, "selected": select_checkpoint(history),
              "training": stats, "gradient_checks": gradient_checks, "tensor_changes": changes,
              "warm_start": source_hashes, "environment": environment, **startup,
              "seconds": time.monotonic() - start,
              "first_update_seconds": stats["step_seconds"][0],
              "warm_seconds_per_training_record": statistics.mean(stats["step_seconds"][1:]) / 8 if len(stats["step_seconds"]) > 1 else None,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "optimizer_scheduler_rng_saved": True, "exact_resume_implemented": False, "test_scored": False}
    write_json(output / "training-result.json", report)


def _reload(root, mode, output):
    prepared, inputs = load_inputs(root, mode)
    training = read_json(output / "training-result.json")
    if not training["checks_pass"] or training["prepared_hash"] != digest(prepared):
        raise ValueError("Training report changed")
    torch = apply_precision()
    verify_environment(prepared["models"])
    from kev.checkpoint import Checkpoint, LoadOptions
    checkpoint = Path(training["selected"]["checkpoint"])
    tokenizer, model = Checkpoint(str(checkpoint)).load("cuda", LoadOptions(dtype=torch.float32, merge=False, fused=False, cuda_graphs=False))
    sample = tuple(part[:8] for part in inputs["validation"])
    score_model(model, tokenizer, sample, prepared["candidate_ids"], prepared["config"], output / "reload", torch)
    before = read_jsonl(checkpoint / "validation.predictions.jsonl")[:8]
    after = read_jsonl(output / "reload.predictions.jsonl")
    if len(before) != len(after) or [r["paper_id"] for r in before] != [r["paper_id"] for r in after]:
        raise ValueError("Reload comparison identities differ")
    difference = max(abs(a["probabilities"][j] - b["probabilities"][j]) for a, b in zip(before, after) for j in prepared["candidate_ids"])
    if difference > 1e-6 or any(a["ranking"] != b["ranking"] for a, b in zip(before, after)):
        raise ValueError(f"Saved/reloaded checkpoint predictions differ: {difference}")
    state = torch.load(checkpoint / "training-state.pt", map_location="cpu", weights_only=False)
    if state["prepared_hash"] != digest(prepared) or state["optimizer_steps"] != training["selected"]["optimizer_steps"] or len(state["optimizer"]["state"]) != 500:
        raise ValueError("Saved optimizer state does not match the selected checkpoint")
    write_json(output / "reload-check.json", {"checks_pass": True, "max_probability_delta": difference,
               "same_rankings": True, "papers": len(after), "optimizer_state_readable": True,
               "exact_resume_implemented": False, "test_scored": False})


def require_completed(path, prepared_hash, mode):
    report = read_json(path)
    if report.get("status") != "complete" or not report.get("checks_pass") or report.get("prepared_hash") != prepared_hash or report.get("mode") != mode:
        raise ValueError(f"A completed matching {mode} is required")
    return report


def check_trial_policy(root, mode, learning_rate, prepared):
    """Apply the same prerequisites even to a directly invoked internal worker."""
    root = Path(root)
    spec, prepared_hash = prepared["config"], digest(prepared)
    if mode == "pilot":
        if learning_rate != spec["pilot"]["learning_rate"]:
            raise ValueError("Use the configured pilot learning rate")
        return
    require_completed(root / "pilot-result.json", prepared_hash, "pilot")
    if mode == "screen":
        if learning_rate not in spec["screen"]["learning_rates"]:
            raise ValueError("Learning rate is outside the configured screen")
        return
    if mode != "full":
        raise ValueError("Unknown trial mode")
    screen = require_completed(root / "screen-result.json", prepared_hash, "screen")
    trials = [require_completed(root / "trials/screen" / f"lr-{rate:g}" / "result.json", prepared_hash, "screen")
              for rate in spec["screen"]["learning_rates"]]
    winner = select_rate(trials, spec["screen"]["learning_rates"])
    if learning_rate != winner["learning_rate"] or screen["selected_learning_rate"] != learning_rate or screen["selected"] != winner["selected"]:
        raise ValueError("Full training must use the completed screen's selection")


def run(root, mode, learning_rate=None):
    if execution_backend() == "modal-b300":
        from journal_suggester.cloud_runtime import require_modal_b300
        require_modal_b300()
    elif execution_backend() == "cuda":
        from journal_suggester.cloud_runtime import require_training_gpu
        require_training_gpu(40)
    elif platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Kev-only training is GB10-only; use SSH. No laptop/CPU fallback.")
    root = Path(root).resolve()
    prepared, _ = load_inputs(root, mode)
    spec, prepared_hash = prepared["config"], digest(prepared)
    if mode == "pilot":
        if learning_rate is not None and learning_rate != spec["pilot"]["learning_rate"]:
            raise ValueError("Use the configured pilot learning rate")
        learning_rate = spec["pilot"]["learning_rate"]
        output = root / "trials/pilot"
    elif mode == "screen":
        require_completed(root / "pilot-result.json", prepared_hash, "pilot")
        if learning_rate is None:
            for rate in spec["screen"]["learning_rates"]:
                finished = root / "trials/screen" / f"lr-{rate:g}" / "result.json"
                if finished.exists():
                    require_completed(finished, prepared_hash, "screen")
                    continue
                subprocess.run([sys.executable, "-m", __spec__.name, "run", "--root", str(root), "--mode", "screen", "--learning-rate", str(rate)], check=True)
            return read_json(root / "screen-result.json")
        if learning_rate not in spec["screen"]["learning_rates"]:
            raise ValueError("Learning rate is outside the configured screen")
        output = root / "trials/screen" / f"lr-{learning_rate:g}"
    else:
        require_completed(root / "pilot-result.json", prepared_hash, "pilot")
        screen = require_completed(root / "screen-result.json", prepared_hash, "screen")
        trials = [require_completed(root / "trials/screen" / f"lr-{rate:g}" / "result.json", prepared_hash, "screen") for rate in spec["screen"]["learning_rates"]]
        winner = select_rate(trials, spec["screen"]["learning_rates"])
        if screen["selected_learning_rate"] != winner["learning_rate"] or (learning_rate is not None and learning_rate != winner["learning_rate"]):
            raise ValueError("Full training must use the completed screen's selected learning rate")
        learning_rate, output = winner["learning_rate"], root / "trials/full"
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch.json", {"mode": mode, "learning_rate": learning_rate, "prepared_hash": prepared_hash,
               "website_model_unchanged": True, "test_scored": False})
    started = time.monotonic()
    try:
        for stage in ("_train", "_reload"):
            command = [sys.executable, "-u", "-m", __spec__.name, stage, "--root", str(root), "--mode", mode, "--output", str(output)]
            if stage == "_train":
                command.extend(["--learning-rate", str(learning_rate)])
            remaining = spec["pilot"]["max_runtime_seconds"] - (time.monotonic() - started) if mode == "pilot" else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Pilot time bound reached")
            with (output / (stage.lstrip("_") + ".log")).open("x") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining)
        result = read_json(output / "training-result.json")
        reload = read_json(output / "reload-check.json")
        if not reload["checks_pass"]:
            raise ValueError("Saved-checkpoint reload did not pass")
        result.update(status="complete", reload=reload, total_seconds=time.monotonic() - started)
        write_json(output / "result.json", result)
        if mode in ("pilot", "full"):
            write_json(root / f"{mode}-result.json", result)
        else:
            paths = [root / "trials/screen" / f"lr-{rate:g}" / "result.json" for rate in spec["screen"]["learning_rates"]]
            if all(path.is_file() for path in paths):
                trials = [require_completed(path, prepared_hash, "screen") for path in paths]
                winner = select_rate(trials, spec["screen"]["learning_rates"])
                write_json(root / "screen-result.json", {"status": "complete", "checks_pass": True, "mode": "screen",
                           "prepared_hash": prepared_hash, "selected_learning_rate": winner["learning_rate"],
                           "selected": winner["selected"], "trials": [str(p) for p in paths], "test_scored": False})
        return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__,
                   "seconds": time.monotonic() - started, "test_scored": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "_train", "_reload"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "run":
        result = run(args.root, args.mode, args.learning_rate)
        print({"status": result["status"], "mode": args.mode, "checks_pass": result["checks_pass"]})
    elif args.stage == "_train":
        if args.output is None or args.learning_rate is None:
            parser.error("_train requires --output and --learning-rate")
        _train(args.root, args.mode, args.learning_rate, args.output)
    else:
        if args.output is None:
            parser.error("_reload requires --output")
        _reload(args.root, args.mode, args.output)


if __name__ == "__main__":
    main()
