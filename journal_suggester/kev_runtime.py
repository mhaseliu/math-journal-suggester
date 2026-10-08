"""Pinned native Kev integration, warm-start validation, and natural-candidate scoring."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .evaluation import metrics
from .gpu import require_gb10
from .io import read_json, read_jsonl, request_digest, write_json, write_jsonl


def released(config):
    return config["kev_model"] + "@" + config["kev_revision"]


class Ranker:
    def __init__(self, run, config=None):
        self.config = config or read_json("configs/models.json")
        self.torch = require_gb10(28)
        self.torch.cuda.reset_peak_memory_stats()
        from kev.checkpoint import Checkpoint, LoadOptions
        self.checkpoint = Checkpoint(run)
        self.tokenizer, self.model = self.checkpoint.load("cuda", LoadOptions(dtype=self.torch.bfloat16, merge=False, fused=False, cuda_graphs=False))

    def predict(self, request, *, use_prefix=True):
        from kev.api import SystemOneRequest, to_record
        from kev.model import training_context
        config = self.config
        clean = {"state": request["state"], "questions": {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in request["questions"].items()}}
        rec, meta = to_record(SystemOneRequest.model_validate(clean))
        enc = self.model.encode(self.tokenizer, rec, max_state=config["max_state"], max_branch=training_context(config["max_state"])["max_branch"], strict=True)
        if len(enc["ids"]) > config["max_request"]:
            raise ValueError("Request exceeds project context budget")
        with self.torch.inference_mode():
            probs = (self.model.probs(enc)[0] if use_prefix else self.model(enc)[0].float().softmax(-1).cpu()).tolist()
        return dict(zip(meta[0]["keys"], probs))


def evaluate(run, directory, output, partition="validation", *, config=None, candidate_k=10):
    directory = Path(directory)
    requests = read_jsonl(directory / f"{partition}.jsonl")
    metadata = read_jsonl(directory / f"{partition}.meta.jsonl")
    if len(requests) != len(metadata) or not requests:
        raise ValueError("Request/metadata count mismatch or empty evaluation")
    ranker = Ranker(run, config)
    predictions, times = [], []
    for request, meta in zip(requests, metadata):
        if request_digest(request) != meta["request_hash"] or meta["inserted"] or "label" in request["questions"]["journal"]:
            raise ValueError("Evaluation requires unchanged, unlabelled natural requests")
        start = time.monotonic()
        probs = ranker.predict(request)
        ranker.torch.cuda.synchronize()
        times.append(time.monotonic() - start)
        ranking = sorted(probs, key=lambda j: (-probs[j], j))
        predictions.append({"paper_id": meta["paper_id"], "target": meta["target"], "candidates": meta["natural_journals"],
                            "ranking": ranking, "probabilities": probs, "request_hash": meta["request_hash"]})
        if len(predictions) % 25 == 0:
            write_jsonl(str(output) + ".partial.predictions.jsonl", predictions)
            print(f"Scored {partition}: {len(predictions)}/{len(requests)}; {sum(times):.1f}s inference", flush=True)
    write_jsonl(str(output) + ".predictions.jsonl", predictions)
    report = {"run": run, "partition": partition, "metrics": metrics(predictions, candidate_k=candidate_k), "seconds": sum(times),
              "warm_seconds_per_query": sum(times[1:]) / max(1, len(times) - 1), "load_options": "bf16 unmerged torch; fused/graphs disabled"}
    report["query_seconds"] = times
    report["peak_cuda_allocated_gib"] = ranker.torch.cuda.max_memory_allocated() / 2 ** 30
    report["peak_cuda_reserved_gib"] = ranker.torch.cuda.max_memory_reserved() / 2 ** 30
    write_json(str(output) + ".json", report)
    Path(str(output) + ".partial.predictions.jsonl").unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


def validate_training(directory, config, tokenizer):
    from kev.data import load_records, materialize
    from kev.model import encode, fits, training_context
    directory = Path(directory)
    report = read_json(directory / "report.json")
    if report["backend"] != "qwen3-embedding-8b":
        raise ValueError("Training requires the Qwen pipeline, not smoke examples")
    rows = load_records(directory / "train.jsonl")
    meta = read_jsonl(directory / "train.meta.jsonl")
    if len(rows) != len(meta):
        raise ValueError("Training metadata mismatch")
    held = {r["group_id"] for partition in ("validation", "test") for r in read_jsonl(directory / f"{partition}.meta.jsonl")}
    for original, m in zip(read_jsonl(directory / "train.jsonl"), meta):
        if request_digest(original) != m["request_hash"] or m["group_id"] in held:
            raise ValueError("Training request altered or held-out group present")
    kept = [r for r in rows if fits(materialize(r), tokenizer, **training_context(config["max_state"]))]
    result = {"input": len(rows), "retained": len(kept), "dropped": len(rows) - len(kept)}
    write_json(directory / "loader_retention.json", result)
    if len(kept) != len(rows):
        raise ValueError(f"Upstream loader would drop examples: {result}")
    for row in rows:
        encoded = encode(tokenizer, materialize(row), max_state=config["max_state"],
                         max_branch=training_context(config["max_state"])["max_branch"], strict=True)
        if len(encoded["ids"]) > config["max_request"]:
            raise ValueError("Training request exceeds project context budget")
    return rows


def training_command(directory, output, ck, config, max_steps=3, learning_rate=2e-5,
                     batch=1, accum=8, checkpointing=True, preserve_candidate_order=False):
    entry = "journal_suggester.ordered_train" if preserve_candidate_order else "kev.train"
    return [sys.executable, "-m", "journal_suggester.gb10_runtime", entry, "--data", str(Path(directory) / "train.jsonl"),
               "--base", ck.meta.base, "--base_revision", ck.meta.base_revision, "--init_from", released(config),
               "--lora", str(ck.meta.lora), "--head_dim", str(ck.meta.head_dim), "--lora_targets", "all",
               "--option_isolation", str(int(ck.meta.option_isolation)), "--special_embeddings", str(int(ck.meta.special_embeddings)),
               "--epochs", "1", "--lr", str(learning_rate), "--batch", str(batch), "--accum", str(accum), "--dtype", "bf16",
               "--weights_dtype", "fp32", "--checkpointing", str(int(checkpointing)), "--seed", str(config["seed"]), "--device", "cuda",
               "--max_state", str(config["max_state"]), "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0",
               "--p_none_pair", "0", "--perm_kl", "0", "--full_ft", "0", "--max_steps", str(max_steps), "--out", str(output)]


def train(directory, output, max_steps=3, learning_rate=2e-5, batch=1, accum=8, checkpointing=True,
          *, config=None, preserve_candidate_order=False):
    require_gb10(36)
    from kev.checkpoint import Checkpoint
    from kev.model import load_tokenizer
    config = config or read_json("configs/models.json")
    ck = Checkpoint(released(config))
    tok = load_tokenizer(ck.meta.base, ck.meta.base_revision)
    validate_training(directory, config, tok)
    if Path(output).exists():
        raise ValueError("Use a new checkpoint directory; existing runs are immutable")
    command = training_command(directory, output, ck, config, max_steps, learning_rate, batch, accum,
                               checkpointing, preserve_candidate_order)
    write_json(Path(directory) / "training_command.json", command)
    subprocess.run(command, check=True)


def gradient_check(directory, output):
    torch = require_gb10(36)
    torch.cuda.reset_peak_memory_stats()
    from kev.checkpoint import Checkpoint, Meta
    from kev.data import materialize
    from kev.model import DecisionModel, load_tokenizer, training_context
    config = read_json("configs/models.json")
    torch.manual_seed(config["seed"])
    ck = Checkpoint(released(config))
    tok = load_tokenizer(ck.meta.base, ck.meta.base_revision)
    rows = validate_training(directory, config, tok)
    model = DecisionModel(ck.meta.base, tok, "cuda", lora=ck.meta.lora, revision=ck.meta.base_revision,
                          head_dim=ck.meta.head_dim, option_isolation=ck.meta.option_isolation,
                          special_embeddings=ck.meta.special_embeddings, lora_targets="all", dtype=torch.float32)
    ours = Meta(base=ck.meta.base, base_revision=ck.meta.base_revision, lora=ck.meta.lora, head_dim=ck.meta.head_dim,
                option_isolation=ck.meta.option_isolation, special_embeddings=ck.meta.special_embeddings)
    init = ck.warm_start(model, ours)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n and not n.startswith("head.") for n in trainable):
        raise ValueError("Unexpected trainable backbone parameters")
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    model.train()
    rec = materialize(rows[0])
    enc = model.encode(tok, rec, max_state=config["max_state"], max_branch=training_context(config["max_state"])["max_branch"], strict=True)
    start = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(enc)[0]
        loss = torch.nn.functional.cross_entropy(logits.float().unsqueeze(0), torch.tensor([rec["questions"][0]["label"]], device="cuda"))
    loss.backward()
    torch.cuda.synchronize()
    norms = {kind: sum(float(p.grad.float().norm()) for n, p in model.named_parameters() if p.grad is not None and ("lora_" in n if kind == "adapter" else n.startswith("head."))) for kind in ("adapter", "head")}
    if not torch.isfinite(loss) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()) or not all(v > 0 for v in norms.values()):
        raise ValueError("Missing/non-finite gradients")
    result = {"loss": float(loss.detach()), "gradient_norm_sums": norms, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "seconds": time.monotonic() - start, "backbone_frozen": True, "warm_start": init}
    result["request_tokens"] = len(enc["ids"])
    result["optimizer_steps"] = 0
    result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    result["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2 ** 30
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["evaluate", "train", "gradient-check"])
    p.add_argument("--directory", default="artifacts/pilot/qwen")
    p.add_argument("--output", required=True)
    p.add_argument("--run")
    p.add_argument("--partition", choices=["validation", "test"], default="validation")
    p.add_argument("--max-steps", type=int, default=3, help="0 = full epoch; default is only a trial")
    p.add_argument("--learning-rate", type=float, default=2e-5, help="Native OneCycle maximum learning rate")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--checkpointing", type=int, choices=[0, 1], default=1)
    a = p.parse_args()
    if a.command == "evaluate":
        evaluate(a.run or released(read_json("configs/models.json")), a.directory, a.output, a.partition)
    elif a.command == "gradient-check":
        gradient_check(a.directory, a.output)
    else:
        train(a.directory, a.output, a.max_steps, a.learning_rate, a.batch, a.accum, bool(a.checkpointing))


if __name__ == "__main__":
    main()
