"""Pinned website checkpoint, using the final evaluation's full-forward path."""
import hashlib
import inspect
import math
import time
from pathlib import Path

from .gpu import require_gb10
from .io import read_json, request_digest
from .ordered_train import verify_kev_revision

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = {"max_state": 5504, "max_request": 6144, "query_tokens": 768, "reference_tokens": 100}
CHECKPOINT_FILES = {"adapter_model.safetensors", "adapter_config.json", "head.pt", "tokenizer.json",
                    "tokenizer_config.json", "chat_template.jinja", "snapshot.json"}


def verify_checkpoint(path, specification, models):
    path = Path(path)
    if path.is_symlink() or (path / "manifest.json").is_symlink():
        raise ValueError("Unexpected checkpoint symlink")
    raw = (path / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != specification["manifest_sha256"]:
        raise ValueError("Website checkpoint manifest differs from the audited selection")
    manifest = read_json(path / "manifest.json")
    if (manifest["selected_epoch"] != specification["epoch"]
            or manifest["optimizer_steps"] != specification["optimizer_steps"]
            or manifest["kev_code_revision"] != models["kev_code_revision"]
            or manifest["base_model"] != models["base_model"]
            or manifest["base_revision"] != models["base_revision"]
            or set(manifest["files"]) != CHECKPOINT_FILES):
        raise ValueError("Website checkpoint identity mismatch")
    for name, expected in manifest["files"].items():
        file = path / name
        if file.is_symlink() or file.stat().st_size != expected["bytes"]:
            raise ValueError("Unexpected checkpoint file: " + name)
        with file.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"]:
            raise ValueError("Checkpoint checksum mismatch: " + name)
    return manifest


class WebsiteRanker:
    def __init__(self, run, models=None):
        models = models or read_json("configs/models.json")
        specification = read_json("configs/website-kev.json")
        self.manifest = verify_checkpoint(run, specification, models)
        verify_kev_revision(models["kev_code_revision"])
        self.config = {**models, **CONTEXT}
        self.torch = torch = require_gb10(28)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        from kev.checkpoint import Checkpoint, LoadOptions
        from transformers.models.qwen3_5 import modeling_qwen3_5
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn
        implementations = [(modeling_qwen3_5.torch_chunk_gated_delta_rule, chunk_gated_delta_rule),
                           (modeling_qwen3_5.causal_conv1d_fn, causal_conv1d_fn)]
        if any(inspect.getclosurevars(wrapper).nonlocals.get("implementation") is not expected
               for wrapper, expected in implementations):
            raise RuntimeError("The verified GB10 fast kernels are required for website Kev inference")
        self.checkpoint = Checkpoint(run)
        meta = self.checkpoint.meta
        if (meta.base != models["base_model"] or meta.base_revision != models["base_revision"]
                or meta.weights_dtype != "fp32" or meta.weights != "lora"
                or meta.extra.get("epoch") != specification["epoch"]
                or meta.extra.get("optimizer_steps") != specification["optimizer_steps"]):
            raise ValueError("Checkpoint metadata differs from the selected training snapshot")
        self.tokenizer, self.model = self.checkpoint.load("cuda", LoadOptions(
            dtype=torch.float32, merge=False, fused=False, cuda_graphs=False))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def verify_reference(self, fixtures):
        if len(fixtures) != 8:
            raise ValueError("Expected eight fixed validation parity fixtures")
        comparison = []
        for fixture in fixtures:
            request, expected = fixture['request'], fixture['prediction']
            if request_digest(request) != expected['request_hash']:
                raise ValueError("Parity fixture changed")
            start = time.monotonic()
            probabilities = self.predict(request)
            ranking = sorted(probabilities, key=lambda key: (-probabilities[key], key))
            comparison.append({'max_probability_delta': max(abs(probabilities[k] - expected['probabilities'][k]) for k in probabilities),
                               'top1_matches': ranking[0] == expected['ranking'][0],
                               'top3_set_matches': set(ranking[:3]) == set(expected['ranking'][:3]),
                               'seconds': time.monotonic() - start})
            if list(probabilities) != expected['candidates']:
                raise ValueError("Parity fixture candidate order changed")
        return comparison

    def predict(self, request):
        from kev.api import SystemOneRequest, to_record
        from kev.model import training_context
        if any("label" in question for question in request["questions"].values()):
            raise ValueError("Website inference must not receive labels")
        record, metadata = to_record(SystemOneRequest.model_validate(request))
        encoded = self.model.encode(self.tokenizer, record, max_state=self.config["max_state"],
                                    max_branch=training_context(self.config["max_state"])["max_branch"], strict=True)
        if len(encoded["ids"]) > self.config["max_request"] or len(metadata) != 1:
            raise ValueError("Website request exceeds the evaluated inference budget")
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            probabilities = self.model.forward_batch([encoded])[0][0].float().softmax(-1).cpu().tolist()
        keys = metadata[0]["keys"]
        if (len(keys) != len(probabilities) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1, abs_tol=1e-5)):
            raise ValueError("Invalid Kev probabilities")
        return dict(zip(keys, probabilities))
