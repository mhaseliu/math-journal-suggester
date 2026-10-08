"""GB10-only model execution. The laptop cannot silently fall back to CPU."""
import os
import platform
import time
from pathlib import Path

from .io import digest, read_json, read_jsonl, write_json


def require_gb10(min_available_gib=0):
    if os.environ.get("JOURNAL_EXECUTION_BACKEND") == "cuda":
        from .cloud_runtime import require_cuda
        return require_cuda()
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Model execution is GB10-only; use SSH. No laptop/CPU fallback.")
    import torch
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name():
        raise RuntimeError("An NVIDIA GB10 CUDA device is required")
    from .gb10_runtime import configure
    configure()
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if memory["MemAvailable"] / 2 ** 20 < min_available_gib:
        raise RuntimeError(f"Need at least {min_available_gib} GiB available before model loading")
    return torch


class QwenEmbedder:
    def __init__(self, config, cache="artifacts/vectors"):
        self.config, self.cache = config, Path(cache)
        self.model = self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        from .cloud_runtime import require_training_gpu
        torch = require_training_gpu(24)
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["embedding_model"], revision=self.config["embedding_revision"], padding_side="left")
        self.model = AutoModel.from_pretrained(self.config["embedding_model"], revision=self.config["embedding_revision"],
                                             dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, papers, query=False):
        import numpy as np
        vectors = []
        for paper in papers:
            text = paper["title"] + "\n" + paper["abstract"]
            if query:
                text = "Instruct: " + self.config["query_instruction"] + "\nQuery:" + text
            key = digest({"model": self.config["embedding_model"], "revision": self.config["embedding_revision"],
                          "text": text, "max_length": self.config["embedding_max_length"], "pooling": "last-token-l2-bf16"})
            path = self.cache / (key + ".npy")
            if path.exists():
                vector = np.load(path, allow_pickle=False)
            else:
                self.load()
                import torch
                self.cache.mkdir(parents=True, exist_ok=True)
                batch = self.tokenizer([text], padding=True, truncation=True, max_length=self.config["embedding_max_length"], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    last = self.model(**batch).last_hidden_state[:, -1, :].float()
                    vector = torch.nn.functional.normalize(last, p=2, dim=1)[0].cpu().numpy()
                np.save(path, vector)
            if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=.001):
                raise ValueError("Invalid cached embedding")
            vectors.append(vector)
            if len(vectors) % 100 == 0:
                print(f"Embedding progress: {len(vectors)}/{len(papers)}", flush=True)
        return np.stack(vectors)


def embed_split(split_dir, output, config, partitions=("reference", "train", "validation", "test")):
    require_gb10(24)
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = QwenEmbedder(config)
    timings = {}
    import torch
    torch.cuda.reset_peak_memory_stats()
    for name in partitions:
        papers = read_jsonl(Path(split_dir) / f"{name}.jsonl")
        if not papers:
            continue
        start = time.monotonic()
        vectors = embedder.encode(papers, query=name != "reference")
        np.save(output / f"{name}.npy", vectors)
        write_json(output / f"{name}.json", {"paper_ids": [p["paper_id"] for p in papers], "input_hash": digest(papers),
                                            "model": config["embedding_model"], "revision": config["embedding_revision"],
                                            "config_hash": digest(config), "role": "document" if name == "reference" else "query"})
        timings[name] = {"seconds": time.monotonic() - start, "n": len(papers), "dimensions": vectors.shape[1]}
        print(f"Embedded {name}: {timings[name]}", flush=True)
    write_json(output / "timings.json", timings)
    write_json(output / "memory.json", {"peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                                        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30})
    # This command exits here, releasing the embedding model before Kev is loaded.


def load_vectors(directory, name, papers, config):
    import numpy as np
    directory = Path(directory)
    meta = read_json(directory / f"{name}.json")
    if meta["input_hash"] != digest(papers) or meta["config_hash"] != digest(config):
        raise ValueError("Stale or misaligned cached vectors")
    values = np.load(directory / f"{name}.npy", allow_pickle=False, mmap_mode="r")
    if len(values) != len(papers):
        raise ValueError("Vector count mismatch")
    return values
