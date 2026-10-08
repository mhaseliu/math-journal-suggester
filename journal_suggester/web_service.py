"""Citation-backed search shared by the web preview and optional GB10 worker."""
import atexit
import os
import json
import select
import shlex
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .io import digest, journals, read_json, read_jsonl
from .records import doi_key
from .retrieval import LexicalIndex, shortlist

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = Path(os.environ.get("JOURNAL_SPLIT_DIR", "artifacts/experiment/splits"))


def validate_query(body):
    if not isinstance(body, dict) or any(not isinstance(body.get(k), str) for k in ("title", "abstract")):
        raise ValueError("Provide a title and abstract.")
    # User text is plain text; preserve inequalities and mathematical notation.
    query = {k: " ".join(body[k].split()) for k in ("title", "abstract")}
    if not 3 <= len(query["title"]) <= 2000 or not 40 <= len(query["abstract"]) <= 30000:
        raise ValueError("Enter a title (3–2,000 characters) and an abstract (40–30,000 characters).")
    if query["title"].casefold() == "introduction":
        raise ValueError("Please use the manuscript title, rather than a section heading.")
    for k in ("doi", "arxiv_id"):
        if isinstance(body.get(k), str) and len(body[k]) <= 300:
            query[k] = body[k]
    return query


def citation(paper):
    doi = doi_key(paper.get("doi", ""))
    url = "https://doi.org/" + doi if doi else paper.get("url", "")
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        url = ""
    return {k: paper.get(k, "") for k in ("paper_id", "title", "abstract", "year")} | {"url": url, "doi": doi}


class SearchService:
    def __init__(self, split_dir=DEFAULT_SPLIT, vectors=None, kev_run=None):
        self.references = read_jsonl(Path(split_dir) / "reference.jsonl")
        if not self.references:
            raise ValueError("Empty reference corpus")
        self.reference_hash = digest(self.references)
        self.names = {j["journal_id"]: j["journal_name"] for j in journals()}
        self.lock = threading.Lock()
        self.ranker = self.embedder = None
        if vectors:
            from .gpu import QwenEmbedder, load_vectors, require_gb10
            require_gb10(48 if kev_run else 24)
            self.config = read_json("configs/models.json")
            self.matrix = load_vectors(vectors, "reference", self.references, self.config)
            import numpy as np
            metadata = read_json(Path(vectors) / "reference.json")
            if (metadata.get("paper_ids") != [p["paper_id"] for p in self.references]
                    or metadata.get("model") != self.config["embedding_model"]
                    or metadata.get("revision") != self.config["embedding_revision"]
                    or metadata.get("role") != "document" or self.matrix.ndim != 2):
                raise ValueError("Reference embeddings do not match the configured corpus/model")
            for start in range(0, len(self.matrix), 512):
                block = self.matrix[start:start + 512]
                if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=.001):
                    raise ValueError("Reference embeddings must be finite unit vectors")
            self.embedder = QwenEmbedder(self.config)
            self.mode = "Qwen embedding similarity · GB10"
            if kev_run:
                from .web_ranker import WebsiteRanker
                self.ranker = WebsiteRanker(kev_run, self.config)
                self.mode = "Qwen similarity + fine-tuned Kev"
        else:
            if kev_run:
                raise ValueError("Kev ranking requires Qwen reference vectors")
            self.index = LexicalIndex(self.references)
            self.mode = "Text similarity · preview"

    def info(self):
        years = [p["year"] for p in self.references]
        return {"demo": all(p["paper_id"].startswith("demo:") for p in self.references), "mode": self.mode, "semantic": self.embedder is not None, "ranking": "kev" if self.ranker else "retrieval",
                "reference_hash": self.reference_hash,
                "embedding_model": self.config["embedding_model"] if self.embedder else None,
                "embedding_revision": self.config["embedding_revision"] if self.embedder else None,
                "journals": len({p["journal_id"] for p in self.references}), "references": len(self.references),
                "years": f"{min(years)}–{max(years)}"}

    def suggest(self, body):
        query = validate_query(body)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another search is running. Please try again in a moment.")
        try:
            if self.embedder:
                with tempfile.TemporaryDirectory(prefix="journal-query-") as cache:
                    self.embedder.cache = Path(cache)
                    scores = (self.matrix @ self.embedder.encode([query], query=True)[0]).tolist()
            else:
                scores = self.index.scores(query)
            if max(scores, default=0) <= 0:
                return {"suggestions": [], "mode": self.mode, "notice": "No matching terms found. Try a more detailed mathematical abstract."}
            candidates, _ = shortlist(query, self.references, scores, top_k=20)
            if self.ranker:
                from .examples import build_request
                from .web_ranker import CONTEXT
                request, _ = build_request(query, candidates, self.names, tokenizer=self.ranker.tokenizer,
                                          **CONTEXT, candidate_order="retrieval")
                probs = self.ranker.predict(request)
                candidates.sort(key=lambda c: (-probs[c["journal_id"]], c["journal_id"]))
            return {"mode": self.mode, "suggestions": [{"journal_id": c["journal_id"], "journal_name": self.names[c["journal_id"]],
                    "references": [citation(p) for p in c["references"]]} for c in candidates[:5]]}
        finally:
            self.lock.release()


class SSHSearchService:
    """Lazy authenticated connection to a persistent GB10 worker."""
    def __init__(self, host, project=None):
        project = project or os.environ.get("JOURNAL_REMOTE_PROJECT")
        if not project:
            raise ValueError("Set JOURNAL_REMOTE_PROJECT to the remote checkout path")
        if not host or host.startswith("-") or any(c.isspace() for c in host):
            raise ValueError("Invalid SSH host")
        command = ("cd " + shlex.quote(project)
                   + " && env JOURNAL_EXECUTION_BACKEND=gb10 JOURNAL_CUDA_MEMORY_GIB=24"
                   + " HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4"
                   + " " + shlex.quote(os.environ.get("JOURNAL_REMOTE_PYTHON", ".venv/bin/python"))
                   + " -u -m journal_suggester.web_worker")
        self.command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
                        "-o", "ServerAliveCountMax=2", host, command]
        self.process = None
        self.lock = threading.Lock()
        refs = read_jsonl(DEFAULT_SPLIT / "reference.jsonl")
        config = read_json("configs/models.json")
        years = [p["year"] for p in refs]
        self.expected = {"mode": "Qwen embedding similarity · GB10", "semantic": True, "ranking": "retrieval",
                         "reference_hash": digest(refs), "embedding_model": config["embedding_model"],
                         "embedding_revision": config["embedding_revision"], "journals": len({p["journal_id"] for p in refs}),
                         "references": len(refs), "years": f"{min(years)}–{max(years)}"}
        atexit.register(self.close)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process:
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass
        self.process = None

    def info(self):
        return dict(self.expected)

    def _receive(self):
        if not select.select([self.process.stdout], [], [], 180)[0]:
            raise RuntimeError("GB10 did not respond in time. Check the connection and try again.")
        line = self.process.stdout.readline(1024 * 1024)
        if not line:
            raise RuntimeError("Could not connect to the GB10 search service. Check SSH access and its environment.")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise ValueError("Invalid GB10 response")
        return result

    def suggest(self, body):
        query = validate_query(body)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another search is running. Please try again in a moment.")
        try:
            if self.process is None or self.process.poll() is not None:
                self.close()
                self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
                if self._receive().get("ready") != self.expected:
                    raise RuntimeError("GB10's model or reference corpus differs from this preview. Sync and check its data before searching.")
            self.process.stdin.write(json.dumps(query) + "\n")
            self.process.stdin.flush()
            result = self._receive()
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result
        except (OSError, ValueError, RuntimeError) as exc:
            self.close()
            raise RuntimeError(str(exc) if isinstance(exc, RuntimeError) else "The GB10 connection was interrupted. Please try again.") from None
        finally:
            self.lock.release()
