"""Optional Qwen lookup after Kev has independently chosen the journals."""
from pathlib import Path
import tempfile

from .io import read_json, read_jsonl
from .retrieval import shortlist


class SimilarPapers:
    def __init__(self, split_dir, vectors):
        from .gpu import QwenEmbedder, load_vectors
        import numpy as np
        self.references = read_jsonl(Path(split_dir) / 'reference.jsonl')
        self.config = read_json('configs/models.json')
        self.matrix = load_vectors(vectors, 'reference', self.references, self.config)
        if self.matrix.ndim != 2 or not np.isfinite(self.matrix).all():
            raise ValueError('Invalid reference vectors')
        if not np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=.001):
            raise ValueError('Reference vectors must be normalized')
        self.embedder = QwenEmbedder(self.config)

    def find(self, query, journals):
        with tempfile.TemporaryDirectory(prefix='journal-query-') as cache:
            self.embedder.cache = Path(cache)
            scores = (self.matrix @ self.embedder.encode([query], query=True)[0]).tolist()
        candidates, _ = shortlist(query, self.references, scores, top_k=95)
        return {c['journal_id']: c['references'] for c in candidates if c['journal_id'] in journals}
