"""Exact all-reference search with journal maximum scores and duplicate exclusion."""
import math
import re
from collections import Counter, defaultdict

from .records import identities


def duplicate(query, reference):
    if query.get("group_id") and query["group_id"] == reference.get("group_id"):
        return True
    return bool(set(identities(query)) & {tuple(k) for k in reference.get("duplicate_keys", identities(reference))})


def shortlist(query, references, scores, top_k=10, target=None):
    if len(references) != len(scores):
        raise ValueError("Reference/vector count mismatch")
    grouped = defaultdict(list)
    query_keys = set(identities(query))
    for paper, score in zip(references, scores):
        if not math.isfinite(float(score)):
            raise ValueError("Non-finite similarity")
        same_group = query.get("group_id") and query["group_id"] == paper.get("group_id")
        paper_keys = paper.get("duplicate_keys")
        if paper_keys is None:
            paper_keys = identities(paper)
        if not same_group and not query_keys.intersection(tuple(k) for k in paper_keys):
            grouped[paper["journal_id"]].append((float(score), paper))
    for group in grouped.values():
        group.sort(key=lambda pair: (-pair[0], pair[1]["paper_id"]))
    ranked = sorted(grouped, key=lambda j: (-grouped[j][0][0], j))[:top_k]
    natural = list(ranked)
    inserted = target is not None and target not in ranked
    if inserted:
        if len(ranked) >= top_k:
            ranked[-1] = target
        else:
            ranked.append(target)
    candidates = [{"journal_id": jid, "score": grouped[jid][0][0] if grouped[jid] else None,
                   "references": [p for _, p in grouped[jid][:2]]} for jid in ranked]
    return candidates, {"natural_journals": natural, "inserted": inserted,
                         "missing_evidence": [c["journal_id"] for c in candidates if len(c["references"]) < 2]}


class LexicalIndex:
    """Dependency-free TF-IDF cosine smoke baseline, explicitly separate from Qwen."""
    backend = "lexical-smoke"

    def __init__(self, references):
        self.references = references
        texts = [self.tokens(p) for p in references]
        df = Counter(word for words in texts for word in set(words))
        self.idf = {w: math.log((1 + len(texts)) / (1 + n)) + 1 for w, n in df.items()}
        self.vectors = [self.vector(words) for words in texts]

    @staticmethod
    def tokens(paper):
        return re.findall(r"[\w]+", (paper["title"] + "\n" + paper["abstract"]).casefold())

    def vector(self, words):
        values = {w: (1 + math.log(n)) * self.idf[w] for w, n in Counter(words).items() if w in self.idf}
        norm = math.sqrt(sum(v * v for v in values.values())) or 1
        return {w: v / norm for w, v in values.items()}

    def scores(self, query):
        q = self.vector(self.tokens(query))
        return [sum(v * doc.get(w, 0) for w, v in q.items()) for doc in self.vectors]
