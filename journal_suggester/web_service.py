"""Rank journals directly with the selected Kev model."""
import math
import threading

from .io import journals


def validate_query(body):
    if not isinstance(body, dict) or any(not isinstance(body.get(k), str) for k in ('title', 'abstract')):
        raise ValueError('Provide a title and abstract.')
    query = {k: ' '.join(body[k].split()) for k in ('title', 'abstract')}
    if not 3 <= len(query['title']) <= 2000 or not 40 <= len(query['abstract']) <= 30000:
        raise ValueError('Enter a title (3–2,000 characters) and an abstract (40–30,000 characters).')
    if query['title'].casefold() == 'introduction':
        raise ValueError('Please use the manuscript title, rather than a section heading.')
    for key in ('doi', 'arxiv_id'):
        if isinstance(body.get(key), str) and len(body[key]) <= 300:
            query[key] = body[key]
    return query


class SearchService:
    def __init__(self, kev_run=None, *, ranker=None):
        if not kev_run and ranker is None:
            raise ValueError('Supply the verified Kev-only checkpoint with --kev-run.')
        self.names = {j['journal_id']: j['journal_name'] for j in journals()}
        self.lock = threading.Lock()
        self.mode = 'Fine-tuned Kev'
        if ranker is None:
            from .web_ranker import KevOnlyRanker
            ranker = KevOnlyRanker(kev_run)
        self.ranker = ranker

    def info(self):
        return {'mode': self.mode, 'ranking': 'kev-only', 'journals': len(self.names),
                'similar_papers': False}

    def suggest(self, body):
        query = validate_query(body)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Another search is running. Please try again in a moment.')
        try:
            probabilities = self.ranker.recommend(query)
            if (set(probabilities) != set(self.names)
                    or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5)):
                raise ValueError('Invalid journal probabilities')
            selected = sorted(probabilities, key=lambda j: (-probabilities[j], j))[:5]
            return {'mode': self.mode, 'suggestions': [
                {'journal_id': journal, 'journal_name': self.names[journal],
                 'references': []}
                for journal in selected]}
        finally:
            self.lock.release()
