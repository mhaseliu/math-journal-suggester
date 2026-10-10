"""Kev journal recommendations with optional, independent similar-paper lookup."""
import math
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .io import journals
from .records import doi_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = Path('artifacts/reference')


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


def citation(paper):
    doi = doi_key(paper.get('doi', ''))
    url = 'https://doi.org/' + doi if doi else paper.get('url', '')
    parsed = urlsplit(url)
    if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username or parsed.password:
        url = ''
    return {k: paper.get(k, '') for k in ('paper_id', 'title', 'abstract', 'year')} | {'url': url, 'doi': doi}


class SearchService:
    def __init__(self, split_dir=None, vectors=None, kev_run=None, *, ranker=None, examples=None):
        if not kev_run and ranker is None:
            raise ValueError('Supply the verified Kev-only checkpoint with --kev-run.')
        if bool(split_dir) != bool(vectors):
            raise ValueError('Similar-paper lookup needs both reference data and vectors.')
        self.names = {j['journal_id']: j['journal_name'] for j in journals()}
        self.lock = threading.Lock()
        self.mode = 'Fine-tuned Kev'
        if ranker is None:
            from .web_ranker import KevOnlyRanker
            ranker = KevOnlyRanker(kev_run)
        self.ranker = ranker
        if vectors:
            from .related_papers import SimilarPapers
            examples = SimilarPapers(split_dir, vectors)
        self.examples = examples

    def info(self):
        return {'mode': self.mode, 'ranking': 'kev-only', 'journals': len(self.names),
                'similar_papers': self.examples is not None}

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
            references = {}
            if self.examples is not None:
                try:
                    references = self.examples.find(query, selected)
                except (RuntimeError, ValueError, OSError):
                    # Related-paper availability cannot change journal recommendations.
                    references = {}
            return {'mode': self.mode, 'suggestions': [
                {'journal_id': journal, 'journal_name': self.names[journal],
                 'references': [citation(p) for p in references.get(journal, [])]}
                for journal in selected]}
        finally:
            self.lock.release()
