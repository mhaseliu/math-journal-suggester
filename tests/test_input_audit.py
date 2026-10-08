import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

from journal_suggester.input_audit import audit_inputs
from journal_suggester.io import digest, request_digest, write_json, write_jsonl


@unittest.skipIf(np is None, 'Run numeric audit fixture in the GB10 environment')
class InputAuditTests(unittest.TestCase):
    def test_independent_audit_accepts_correct_inputs_and_rejects_tampering(self):
        class Tokenizer:
            def encode(self, text, **kwargs): return text.split()
            def decode(self, words, **kwargs): return ' '.join(words)
        tok = Tokenizer()
        models = {'query_tokens': 768, 'reference_tokens': 100}
        catalog = [{'journal_id': f'j{i:02}', 'journal_name': f'Journal {i}'} for i in range(20)]
        query = {'paper_id': 'query', 'group_id': 'query', 'title': 'Query title', 'abstract': 'Query abstract', 'journal_id': 'j00'}
        refs = [query] + [{'paper_id': f'r{i:02}{suffix}', 'group_id': f'r{i:02}{suffix}',
                          'title': f'Paper r{i:02}{suffix}', 'abstract': 'Mathematical abstract', 'journal_id': f'j{i:02}'}
                         for i in range(20) for suffix in 'ab']
        order = [j['journal_id'] for j in catalog]
        evidence = {j: [f'r{i:02}a', f'r{i:02}b'] for i, j in enumerate(order)}
        state = ['Manuscript:\nQuery title Query abstract']
        for i, j in enumerate(catalog):
            state.append('Candidate journal: ' + j['journal_name'])
            state += [f'Reference: Paper r{i:02}{s} Mathematical abstract' for s in 'ab']
        request = {'state': '\n\n'.join(state), 'questions': {'journal': {
            'criteria': {j['journal_id']: j['journal_name'] for j in catalog}, 'label': 'j00'}}}
        meta = {'paper_id': 'query', 'target': 'j00', 'natural_journals': order, 'candidates': order,
                'inserted': False, 'evidence_ids': evidence, 'request_hash': request_digest(request)}
        with tempfile.TemporaryDirectory() as tmp, patch('journal_suggester.input_audit.journals', return_value=catalog):
            root = Path(tmp); source, vectors, directory = root / 'source', root / 'queries', root / 'inputs'
            write_jsonl(source / 'splits/reference.jsonl', refs)
            for location, name, papers, role in [(source / 'vectors', 'reference', refs, 'document'), (vectors, 'train', [query], 'query')]:
                location.mkdir(parents=True, exist_ok=True)
                values = np.zeros((len(papers), 4096), dtype=np.float32); values[:, 0] = 1
                np.save(location / f'{name}.npy', values)
                write_json(location / f'{name}.json', {'input_hash': digest(papers), 'config_hash': digest(models),
                    'paper_ids': [p['paper_id'] for p in papers], 'role': role})
            write_jsonl(directory / 'train.jsonl', [request])
            write_jsonl(directory / 'train.meta.jsonl', [meta])
            result = audit_inputs(source, vectors, 'train', [query], directory, models, tok, training=True)
            self.assertTrue(result['checks_pass'])
            meta['evidence_ids']['j00'][0] = 'query'
            write_jsonl(directory / 'train.meta.jsonl', [meta])
            with self.assertRaises(AssertionError):
                audit_inputs(source, vectors, 'train', [query], directory, models, tok, training=True)
            meta['evidence_ids']['j00'][0] = 'r00a'
            request['questions']['journal']['criteria'] = dict(reversed(list(request['questions']['journal']['criteria'].items())))
            meta['request_hash'] = request_digest(request)
            write_jsonl(directory / 'train.jsonl', [request]); write_jsonl(directory / 'train.meta.jsonl', [meta])
            with self.assertRaises(AssertionError):
                audit_inputs(source, vectors, 'train', [query], directory, models, tok, training=True)


if __name__ == '__main__':
    unittest.main()
