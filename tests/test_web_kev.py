import hashlib
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from journal_suggester.examples import build_request
from journal_suggester.web_ranker import CHECKPOINT_FILES, CONTEXT, verify_checkpoint
from journal_suggester.web_service import SearchService




class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = {'kev_code_revision': 'code-pin', 'base_model': 'base', 'base_revision': 'base-pin'}
        files = {}
        for name in CHECKPOINT_FILES:
            data = name.encode()
            (self.root / name).write_bytes(data)
            files[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        manifest = {**self.models, 'selected_epoch': 2, 'optimizer_steps': 2000, 'files': files}
        raw = json.dumps(manifest).encode()
        (self.root / 'manifest.json').write_bytes(raw)
        self.spec = {'epoch': 2, 'optimizer_steps': 2000, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}

    def test_valid_checkpoint_and_same_size_tampering(self):
        self.assertEqual(verify_checkpoint(self.root, self.spec, self.models)['selected_epoch'], 2)
        file = self.root / 'head.pt'
        file.write_bytes(b'x' * file.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verify_checkpoint(self.root, self.spec, self.models)

    def test_manifest_and_wrong_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            verify_checkpoint(self.root, {**self.spec, 'epoch': 3.5}, self.models)
        file = self.root / 'manifest.json'
        file.write_text(file.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            verify_checkpoint(self.root, self.spec, self.models)

    def test_symlink_rejected_even_if_contents_match(self):
        file = self.root / 'head.pt'
        file.rename(self.root / 'elsewhere')
        file.symlink_to('elsewhere')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            verify_checkpoint(self.root, self.spec, self.models)


class RerankTests(unittest.TestCase):
    def test_kev_reorders_candidates_and_preserves_exact_citations(self):
        service = SearchService.__new__(SearchService)
        service.lock = threading.Lock()
        service.embedder = None
        service.index = SimpleNamespace(scores=lambda query: [1.0] * 40)
        service.mode = 'Qwen similarity + fine-tuned Kev'
        service.names = {f'j{i}': f'Journal {i}' for i in range(20)}
        service.references = []
        candidates = []
        for i in range(20):
            refs = [{'paper_id': f'{i}-{j}', 'journal_id': f'j{i}', 'title': f'Reference {i}-{j}',
                     'abstract': 'Original reference abstract.', 'year': 2020} for j in range(2)]
            service.references.extend(refs)
            candidates.append({'journal_id': f'j{i}', 'score': 1 - i / 20, 'references': refs})
        def predict(request):
            question = request['questions']['journal']
            self.assertNotIn('label', question)
            self.assertEqual(list(question['criteria']), list(service.names))
            return {f'j{i}': (i + 1) / 210 for i in range(20)}
        service.ranker = SimpleNamespace(tokenizer=None, predict=Mock(side_effect=predict))
        query = {'title': 'Spectral gaps of graphs', 'abstract': 'We study spectral gaps in regular graphs and mixing times.'}
        with patch('journal_suggester.web_service.shortlist', return_value=(candidates, {})) as shortlist, \
                patch('journal_suggester.examples.build_request', wraps=build_request) as build:
            result = service.suggest(query)
        self.assertEqual(shortlist.call_args.kwargs['top_k'], 20)
        for name, value in CONTEXT.items():
            self.assertEqual(build.call_args.kwargs[name], value)
        self.assertEqual(build.call_args.kwargs['candidate_order'], 'retrieval')
        self.assertEqual([j['journal_id'] for j in result['suggestions']], ['j19', 'j18', 'j17', 'j16', 'j15'])
        originals = {r['paper_id']: r for r in service.references}
        for journal in result['suggestions']:
            self.assertEqual(len(journal['references']), 2)
            for reference in journal['references']:
                original = originals[reference['paper_id']]
                self.assertEqual(original['journal_id'], journal['journal_id'])
                self.assertEqual(reference['title'], original['title'])
                self.assertEqual(reference['abstract'], original['abstract'])
        self.assertFalse(service.lock.locked())


if __name__ == '__main__':
    unittest.main()
