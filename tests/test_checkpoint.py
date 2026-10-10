import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from journal_suggester.web_ranker import CHECKPOINT_FILES, verify_checkpoint


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
