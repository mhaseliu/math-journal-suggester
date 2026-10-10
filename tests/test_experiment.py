from pathlib import Path
import hashlib
import shutil
import tempfile
import tomllib
import unittest

from journal_suggester.experiment import select, verify_results
from journal_suggester.io import (
    digest,
    journals,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


class ExperimentTests(unittest.TestCase):
    def test_bundled_resources_match_canonical_files(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text())
        package_data = config["tool"]["setuptools"]["package-data"]["journal_suggester"]
        for name in package_data:
            if not name.startswith("resources/"):
                continue
            with self.subTest(resource=name):
                bundled = root / "journal_suggester" / name
                canonical = root / name.removeprefix("resources/")
                self.assertEqual(bundled.read_bytes(), canonical.read_bytes())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = [j["journal_id"] for j in journals()]
        self.parts = {}
        for part, size in (("reference", 95), ("validation", 95), ("test", 1000)):
            self.parts[part] = [
                {
                    "paper_id": f"{part}-{i}",
                    "group_id": f"{part}-{i}",
                    "title": f"Title {part} {i}",
                    "abstract": f"Abstract {part} {i}",
                    "journal_id": self.ids[i % 95],
                    "year": 2020,
                }
                for i in range(size)
            ]
            write_jsonl(self.root / f"source/{part}.jsonl", self.parts[part])
        for part, source in (
            ("train", "reference"),
            ("screen", "reference"),
            ("validation", "validation"),
            ("test", "test"),
        ):
            write_jsonl(
                self.root / f"ids/{part}.jsonl",
                [
                    {"paper_id": p["paper_id"], "record_sha256": digest(p)}
                    for p in self.parts[source]
                ],
            )
        self.manifest = {
            "fingerprint": "fixture",
            "partition_hashes": {
                part: digest(rows) for part, rows in self.parts.items()
            },
        }
        write_json(self.root / "ids/manifest.json", self.manifest)
        config = read_json("configs/kev-only.json")
        config["full"]["training_papers"] = config["screen"]["training_papers"] = 95
        config["validation_papers"] = 95
        write_json(self.root / "config.json", config)

    def select(self):
        return select(
            self.root / "source",
            self.root / "output",
            self.root / "ids",
            self.root / "config.json",
        )

    def test_exact_selection_excludes_test_text_and_refuses_overwrite(self):
        self.assertTrue(self.select()["checks_pass"])
        self.assertEqual(
            read_jsonl(self.root / "output/source/train.jsonl"), self.parts["reference"]
        )
        self.assertEqual(read_json(self.root / "output/selection.json")["test_hash"], digest(self.parts["test"]))
        test = read_jsonl(self.root / "output/source/test-identities.jsonl")
        self.assertEqual(len(test), 1000)
        self.assertTrue(
            all("title" not in row and "abstract" not in row for row in test)
        )
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.select()

    def test_changed_source_text_is_rejected(self):
        self.parts["test"][0]["abstract"] += " changed"
        write_jsonl(self.root / "source/test.jsonl", self.parts["test"])
        with self.assertRaisesRegex(ValueError, "benchmark"):
            self.select()
        self.assertFalse((self.root / "output").exists())

    def test_overlap_is_rejected_even_with_matching_source_hashes(self):
        self.parts["test"][0]["abstract"] = self.parts["reference"][0]["abstract"]
        write_jsonl(self.root / "source/test.jsonl", self.parts["test"])
        self.manifest["partition_hashes"]["test"] = digest(self.parts["test"])
        write_json(self.root / "ids/manifest.json", self.manifest)
        write_jsonl(
            self.root / "ids/test.jsonl",
            [
                {"paper_id": p["paper_id"], "record_sha256": digest(p)}
                for p in self.parts["test"]
            ],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.select()
        self.assertFalse((self.root / "output").exists())

    def test_published_scores_recompute_without_models(self):
        result = verify_results()
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(
            result["metrics"], {"top1": 0.572, "top3": 0.789, "top5": 0.853}
        )

    def test_baseline_requests_must_match_fine_tuned_requests(self):
        copied = self.root / "saved-results"
        shutil.copytree("results", copied)
        path = copied / "predictions/released_kev.jsonl"
        rows = read_jsonl(path)
        rows[0]["request_hash"] = "changed"
        write_jsonl(path, rows)
        verification = read_json(copied / "verification.json")
        verification["released_predictions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(copied / "verification.json", verification)
        with self.assertRaisesRegex(ValueError, "test requests differ"):
            verify_results(copied)

    def test_released_baseline_scores_recompute_without_models(self):
        self.assertEqual(verify_results()["released_kev"], {"top1": 0.157, "top3": 0.298, "top5": 0.373})
