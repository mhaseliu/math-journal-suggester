import json
import tempfile
import unittest
from pathlib import Path

from journal_suggester.catalog import valid_issn
from journal_suggester.collect import crossref_record, inverted_abstract, publication_year
from journal_suggester.evaluation import metrics, paired_top3
from journal_suggester.examples import build_request
from journal_suggester.gpu import require_gb10
from journal_suggester.io import read_jsonl, write_jsonl
from journal_suggester.records import clean_text, deduplicate
from journal_suggester.retrieval import shortlist
from journal_suggester.splits import quotas, split


def paper(i, journal="j"):
    return {"paper_id": f"p{i}", "title": f"Mathematical title {i}", "abstract": "A sufficiently long mathematical abstract about rings and algebra.",
            "journal_id": journal, "year": 2020, "doi": f"10.1234/{i}", "arxiv_id": "", "url": f"https://doi.org/10.1234/{i}"}


class PipelineTests(unittest.TestCase):
    def test_issn_checksum(self):
        self.assertTrue(valid_issn("0001-5962"))
        self.assertFalse(valid_issn("0001-5963"))

    def test_real_abstract_normalization_preserves_math(self):
        self.assertEqual(clean_text("<jats:title>Abstract</jats:title><jats:p>We study <i>x</i> &lt; y.</jats:p>"), "We study x < y.")
        self.assertEqual(clean_text("Abstract harmonic analysis"), "Abstract harmonic analysis")
        self.assertEqual(clean_text("We study Annals of probability in mathematical prose."), "We study Annals of probability in mathematical prose.")
        self.assertEqual(clean_text("We prove a theorem. To appear in Journal of X."), "We prove a theorem.")
        self.assertEqual(clean_text("To appear in Journal of X. We study algebra."), "We study algebra.")

    def test_inverted_index(self):
        self.assertEqual(inverted_abstract({"a": [0, 2], "ring": [1]}), "a ring a")
        with self.assertRaises(ValueError):
            inverted_abstract({"x": [0], "y": [0]})
        with self.assertRaises(ValueError):
            inverted_abstract({"x": [3]})

    def test_publication_year_is_not_preprint_year(self):
        self.assertEqual(publication_year({"published-print": {"date-parts": [[2020]]}, "published-online": {"date-parts": [[2019]]}}), (2020, "published-print"))

    def test_crossref_issn_and_work_type(self):
        journal = {"journal_id": "j", "issns": "0001-5962"}
        row = {"title": ["A work"], "DOI": "10.1/a", "type": "journal-article", "ISSN": ["1234-5678"], "issued": {"date-parts": [[2020]]}}
        self.assertEqual(crossref_record(row, journal)[1], "issn_mismatch")
        row["ISSN"] = ["0001-5962"]
        row["title"] = ["Erratum to a work"]
        self.assertEqual(crossref_record(row, journal)[1], "unsuitable_title")

    def test_transitive_duplicates_and_conflicting_labels(self):
        a, b, c = paper(1), paper(2), paper(3)
        b["doi"] = a["doi"]
        c["title"] = b["title"]
        good, bad = deduplicate([a, b, c])
        self.assertEqual(len(good), 1)
        self.assertEqual(len(good[0]["versions"]), 3)
        c["journal_id"] = "different"
        good, bad = deduplicate([a, b, c])
        self.assertFalse(good)
        self.assertEqual(bad[0]["reason"], "conflicting_journal_labels")

    def test_math_symbols_not_erased_from_title_identity(self):
        a, b = paper(1), paper(2)
        a["title"], b["title"] = "Bounds on x+y", "Bounds on x-y"
        self.assertEqual(len(deduplicate([a, b])[0]), 2)

    def test_quota_redistribution(self):
        result = quotas(10, {"a": 90, "b": 10}, {"a": 2, "b": 30})
        self.assertEqual(result, {"a": 2, "b": 8})

    def test_frozen_stratified_split_and_version_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [paper(i, f"j{i // 10}") for i in range(30)]
            rows.append({**rows[0], "paper_id": "another-version"})
            write_jsonl(p / "input", rows)
            first = split(p / "input", p / "out", 12, 6, 6)
            second = split(p / "input", p / "out", 12, 6, 6)
            self.assertEqual(first, second)
            pools = [{r["group_id"] for r in read_jsonl(p / "out" / f"{s}.jsonl")} for s in ("reference", "validation", "test")]
            self.assertEqual(sum(len(g) for g in pools), 30)
            self.assertFalse(pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])
            with self.assertRaises(ValueError):
                split(p / "input", p / "out", 11, 6, 6)

    def test_natural_candidates_do_not_use_target(self):
        refs = [paper(i, f"j{i}") for i in range(12)]
        query = paper(100, "j11")
        candidates, info = shortlist(query, refs, list(range(12, 0, -1)))
        self.assertNotIn("j11", info["natural_journals"])
        forced, inserted = shortlist(query, refs, list(range(12, 0, -1)), target="j11")
        self.assertEqual(len(forced), 10)
        self.assertTrue(inserted["inserted"])
        self.assertEqual(len(candidates), 10)

    def test_self_versions_excluded_before_scoring(self):
        a, b = paper(1), paper(2)
        b["title"] = a["title"]
        refs, _ = deduplicate([a, b, paper(3)])
        choices, _ = shortlist(b, refs, [100, .1])
        self.assertEqual([p["paper_id"] for c in choices for p in c["references"]], ["p3"])

    def test_missing_target_metrics_denominator(self):
        rows = [{"target": "a", "candidates": ["a", "b"], "ranking": ["a", "b"]},
                {"target": "c", "candidates": ["a", "b"], "ranking": ["b", "a"]}]
        result = metrics(rows)
        self.assertEqual(result["top3"], .5)
        self.assertEqual(result["mrr"], .5)
        self.assertEqual(result["conditional_top1"], 1)

    def test_examples_do_not_encode_retrieval_metadata(self):
        choices = [{"journal_id": "j", "score": .98, "references": [paper(1), paper(2)]}]
        request, _ = build_request(paper(9), choices, {"j": "Journal"})
        self.assertNotIn("label", request["questions"]["journal"])
        for field in ("score", "inserted", "doi", "year", "paper_id"):
            self.assertNotIn(field, json.dumps(request))

    def test_paired_bootstrap_rejects_different_queries(self):
        with self.assertRaises(ValueError):
            paired_top3([{"paper_id": "a"}], [{"paper_id": "b"}])

    def test_candidate_order_survives_serialization_and_changes_hash(self):
        from journal_suggester.io import request_digest
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.jsonl"
            row = {"questions": {"journal": {"criteria": {"z": "Last", "a": "First"}}}}
            write_jsonl(path, [row])
            restored = read_jsonl(path)[0]
            self.assertEqual(list(restored["questions"]["journal"]["criteria"]), ["z", "a"])
            other = {"questions": {"journal": {"criteria": {"a": "First", "z": "Last"}}}}
            self.assertNotEqual(request_digest(row), request_digest(other))

    def test_laptop_cannot_enter_model_execution(self):
        from unittest.mock import patch
        with patch("journal_suggester.gpu.platform.machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "GB10-only"):
                require_gb10()

    def test_offline_pipeline_schema_and_heldout_boundaries(self):
        from journal_suggester.pipeline import prepare
        from journal_suggester.io import read_json
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            ids = ["journal-of-algebra", "journal-of-number-theory", "algebraic-geometry"]
            write_jsonl(p / "input", [paper(i, ids[i // 10]) for i in range(30)])
            split(p / "input", p / "split", 12, 6, 6)
            report = prepare(p / "split", p / "prepared")
            self.assertEqual(report["backend"], "lexical-smoke")
            self.assertEqual(report["train"]["n"], 12)
            for name in ("validation", "test"):
                for row in read_jsonl(p / "prepared" / f"{name}.jsonl"):
                    self.assertNotIn("label", row["questions"]["journal"])
            for row, meta in zip(read_jsonl(p / "prepared" / "train.jsonl"), read_jsonl(p / "prepared" / "train.meta.jsonl")):
                self.assertEqual(row["questions"]["journal"]["label"], meta["target"])
                self.assertNotIn(meta["paper_id"], [i for group in meta["evidence_ids"].values() for i in group])
            self.assertFalse((p / "prepared" / "retrieval.test.predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
