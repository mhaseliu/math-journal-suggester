import tempfile
import unittest
from pathlib import Path

from journal_suggester.corpus import openalex_match, ordered_candidates, publisher_abstract
from journal_suggester.io import read_jsonl, write_json, write_jsonl
from journal_suggester.records import deduplicate, identities
from journal_suggester.splits import floor_quotas, split
def paper(i, journal="j"):
    return {"paper_id": f"p{i}", "title": f"Mathematical title {i}", "abstract": "A sufficiently long mathematical abstract about rings and algebra.",
            "journal_id": journal, "year": 2020, "doi": f"10.1234/{i}", "arxiv_id": "", "url": f"https://doi.org/10.1234/{i}"}


class EvaluationCorpusTests(unittest.TestCase):
    def test_rare_journal_floor_and_publication_weights(self):
        counts = {"large": 999, "small": 1}
        self.assertEqual(floor_quotas(10, counts), {"large": 9, "small": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            # Equal available counts must not override the publication-count distribution.
            rows = [paper(i, "large" if i < 30 else "small") for i in range(60)]
            write_jsonl(path / "input", rows)
            result = split(path / "input", path / "out", 0, 10, 10, counts=counts, evaluation_floor=1)
            self.assertEqual(result["validation_quotas"], {"large": 9, "small": 1})
            self.assertEqual(result["test_quotas"], result["validation_quotas"])
            groups = [{key for p in read_jsonl(path / "out" / f"{name}.jsonl") for key in identities(p)}
                      for name in ("reference", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            write_jsonl(path / "out/test.jsonl", [])
            with self.assertRaisesRegex(ValueError, "partition was altered"):
                split(path / "input", path / "out", 0, 10, 10, counts=counts, evaluation_floor=1)

    def test_shortage_fails_before_freezing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_jsonl(path / "input", [paper(i, "large") for i in range(30)])
            with self.assertRaisesRegex(ValueError, "split not frozen"):
                split(path / "input", path / "out", 0, 10, 10, counts={"large": 999, "missing": 1}, evaluation_floor=1)
            self.assertFalse((path / "out" / "manifest.json").exists())

    def test_old_duplicate_alias_survives_enrichment(self):
        a, b = paper(1), paper(2)
        a["duplicate_keys"] = [["doi", b["doi"]]]
        a["versions"] = ["old-preprint"]
        grouped, _ = deduplicate([a, b])
        self.assertEqual(len(grouped), 1)
        self.assertIn("old-preprint", grouped[0]["versions"])

    def test_seeded_collection_is_independent_of_api_order_and_excludes_pilot(self):
        papers = [paper(i) for i in range(20)]
        excluded = set(identities(papers[0]))
        a = ordered_candidates(papers, 42, "j", excluded)
        b = ordered_candidates(list(reversed(papers)), 42, "j", excluded)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 19)
        self.assertNotEqual([p["paper_id"] for p in a], sorted(p["paper_id"] for p in a))

    def test_abstract_merge_requires_both_doi_and_title(self):
        p = paper(1)
        self.assertTrue(openalex_match(p, {"doi": p["doi"], "title": p["title"]}))
        self.assertFalse(openalex_match(p, {"doi": p["doi"], "title": "Different work"}))
        self.assertFalse(openalex_match(p, {"doi": "10.1/other", "title": p["title"]}))

    def test_publisher_content_must_belong_to_the_work(self):
        p = paper(1, "new-york-journal-of-mathematics")
        body = '<p class="style12">Different title</p>Abstract<blockquote>Some abstract</blockquote>'
        self.assertEqual(publisher_abstract(p, body), "")
        p["journal_id"] = "asterisque"
        body = '<strong>DOI :</strong> 10.1/other<div class="product__data-resume data-english">Some abstract</div>'
        self.assertEqual(publisher_abstract(p, body), "")


    def test_collector_resume_uses_completed_journal_without_network(self):
        from unittest.mock import patch
        from journal_suggester.corpus import collect
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_jsonl(path / "ledger", [paper(i) for i in range(10)])
            write_jsonl(path / "pilot", [paper(20)])
            write_json(path / "counts", {"j": 10})
            write_json(path / "config", {"ledger": str(path / "ledger"), "publication_counts": str(path / "counts"),
                       "exclude_previous_pilot": str(path / "pilot"), "validation_queries": 1, "test_queries": 1,
                       "reference_target": 3, "minimum_per_journal": {"validation": 1, "test": 1, "reference": 3},
                       "collection_spares_per_journal": 0, "seed": 42})
            def enrich(rows, *args):
                for row in rows:
                    row["abstract"] = " ".join(["mathematics"] * 20)
                return [{"paper_id": p["paper_id"], "abstract_source": "fixture"} for p in rows], []
            with patch("journal_suggester.corpus.journals", return_value=[{"journal_id": "j", "issns": ""}]), \
                 patch("journal_suggester.corpus.crossref_abstracts", return_value={}), \
                 patch("journal_suggester.corpus.enrich_batch", side_effect=enrich) as network:
                collect(path / "config", path / "out")
                self.assertEqual(network.call_count, 1)
                collect(path / "config", path / "out", offline=True)
                self.assertEqual(network.call_count, 1)
                self.assertEqual(len(read_jsonl(path / "out/papers.jsonl")), 5)

    def test_arxiv_cooldown_preserves_cache_and_stops_new_requests(self):
        import hashlib
        import time
        from unittest.mock import patch
        from journal_suggester.http import CachedHTTP
        with tempfile.TemporaryDirectory() as tmp:
            client = CachedHTTP(tmp)
            url = "https://export.arxiv.org/api/query"
            Path(tmp, hashlib.sha256(url.encode()).hexdigest() + ".body").write_text("cached feed")
            client.arxiv_retry_at = time.monotonic() + 300
            with patch("urllib.request.urlopen") as network:
                self.assertEqual(client.get(url), "cached feed")
                with self.assertRaisesRegex(RuntimeError, "cooling down"):
                    client.get(url, {"id_list": "new"})
                network.assert_not_called()
