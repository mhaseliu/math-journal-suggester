import json
import tempfile
import unittest
from pathlib import Path

from journal_suggester.counts import aggregate
from journal_suggester.http import CachedHTTP
from journal_suggester.io import digest, read_json, write_json, write_jsonl
from journal_suggester.records import deduplicate
from journal_suggester.series_counts import (
    ams_index, asterisque_rows, bulk_matches, check_memoir_matches, memoirs_rows, smf_cards,
)


class SeriesCountTests(unittest.TestCase):
    def test_memoir_corrections_are_excluded_under_the_same_work_rule(self):
        records = [{"number": 1332, "volume": 263, "doi": "10.1090/memo/1332", "title": "Corrigendum and improvements to a theorem",
                    "status": "matched", "year": 2021, "matches": [{"accepted": True, "year": "2021", "zbl_id": "1"}]}]
        window = {"first_number": 1332, "last_number": 1332, "first_volume": 263, "last_volume": 263, "boundary_checks": {}}
        rows, ledger, unresolved = memoirs_rows(records, {}, window)
        self.assertFalse(rows or unresolved)
        self.assertEqual(ledger[0]["status"], "excluded_unsuitable_title")

    def test_generic_titles_need_identifiers_and_keep_stable_groups(self):
        def paper(i, doi="", journal="a"):
            return {"paper_id": str(i), "title": "Introduction", "journal_id": journal, "doi": doi}
        a, b, c = paper(1, "10.1/a"), paper(2, "10.1/b"), paper(3, journal="b")
        good, bad = deduplicate([a, b, c, paper(4, journal="b")])
        self.assertEqual(len(good), 4)
        self.assertEqual(len({p["group_id"] for p in good}), 4)
        self.assertFalse(bad)
        self.assertEqual(len(deduplicate([a, {**a, "paper_id": "alias"}])[0]), 1)

    def test_ams_product_code_wins_over_typo_without_treating_copyright_as_year(self):
        body = "productDetailObj = " + json.dumps({"SeriesVolumes": [{
            "ProductCode": "MEMO/320/1627", "CopyrightYear": "2026", "IssueNumber": 1227}]}) + ";"
        row = ams_index(body)[1627]
        self.assertTrue(row["publisher_number_conflict"])
        self.assertEqual(row["volume"], 320)
        self.assertNotIn("year", row)

    def test_smf_catalog_validates_year_and_decodes_urls(self):
        body = ('<a class="publication card" href="/publications/a?x=1&amp;y=2">'
                '<h3 class="publication__title">A theorem</h3><div class="publication__text">Author</div>'
                '<div class="publication__collection">Astérisque</div>'
                '<div class="publication__more_info">Article - Tome 415 - pp 215-222 - 2020</div></a>')
        self.assertEqual(smf_cards(body, 2020)[0]["url"], "https://smf.emath.fr/publications/a?x=1&y=2")
        with self.assertRaises(ValueError):
            smf_cards(body, 2021)

    def test_smf_counts_components_recovers_missing_card_and_excludes_reprints(self):
        def book(volume, title, doi=""):
            return {"volume": volume, "kind": "Livre", "title": title, "year": 2020,
                    "url": "https://smf/" + volume, "dois": [doi] if doi else [], "contents": []}
        a, b, c = book("415", "Collected works"), book("452", "Networks", "10.1/reused"), book("223", "Reprint")
        child = {"url": "https://smf/child", "title": "Lattice hydrodynamics"}
        a["contents"] = [child, {"url": "https://smf/preface", "title": "Preface"}]
        rows, rejected, recovered = asterisque_rows([a, b, c], {r["volume"]: r for r in [a, b, c]},
            {"223": {"original_year": 1994}}, {"452": {"doi": "10.1/reused"}})
        self.assertEqual({r["title"] for r in rows}, {"Networks", "Lattice hydrodynamics"})
        self.assertEqual(next(r for r in rows if r["volume"] == "452")["paper_id"], b["url"])
        self.assertEqual({r["reason"] for r in rejected}, {"enclosing_collection", "unsuitable_title", "reprint"})
        self.assertTrue(any(r["url"] == child["url"] for r in recovered))

    def test_zbmath_preprint_and_wrong_number_cannot_supply_publication_year(self):
        match = {"title": "A theorem", "authors": "Smith, A.", "source": "Mem. Am. Math. Soc. 1234 (2020).", "year": "2020"}
        checked = check_memoir_matches("A theorem", ["Smith"], 1234, [match,
            {**match, "source": "arXiv:1234.5678 (2015).", "year": "2015"},
            {**match, "source": "Mem. Am. Math. Soc. 12345 (2020)."},
            {**match, "authors": "Jones, A."}])
        self.assertEqual([m["accepted"] for m in checked], [True, False, False, False])

    def test_bulk_offline_cache_uses_list_authors(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"queries": [{"t": "A theorem", "a": ["Smith"]}]}
            write_json(Path(tmp) / ("bulk-" + digest(payload) + ".json"), {"response": {"results": [[]]}})
            self.assertEqual(bulk_matches(CachedHTTP(cache=tmp, offline=True), [{"t": "A theorem", "a": "Smith"}]), [[]])

    def test_memoir_review_corrects_year_but_never_infers_missing_publication(self):
        records = [{"number": n, "volume": 245, "doi": f"10.1/{n}", "title": f"Work {n}",
                    "status": "matched", "year": 2016,
                    "matches": [{"accepted": True, "year": "2016", "zbl_id": "0000.00000"}]}
                   for n in (1156, 1157)]
        window = {"first_number": 1156, "last_number": 1157, "first_volume": 245,
                  "last_volume": 245, "boundary_checks": {}}
        overrides = {str(n): {"doi": f"10.1/{n}", "year": y, "source_kind": "reviewed", "evidence": ["https://source"]}
                     for n, y in [(1156, 2017), (1157, None)]}
        rows, ledger, unresolved = memoirs_rows(records, overrides, window)
        self.assertEqual([(r["number"], r["year"]) for r in rows], [(1156, 2017)])
        self.assertEqual(unresolved[0]["number"], 1157)
        with self.assertRaises(ValueError):
            memoirs_rows(records[:1], overrides, window)
        with tempfile.TemporaryDirectory() as tmp:
            jid = "memoirs-of-the-american-mathematical-society"
            for kind, status, papers in [("journals", "needs_series_review", []),
                                         ("publishers", "needs_series_review", rows)]:
                directory = Path(tmp) / kind / jid
                write_json(directory / "audit.json", {"status": status})
                write_jsonl(directory / "eligible.jsonl", papers)
            result = aggregate(tmp, [{"journal_id": jid, "journal_name": "Memoirs"}])
            self.assertEqual(result["journals"][jid]["observed_eligible_groups"], 1)
            self.assertIsNone(read_json(Path(tmp) / "counts.json")[jid])

            # Explicit review excludes a record even when its source claims an in-window year.
            overrides["1157"].update(exclude=True, year=2016, source_kind="user_manual_review")
            rows, ledger, unresolved = memoirs_rows(records, overrides, window)
            self.assertFalse(unresolved)
            self.assertEqual(len(rows), 1)
            excluded = next(r for r in ledger if r["number"] == 1157)
            self.assertEqual(excluded["status"], "excluded_after_review")
            self.assertIsNone(excluded["year"])
            self.assertEqual(excluded["source_url"], "")
            write_json(directory / "audit.json", {"status": "complete"})
            write_jsonl(directory / "eligible.jsonl", rows)
            aggregate(tmp, [{"journal_id": jid, "journal_name": "Memoirs"}])
            self.assertEqual(read_json(Path(tmp) / "counts.json")[jid], 1)


if __name__ == "__main__":
    unittest.main()
