import tempfile
import unittest
from pathlib import Path

from journal_suggester.counts import aggregate, scan_journal
from journal_suggester.io import read_json, write_json, write_jsonl


JOURNAL = {"journal_id": "example", "journal_name": "Example", "issns": "0001-5962;1871-2509"}


def item(doi="10.1/a", title="A theorem", print_year=2020, online_year=2019):
    return {"DOI": doi, "title": [title], "ISSN": ["0001-5962"], "type": "journal-article",
            "published-print": {"date-parts": [[print_year]]}, "published-online": {"date-parts": [[online_year]]}}


class CountTests(unittest.TestCase):
    def test_title_refinement_preserves_original_audit(self):
        from unittest.mock import patch
        from journal_suggester.counts import METHOD, refine_titles
        from journal_suggester.io import digest, read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "old", Path(tmp) / "new"
            method = {**METHOD, "version": 3, "unsuitable_title_pattern": "^Frontmatter"}
            write_json(source / "manifest.json", {"method": method, "fingerprint": digest({"method": method, "catalog": [JOURNAL]})})
            rows = [{"paper_id": str(i), "title": title, "journal_id": "example", "year": 2020}
                    for i, title in enumerate(["A graph cover theorem", "COM volume 153 Issue 6 Cover and Front matter"])]
            write_jsonl(source / "journals/example/eligible.jsonl", rows)
            write_json(source / "journals/example/audit.json", {"status": "complete", "eligible_dois_before_title_dedup": 2})
            with patch("journal_suggester.counts.journals", return_value=[JOURNAL]):
                refine_titles(source, target)
            self.assertEqual(read_json(target / "counts.json"), {"example": 1})
            self.assertEqual(len(read_jsonl(source / "journals/example/eligible.jsonl")), 2)
            self.assertEqual(len(read_jsonl(target / "title-exclusions.jsonl")), 1)

    def test_frontmatter_variants_excluded_without_rejecting_index_mathematics(self):
        from journal_suggester.collect import crossref_record
        for title in ("Frontmatter", "Front matter", "Back-matter", "Special issue in memory of a mathematician", "Author index",
                      "COM volume 153 Issue 6 Cover and Front matter", "Publisher Correction: A theorem",
                      "A theorem – CORRIGENDUM", "RETRACTED ARTICLE: A theorem", "Errata to a theorem"):
            self.assertEqual(crossref_record(item(title=title), JOURNAL)[1], "unsuitable_title")
        self.assertIsNotNone(crossref_record(item(title="Index formulas for elliptic operators"), JOURNAL)[0])
        for title in ("Path Cover and Path Pack Inequalities", "Quantum Error Correction", "Tycho Brahe’s Calculi ad Corrigenda Elementa Orbitae Saturni"):
            self.assertIsNotNone(crossref_record(item(title=title), JOURNAL)[0])

    def test_date_union_issn_dedup_and_abstract_independence(self):
        class HTTP:
            def json(self, url, params):
                # Online 2015 / print 2016 must be found via the print-date request.
                rows = [item()]
                if "from-print-pub-date" in params["filter"]:
                    rows.append(item("10.1/b", "Boundary theorem", 2016, 2015))
                rows += [item("10.1/c", "Outside", 2015, 2016), item("10.1/d", "Erratum to a theorem")]
                return {"message": {"items": rows, "total-results": len(rows)}}
        rows, audit = scan_journal(JOURNAL, HTTP())
        self.assertEqual({p["doi"] for p in rows}, {"10.1/a", "10.1/b"})
        self.assertEqual(audit["status"], "complete")
        self.assertEqual(audit["unique_record_keys"], 4)
        self.assertEqual(audit["rejected"], {"outside_publication_window": 1, "unsuitable_title": 1})
        self.assertTrue(all(p["abstract"] == "" for p in rows))

    def test_unavailable_endpoint_does_not_become_zero(self):
        class HTTP:
            def json(self, url, params):
                raise RuntimeError("HTTP 404: endpoint")
        rows, audit = scan_journal(JOURNAL, HTTP())
        self.assertEqual(audit["status"], "incomplete")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "journals/example"
            write_json(directory / "audit.json", audit)
            write_jsonl(directory / "eligible.jsonl", rows)
            aggregate(tmp, [JOURNAL])
            self.assertIsNone(read_json(Path(tmp) / "counts.json")["example"])

    def test_cross_journal_conflicts_are_quarantined(self):
        from journal_suggester.collect import crossref_record
        with tempfile.TemporaryDirectory() as tmp:
            catalog = [JOURNAL, {**JOURNAL, "journal_id": "other"}]
            for journal in catalog:
                directory = Path(tmp) / "journals" / journal["journal_id"]
                row, _ = crossref_record(item(), journal)
                write_json(directory / "audit.json", {"status": "complete"})
                write_jsonl(directory / "eligible.jsonl", [row])
            result = aggregate(tmp, catalog)
            self.assertEqual(read_json(Path(tmp) / "counts.json"), {"example": 0, "other": 0})
            self.assertEqual(result["journals"]["example"]["conflicting_journal_groups"], 1)

    def test_short_page_below_reported_total_is_incomplete(self):
        class HTTP:
            def json(self, url, params):
                return {"message": {"items": [item()], "total-results": 10}}
        rows, audit = scan_journal(JOURNAL, HTTP())
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["status"], "incomplete")

    def test_publisher_archive_requires_all_links_and_excludes_corrections(self):
        from journal_suggester.publisher_counts import parse_nyjm
        body = ('<td class="style60">A theorem on rings</td><a href="/j/2025/31-1.html">abstract</a>'
                '<hr><td class="style60">Correction to a theorem</td><a href="/j/2025/31-2.html">abstract</a>')
        rows, rejected = parse_nyjm(body, 2025)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["year"], 2025)
        self.assertEqual(len(rejected), 1)
        with self.assertRaises(ValueError):
            parse_nyjm(body + '<hr><a href="/j/2025/31-3.html">abstract</a>', 2025)


if __name__ == "__main__":
    unittest.main()
