import unittest

from journal_suggester.evaluation import metrics, paired_top3
from journal_suggester.examples import build_request
from journal_suggester.io import request_digest
from journal_suggester.order_check import averaged_probabilities, reverse_request, validate_probabilities
from journal_suggester.retrieval import shortlist


class OrderCheckTests(unittest.TestCase):
    def test_similarity_order_keeps_evidence_and_choices_aligned(self):
        refs = [{"paper_id": str(i), "journal_id": j, "title": title, "abstract": "Evidence " + str(i)}
                for i, (j, title) in enumerate((("b", "B lower"), ("a", "A lower"), ("a", "A closest"), ("b", "B closest")))]
        query = {"paper_id": "query", "title": "Query", "abstract": "Query content"}
        candidates, _ = shortlist(query, refs, [.2, .4, .9, .9], top_k=2)
        self.assertEqual([c["journal_id"] for c in candidates], ["a", "b"])
        self.assertEqual([[r["title"] for r in c["references"]] for c in candidates], [["A closest", "A lower"], ["B closest", "B lower"]])
        names = {"a": "Journal A", "b": "Journal B"}
        first, _ = build_request(query, list(reversed(candidates)), names, candidate_order="retrieval", seed=1)
        second, _ = build_request(query, candidates, names, candidate_order="retrieval", seed=999)
        self.assertEqual(request_digest(first), request_digest(second))
        self.assertEqual(list(first["questions"]["journal"]["criteria"]), ["a", "b"])
        self.assertLess(first["state"].index("Journal A"), first["state"].index("Journal B"))
        self.assertLess(first["state"].index("A closest"), first["state"].index("A lower"))

    def test_reversal_keeps_each_reference_with_its_journal(self):
        choices = [{"journal_id": j, "references": [{"title": j + " proof", "abstract": "distinct evidence " + j}]} for j in ("a", "b", "c")]
        original, _ = build_request({"paper_id": "q", "title": "Query", "abstract": "text"}, choices, {j: "Journal " + j for j in ("a", "b", "c")})
        reverse = reverse_request(original)
        self.assertEqual(request_digest(reverse_request(reverse)), request_digest(original))
        self.assertNotEqual(request_digest(reverse), request_digest(original))
        self.assertEqual(reverse["state"].split("\n\nCandidate journal: ")[0], original["state"].split("\n\nCandidate journal: ")[0])
        for j in ("a", "b", "c"):
            self.assertIn("Journal " + j + "\n\nReference: " + j + " proof distinct evidence " + j, reverse["state"])
        self.assertEqual(reverse_request(original, evidence=False)["state"], original["state"])

    def test_average_aligns_keys_not_positions_and_rejects_invalid(self):
        self.assertEqual(averaged_probabilities({"a": .8, "b": .2}, {"b": .6, "a": .4}), {"a": .6000000000000001, "b": .4})
        for probabilities in ({"a": float("nan")}, {"a": .2}, {"b": 1.}):
            with self.assertRaises(ValueError):
                validate_probabilities(probabilities, ["a"])

    def test_changed_shortlist_requires_explicit_comparison_mode(self):
        a = [{"paper_id": "p", "target": "d", "candidates": ["a", "b", "c"], "ranking": ["a", "b", "c"], "request_hash": "old"}]
        b = [{"paper_id": "p", "target": "d", "candidates": ["a", "b", "c", "d"], "ranking": ["d", "a", "b", "c"], "request_hash": "new"}]
        with self.assertRaises(ValueError):
            paired_top3(a, b)
        self.assertEqual(paired_top3(a, b, allow_changed_inputs=True)["top3_delta"], 1)
        self.assertEqual(metrics(b, candidate_k=20)["candidate_recall_at_20"], 1)
        self.assertNotIn("candidate_recall_at_10", metrics(b, candidate_k=20))
        b[0]["target"] = "a"
        with self.assertRaises(ValueError):
            paired_top3(a, b, allow_changed_inputs=True)
