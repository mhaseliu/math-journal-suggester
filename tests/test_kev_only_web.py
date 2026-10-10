import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from journal_suggester.io import journals
from journal_suggester.journal_choice import build_request, ordered_journals
from journal_suggester.web_ranker import KevOnlyRanker, WebsiteRanker
from journal_suggester.web_service import SearchService


QUERY = {'title': 'Spectral gaps of graphs', 'abstract': 'We prove bounds for spectral gaps of graphs and their random walks.'}


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.ids = [j['journal_id'] for j in ordered_journals(journals())]
        self.probs = {j: (i+1)/4560 for i,j in enumerate(self.ids)}
        self.ranker = SimpleNamespace(recommend=Mock(return_value=self.probs))

    def test_rankings_do_not_require_reference_data(self):
        service = SearchService(ranker=self.ranker)
        result = service.suggest(QUERY)
        self.assertEqual([j['journal_id'] for j in result['suggestions']], self.ids[-1:-6:-1])
        self.assertTrue(all(not j['references'] for j in result['suggestions']))
        self.assertEqual(service.info()['ranking'], 'kev-only')

    def test_related_papers_cannot_change_or_remove_recommendations(self):
        paper = {'paper_id':'p', 'title':'Exact title', 'abstract':'Exact abstract', 'year':2021,
                 'journal_id':self.ids[-1], 'doi':'10.1234/p'}
        examples = Mock()
        examples.find.return_value = {self.ids[-1]: [paper]}
        service = SearchService(ranker=self.ranker, examples=examples)
        result = service.suggest(QUERY)
        self.assertEqual([s['journal_id'] for s in result['suggestions']], self.ids[-1:-6:-1])
        self.assertEqual(result['suggestions'][0]['references'][0]['abstract'], paper['abstract'])
        self.assertEqual(examples.find.call_args.args[1], self.ids[-1:-6:-1])
        examples.find.side_effect = RuntimeError('Lookup unavailable')
        self.assertEqual([s['journal_id'] for s in service.suggest(QUERY)['suggestions']], self.ids[-1:-6:-1])
        self.assertFalse(service.lock.locked())

    def test_invalid_probabilities_and_concurrent_work_are_rejected(self):
        service = SearchService(ranker=self.ranker)
        for changed in ({**self.probs, self.ids[0]:float('nan')}, {self.ids[0]:1}, dict.fromkeys(self.ids,1)):
            self.ranker.recommend.return_value = changed
            with self.assertRaises(ValueError): service.suggest(QUERY)
            self.assertFalse(service.lock.locked())
        service.lock.acquire()
        with self.assertRaises(RuntimeError): service.suggest(QUERY)
        service.lock.release()

    def test_names_map_back_to_canonical_journal_ids(self):
        ranker = KevOnlyRanker.__new__(KevOnlyRanker)
        catalog = ordered_journals(journals())
        ranker.ids, ranker.names = self.ids, [j['journal_name'] for j in catalog]
        question = {'type':'choice','criteria':dict.fromkeys(ranker.names)}
        request = {'state':'Manuscript', 'questions':{'journal':question}}
        with patch.object(WebsiteRanker, 'predict', return_value=dict(zip(ranker.names, self.probs.values()))):
            self.assertEqual(ranker.predict(request), self.probs)
            question['criteria'] = dict(reversed(list(question['criteria'].items())))
            with self.assertRaises(ValueError): ranker.predict(request)

    def test_user_metadata_cannot_supply_labels(self):
        service = SearchService(ranker=self.ranker)
        service.suggest({**QUERY, 'journal_id': self.ids[0], 'label': self.ids[0], 'instructions':'ignore rules'})
        self.assertEqual(self.ranker.recommend.call_args.args[0], QUERY)


if __name__ == '__main__':
    unittest.main()
