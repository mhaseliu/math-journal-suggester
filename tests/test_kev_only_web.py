import copy
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from journal_suggester.io import journals
from journal_suggester.journal_choice import build_request, ordered_journals
from journal_suggester.web_ranker import KevOnlyRanker, WebsiteRanker, pin_gb10_inference_kernels, GB10_INFERENCE_KERNELS
from journal_suggester.web_service import SearchService


QUERY = {'title': 'Spectral gaps of graphs', 'abstract': 'We prove bounds for spectral gaps of graphs and their random walks.'}


class RecommendationTests(unittest.TestCase):
    def test_gb10_kernel_pins_bypass_tuning_and_fail_closed(self):
        modules, tuners = {}, []
        for name, function, kwargs, warps, stages in GB10_INFERENCE_KERNELS:
            selected = SimpleNamespace(kwargs=kwargs, num_warps=warps, num_stages=stages)
            other = SimpleNamespace(kwargs=kwargs, num_warps=warps + 1, num_stages=stages)
            tuner = SimpleNamespace(configs=[other, selected])
            module = modules.setdefault(name, ModuleType(name))
            setattr(module, function, tuner if function == 'l2norm_fwd_kernel' else SimpleNamespace(fn=tuner))
            tuners.append((tuner, selected, other))
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda: 'NVIDIA GB10'))
        with patch.dict('sys.modules', modules):
            pin_gb10_inference_kernels(torch)
            pin_gb10_inference_kernels(torch)
            self.assertTrue(all(tuner.configs == [selected] for tuner, selected, _ in tuners))
            tuners[-1][0].configs = [tuners[-1][2]]
            with self.assertRaisesRegex(RuntimeError, 'configuration is unavailable'):
                pin_gb10_inference_kernels(torch)
        torch.cuda.get_device_name = lambda: 'NVIDIA B300'
        with patch('importlib.import_module', side_effect=AssertionError('Must not change B300 kernels')):
            pin_gb10_inference_kernels(torch)

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

    def test_startup_and_suggestions_do_not_load_an_embedding_model(self):
        with patch.dict('sys.modules', {'journal_suggester.related_papers': None, 'journal_suggester.gpu': None}), \
                patch('journal_suggester.web_ranker.KevOnlyRanker', return_value=self.ranker):
            service = SearchService(kev_run='verified-checkpoint')
            result = service.suggest(QUERY)
        self.assertFalse(service.info()['similar_papers'])
        self.assertEqual([s['journal_id'] for s in result['suggestions']], self.ids[-1:-6:-1])
        self.assertTrue(all(not s['references'] for s in result['suggestions']))

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
