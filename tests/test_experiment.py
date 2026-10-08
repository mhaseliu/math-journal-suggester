import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_suggester.experiment import check_partitions, load_selection, select, select_queries, verify_results
from journal_suggester.io import digest, read_json, write_json, write_jsonl


def paper(i, journal='a'):
    return {'paper_id':f'paper-{i}','group_id':f'group-{i}','journal_id':journal,
        'title':f'Distinct topic {i}', 'abstract':f'We prove theorem number {i} about a specific family of mathematical objects using explicit estimates and a constructive argument.',
        'doi':f'10.1234/work-{i}','arxiv_id':'','year':2024,'url':''}


class ExperimentTests(unittest.TestCase):
    def test_selection_is_order_independent_proportional_and_caps_shortages(self):
        refs=[paper(i,'a' if i<2 else 'b') for i in range(12)]
        a,quotas=select_queries(refs,{'a':90,'b':10},8,42)
        b,_=select_queries(list(reversed(refs)),{'b':10,'a':90},8,42)
        self.assertEqual(a,b)
        self.assertEqual(quotas,{'a':2,'b':6})

    def test_cross_split_version_leakage_rejected(self):
        parts={'reference':[paper(1)],'validation':[paper(2)],'test':[paper(3)]}
        check_partitions(parts)
        parts['test'][0]['doi']=parts['reference'][0]['doi']
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            check_partitions(parts)

    def test_standalone_selection_needs_no_prior_runs_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source'
            parts={'reference':[paper(i,'a' if i<4 else 'b') for i in range(8)],
                   'validation':[paper(10),paper(11,'b')],'test':[paper(12),paper(13,'b')]}
            for name,rows in parts.items():write_jsonl(source/f'{name}.jsonl',rows)
            write_json(source/'manifest.json',{'fingerprint':'fixture','partition_hashes':{k:digest(v) for k,v in parts.items()}})
            write_json(root/'counts.json',{'a':1,'b':1})
            write_json(root/'config.json',{'training_papers':6,'validation_papers':2,'seed':42})
            write_json(root/'models.json',{'embedding_model':'fixture'})
            out=root/'new'
            select(source,out,root/'counts.json',root/'config.json',root/'models.json')
            saved,loaded=load_selection(out)
            self.assertEqual(len(loaded['train']),6)
            with self.assertRaisesRegex(ValueError,'new experiment'):
                select(source,out,root/'counts.json',root/'config.json',root/'models.json')
            loaded['train'][0]['abstract']='Changed content'
            write_jsonl(out/'splits/train.jsonl',loaded['train'])
            with self.assertRaisesRegex(ValueError,'data changed'):load_selection(out)

    def test_published_results_recompute_without_models(self):
        directory=Path(__file__).resolve().parents[1]/'results'
        result=verify_results(directory)
        self.assertTrue(result['checks_pass'])
        self.assertEqual(result['model_calls'],0)

    def test_public_headline_tampering_detected(self):
        import journal_suggester.experiment as experiment
        original=experiment.read_json
        def altered(path):
            result=original(path)
            if Path(path).name=='test.json':result['metrics']['fine_tuned_kev']['top3']=0.99
            return result
        directory=Path(__file__).resolve().parents[1]/'results'
        with patch.object(experiment,'read_json',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'Result mismatch'):verify_results(directory)
