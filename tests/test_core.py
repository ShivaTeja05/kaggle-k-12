import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from k12.core import row_scores, save_bundle, aligned_bundle, make_folds, write_submission


class CompetitionContracts(unittest.TestCase):
    def test_official_ranking_examples_and_duplicates(self):
        np.testing.assert_allclose(row_scores(['ALG-06']*6,
            ['ALG-06 CAL-01 NONE', 'NONE ALG-06 CAL-01', 'NONE CAL-01 ALG-06',
             'NONE CAL-01 WOP-01', 'ALG-06 ALG-06 ALG-06', 'NONE NONE ALG-06']),
            [1,.5,1/3,0,1,1/3])

    def test_id_and_class_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.npz';save_bundle(p,['b','a'],['NONE','ALG-06'],[[.8,.2],[.1,.9]])
            np.testing.assert_allclose(aligned_bundle(p,['a','b'],['ALG-06','NONE']),[[.9,.1],[.2,.8]],atol=1e-6)
            with self.assertRaises(ValueError): aligned_bundle(p,['a','c'],['NONE','ALG-06'])

    def test_duplicates_stay_together_and_singletons_are_evaluated(self):
        df=pd.DataFrame({'question_id':['a','a','b','c','d','e'],
                         'transcription_steps':['L1 | 2','L1 | 2','3','4','5','6'],
                         'root_code':['NONE','NONE','A','A','B','C']})
        f=make_folds(df,3); self.assertEqual(f[0],f[1]);self.assertTrue((f>=0).all())
        np.testing.assert_array_equal(f,make_folds(df,3))

    def test_submission_has_unique_valid_rankings(self):
        with tempfile.TemporaryDirectory() as d:
            out=write_submission(pd.DataFrame({'trace_id':['x']}),[[.1,.7,.2]],['A','NONE','B'],Path(d)/'submission.csv')
            self.assertEqual(out.root_code[0],'NONE B A')
            with self.assertRaises(ValueError): write_submission(pd.DataFrame({'trace_id':['x']}),[[np.nan,1]],['A','NONE'],Path(d)/'bad.csv')

    def test_classifier_prompt_cannot_read_training_annotations(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from k12.vision import messages
        row=SimpleNamespace(question_image_file='diagram.png',image_file='work.jpeg',grade=6,
            topic_name='Fractions',question_text='1/2 + 1/3',correct_answer='5/6',
            transcription_steps='SECRET_GOLD_TRANSCRIPT',root_code='SECRET_LABEL',
            misconceptions='SECRET_ANNOTATION',clean='SECRET_CLEAN',final_answer_correct='SECRET_CORRECT')
        with patch('k12.vision.image',return_value='IMAGE') as loader:
            m=messages(row,Path('/data'),'train')
        self.assertNotIn('SECRET',str(m));self.assertEqual(loader.call_count,2)

    def test_generated_labels_keep_official_case_and_fill_valid_unique_slots(self):
        from k12.taxonomy_ranker import parse_codes
        result=parse_codes('<think>ALG-06</think> dec-01 NONE dec-01 JUNK-99',
                           ['Dec-01','NONE','ALG-06'],['NONE','ALG-06','Dec-01'])
        self.assertEqual(result,['Dec-01','NONE','ALG-06'])

    def test_rejected_alternatives_are_not_ranked_above_final_codes(self):
        from k12.taxonomy_ranker import parse_codes
        result=parse_codes('CAL-01 is a downstream slip.\nFINAL: ALG-06 NONE CAL-01',
                           ['ALG-06','NONE','CAL-01'],[])
        self.assertEqual(result,['ALG-06','NONE','CAL-01'])

    def test_kaggle_exit_zero_quota_error_is_not_a_launched_job(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('launch',Path(__file__).parents[1]/'scripts/launch_kaggle.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertFalse(m.successful_push(0,'Kernel push error: Maximum batch GPU session count of 2 reached.'))
        self.assertTrue(m.successful_push(0,'Kernel version 2 successfully pushed.'))

    def test_gpu_budget_reserves_running_jobs_and_safety_margin(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('launch',Path(__file__).parents[1]/'scripts/launch_kaggle.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        active=[{'timeout_seconds':21600,'observed_seconds':1200},{'timeout_seconds':7200,'observed_seconds':1000}]
        self.assertGreater(m.projected_hours(1.21,active,7200),10)
        self.assertLess(m.projected_hours(1.21,active,1800),10)

    def test_gpu_budget_rejects_jobs_that_would_spend_quota_reserve(self):
        import importlib.util
        import json
        from subprocess import CompletedProcess
        from unittest.mock import patch

        spec = importlib.util.spec_from_file_location('launch', Path(__file__).parents[1] / 'scripts/launch_kaggle.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        quota = [{'resource': 'GPU', 'used': '24.00h', 'remaining': '6.00h', 'refreshAt': 'window'}]
        policy = {'quota_refresh_at': 'window', 'hard_cap_total_used_hours': 30,
                  'minimum_remaining_hours': 4, 'per_session_overhead_hours': .1,
                  'safety_margin_hours': .25,
                  'reservations': [{'kernel': 'owner/active', 'timeout_seconds': 3600}]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            budget = folder / 'budget.json'
            budget.write_text(json.dumps(policy))
            responses = [CompletedProcess([], 0, json.dumps(quota)),
                         CompletedProcess([], 0, 'KernelWorkerStatus.RUNNING')]
            with patch.object(launcher, 'account_active_refs', return_value=['owner/active']), \
                 patch.object(launcher.subprocess, 'run', side_effect=responses), \
                 patch('builtins.print'):
                with self.assertRaisesRegex(SystemExit, 'less than 4 hours'):
                    launcher.budget_check(folder, 3600, budget)
            report = json.loads((folder / 'budget_check.json').read_text())
            self.assertFalse(report['admitted'])
            self.assertAlmostEqual(report['projected_worst_case_remaining_hours'], 3.55)

    def test_budget_tracks_kaggles_canonical_name_after_title_slug_change(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('launch',Path(__file__).parents[1]/'scripts/launch_kaggle.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        actual=m.pushed_kernel_ref('Kernel version 1 successfully pushed. https://www.kaggle.com/code/owner/actual-slug','owner/requested-slug')
        self.assertEqual(actual,'owner/actual-slug')

    def test_budget_discovers_recent_jobs_from_other_projects(self):
        import importlib.util
        from datetime import datetime,timezone
        spec=importlib.util.spec_from_file_location('launch',Path(__file__).parents[1]/'scripts/launch_kaggle.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        rows=[{'ref':'owner/other-project','lastRunTime':'2026-11-08T15:59:59.933000'},
              {'ref':'owner/old','lastRunTime':'2026-10-08T15:59:59Z'}]
        refs=m.recent_account_refs(rows,datetime(2026,11,8,17,tzinfo=timezone.utc))
        self.assertEqual(refs,['owner/other-project'])
        with self.assertRaises(RuntimeError):m.recent_account_refs([rows[0]]*200,datetime(2026,11,8,17,tzinfo=timezone.utc))

    def test_demonstrations_use_fitting_annotations_only(self):
        from types import SimpleNamespace
        from k12.taxonomy_ranker import labelled_examples
        fitting=pd.DataFrame({'question_id':['q'],'transcription_steps':['TRAIN_WORK'],
            'root_code':['CAL-01'],'misconceptions':['[{"root":true,"text":"TRAIN_REASON"}]']})
        target=SimpleNamespace(question_id='q',transcription_steps='HELD_OUT_WORK',misconceptions='HELD_OUT_REASON')
        examples=labelled_examples(target,fitting)
        self.assertIn('TRAIN_REASON',examples);self.assertNotIn('HELD_OUT',examples)


if __name__=='__main__': unittest.main()
