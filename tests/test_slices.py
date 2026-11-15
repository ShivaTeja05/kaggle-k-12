import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT / 'scripts'):  # the data tests compare against scripts/fuse_candidate.py
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from k12.slices import HEADER, SLICES, format_report, print_reports, slice_masks, slice_report

# git-ignored inputs of scripts/eval_router.py; the data tests skip when any is absent
NEEDED = ['data/raw/train.csv', 'data/raw/taxonomy_v1.csv', 'runs/deep/q38/pod_out/q38_dev_think.jsonl',
          'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz', 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
          'runs/pod/tx/text_tx_dev_e3.npz', 'runs/pod/dev-distill/validation_epoch_2.npz']
HAVE_DATA = all((ROOT / p).exists() for p in NEEDED)


def hand_example():
    """Fit rows: q1 shows A and NONE, q2 shows B. Six scored rows with one known slice each."""
    fit = pd.DataFrame({'question_id': ['q1', 'q1', 'q1', 'q2'], 'root_code': ['A', 'NONE', 'A', 'B']})
    y = np.array(['NONE', 'A', 'B', 'B', 'C', 'NONE'])
    qid = np.array(['q1', 'q1', 'q1', 'q2', 'q9', 'q9'])
    return fit, y, qid


class SliceContracts(unittest.TestCase):
    def test_hand_example_has_the_known_slices(self):
        m = slice_masks(*hand_example())
        np.testing.assert_array_equal(m['none'], [1, 0, 0, 0, 0, 1])          # NONE rows, seen question or not
        np.testing.assert_array_equal(m['seen_pair'], [0, 1, 0, 1, 0, 0])     # (q1, A) and (q2, B) are in the fit rows
        np.testing.assert_array_equal(m['hard'], [0, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(m['hard_new_code'], [0, 0, 1, 0, 0, 0])  # B is new for q1, though q2 has it
        np.testing.assert_array_equal(m['hard_unseen_question'], [0, 0, 0, 0, 1, 0])
        self.assertTrue(all(v.dtype == bool for v in m.values()))

    def test_masks_are_disjoint_and_cover_every_row(self):
        rng = np.random.default_rng(0)
        codes = np.array(['NONE', 'A', 'B', 'C', 'D'])
        for trial in range(20):
            fit = pd.DataFrame({'question_id': rng.integers(0, 12, 60).astype(str), 'root_code': rng.choice(codes, 60)})
            y, qid = rng.choice(codes, 80), rng.integers(0, 16, 80).astype(str)
            m = slice_masks(fit, y, qid)
            total = sum(m[k].astype(int) for k in SLICES)
            np.testing.assert_array_equal(total, np.ones(80, int))
            np.testing.assert_array_equal(m['hard'], m['hard_new_code'] | m['hard_unseen_question'])
            self.assertFalse((m['hard_new_code'] & m['hard_unseen_question']).any())
            np.testing.assert_array_equal(m['none'], y == 'NONE')

    def test_a_label_in_the_fit_rows_of_another_question_is_still_hard(self):
        fit = pd.DataFrame({'question_id': ['q1', 'q2'], 'root_code': ['A', 'B']})
        m = slice_masks(fit, np.array(['B']), np.array(['q1']))
        self.assertTrue(m['hard'][0] and m['hard_new_code'][0])

    def test_report_means_counts_and_row_subset(self):
        m = slice_masks(*hand_example())
        scores = np.array([1., .5, 0., 1., 1 / 3, 0.])
        r = slice_report(scores, m)
        self.assertEqual((r['rows'], r['n_none'], r['n_seen_pair'], r['n_hard']), (6, 2, 2, 2))
        self.assertAlmostEqual(r['map3'], scores.mean()); self.assertAlmostEqual(r['none'], .5)
        self.assertAlmostEqual(r['seen_pair'], .75); self.assertAlmostEqual(r['hard'], 1 / 6)
        sub = slice_report(scores, m, np.array([1, 1, 0, 0, 0, 0], bool))
        self.assertEqual((sub['rows'], sub['n_none'], sub['n_seen_pair'], sub['n_hard']), (2, 1, 1, 0))
        self.assertAlmostEqual(sub['map3'], .75); self.assertIsNone(sub['hard'])  # an empty slice is not a score
        with self.assertRaises(ValueError):
            slice_report(scores[:5], m)

    def test_text_lines_are_aligned_and_the_header_prints_once(self):
        m = slice_masks(*hand_example())
        r = slice_report(np.array([1., .5, 0., 1., 1 / 3, 0.]), m)
        line = format_report('submitted', r)
        self.assertEqual(len(line), len(HEADER)); self.assertNotIn('\n', line)
        self.assertIn('0.4722', line); self.assertIn('(  2)', line)
        self.assertIn('n/a', format_report('subset', slice_report(np.ones(6), m, np.array([1, 1, 0, 0, 0, 0], bool))))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_reports('Title', [('one', r), ('two', r)], {'two': '+0.0100'})
        text = out.getvalue()
        self.assertEqual(text.count(HEADER), 1); self.assertEqual(len(text.strip().splitlines()), 4)
        self.assertTrue(text.strip().splitlines()[-1].endswith('+0.0100'))


@unittest.skipUnless(HAVE_DATA, 'git-ignored dev bundles or answers are absent')
class DevSlices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eval_router
        cls.D = eval_router.load_dev()

    def test_dev_slice_counts(self):
        D = self.D
        count = lambda rows: tuple(int((D.masks[k] & rows).sum()) for k in SLICES)
        self.assertEqual(len(D.y), 301); self.assertEqual(int(D.have.sum()), 270)
        self.assertEqual(count(np.ones(301, bool)), (101, 94, 106))
        self.assertEqual(count(D.have), (101, 86, 83))
        r = slice_report(D.sub, D.masks, D.have)
        self.assertEqual((r['rows'], r['n_none'], r['n_seen_pair'], r['n_hard']), (270, 101, 86, 83))

    def test_masks_identical_to_unseen_pairs(self):
        from fuse_candidate import unseen_pairs
        D = self.D
        newc, unq = unseen_pairs(D.fit, D.y, D.qid)
        np.testing.assert_array_equal(D.masks['hard_new_code'], newc)
        np.testing.assert_array_equal(D.masks['hard_unseen_question'], unq)
        hard, none = newc | unq, D.y == 'NONE'  # the three masks as scripts/fuse_vlm.py builds them
        np.testing.assert_array_equal(D.masks['hard'], hard)
        np.testing.assert_array_equal(D.masks['none'], none)
        np.testing.assert_array_equal(D.masks['seen_pair'], ~none & ~hard)


if __name__ == '__main__':
    unittest.main()
