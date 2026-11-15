"""The submitted 0.73183 file can be rebuilt byte for byte from the saved test prediction bundles.

This pins the memoriser: every later change (router, shortlist, new fusion modes) must leave this file unchanged
unless it is switched on. The rebuild follows the staging branch of scripts/fuse_candidate.py:

    fuse(geo(TEST), ranker_test, w=0.5, lam=0.75, target=marginals(all 1,500 train root codes), iters=50)

Only saved prediction bundles and the ids in test.csv are read. No test photo is opened and nothing is submitted.
The bundles are git-ignored, so the tests skip when they are absent. Two negative controls (30 iterations; fit-row
marginals) must NOT reproduce the file, so the check cannot pass vacuously.
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT / 'scripts'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

SUBMITTED_SHA256 = '5354bd618564bddd73b9cd7a82c3ae27608fb9543929c5099e6045f05452570f'
SUBMITTED_CSV = ROOT / 'artifacts/submission-3way-pw14/submission.csv'
RANKER_TEST = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_test/predictions.npz'
BLEND_TEST = ('runs/pod/text32-full/test.npz', 'runs/pod/text-full/test.npz', 'runs/pod/full-distill/test.npz')
NEEDED = [ROOT / p for p in BLEND_TEST + (RANKER_TEST, 'data/raw/train.csv', 'data/raw/test.csv', 'data/raw/taxonomy_v1.csv')]
MISSING = [str(p.relative_to(ROOT)) for p in NEEDED if not p.exists()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipIf(MISSING, 'git-ignored inputs are missing: ' + ', '.join(MISSING))
class ReproduceSubmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fuse_candidate as F
        from k12.core import aligned_bundle, label_space, make_folds, read_csv
        cls.F = F
        cls.train = read_csv(ROOT / 'data/raw/train.csv'); cls.test = read_csv(ROOT / 'data/raw/test.csv')
        cls.classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
        cls.fit = cls.train.loc[make_folds(cls.train) != 0]
        ids = cls.test.trace_id.tolist()
        cls.blend = F.geo(F.TEST, ids, cls.classes)
        cls.ranker = aligned_bundle(ROOT / RANKER_TEST, ids, cls.classes)

    def rebuilt_sha256(self, codes, iters, w=.5, lam=.75):
        """Hash of the CSV that write_submission produces for one setting (written to a temporary directory)."""
        from k12.core import write_submission
        Q = self.F.fuse(self.blend, self.ranker, w, lam, self.F.marginals(codes, self.classes), iters)
        with tempfile.TemporaryDirectory() as d:
            write_submission(self.test, Q, self.classes, Path(d) / 'submission.csv')
            return sha256(Path(d) / 'submission.csv')

    def test_inputs_are_the_ones_the_staging_code_names(self):
        self.assertEqual(tuple(p for p, _ in self.F.TEST), BLEND_TEST)
        self.assertEqual([w for _, w in self.F.TEST], [.35, .30, .35])
        self.assertEqual(len(self.train), 1500); self.assertEqual(len(self.fit), 1199)
        self.assertEqual(self.blend.shape, (len(self.test), 76)); self.assertEqual(self.ranker.shape, self.blend.shape)

    def test_rebuilt_file_is_byte_identical_to_the_submitted_one(self):
        self.assertEqual(self.rebuilt_sha256(self.train.root_code, 50), SUBMITTED_SHA256)

    def test_staged_file_on_disk_is_the_submitted_one(self):
        if not SUBMITTED_CSV.exists():
            self.skipTest('artifacts/submission-3way-pw14/submission.csv is not on disk')
        self.assertEqual(sha256(SUBMITTED_CSV), SUBMITTED_SHA256)

    def test_negative_control_30_iterations_do_not_match(self):
        self.assertNotEqual(self.rebuilt_sha256(self.train.root_code, 30), SUBMITTED_SHA256)

    def test_negative_control_fit_row_marginals_do_not_match(self):
        self.assertNotEqual(self.rebuilt_sha256(self.fit.root_code, 50), SUBMITTED_SHA256)


if __name__ == '__main__':
    unittest.main()
