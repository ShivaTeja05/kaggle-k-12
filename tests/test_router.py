import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / 'src', ROOT / 'scripts'):  # sinkhorn is compared against scripts/fuse_candidate.py
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from k12 import router

# git-ignored inputs of scripts/eval_router.py; the data tests skip when any is absent
NEEDED = ['data/raw/train.csv', 'data/raw/taxonomy_v1.csv', 'runs/deep/q38/pod_out/q38_dev_think.jsonl',
          'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz', 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
          'runs/pod/tx/text_tx_dev_e3.npz', 'runs/pod/dev-distill/validation_epoch_2.npz']
HAVE_DATA = all((ROOT / p).exists() for p in NEEDED)
K = 76


def synthetic(rows=40, seed=0):
    """A memoriser, its balanced form, reasoner shares (three distinct codes on answered rows) and a send flag."""
    rng = np.random.default_rng(seed)
    M = router.norm(rng.gamma(.3, size=(rows, K)) + 1e-9)
    target = router.norm(rng.random((1, K)) + .1)[0]
    sub = router.sinkhorn(M, target, .75, 30)
    S = np.zeros((rows, K)); answered = rng.random(rows) < .8; answered[:4] = [True, True, False, False]
    for i in np.flatnonzero(answered):
        S[i, rng.choice(K, 3, replace=False)] = (.6, .25, .15)
    sent = answered & (rng.random(rows) < .6); sent[:4] = [True, False, False, False]
    return M, sub, S, sent, answered, target, rng


def order(P):
    return np.argsort(-P, axis=1, kind='stable')


class FusionContracts(unittest.TestCase):
    def test_sinkhorn_equals_fuse_candidate(self):
        from fuse_candidate import sinkhorn
        rng = np.random.default_rng(1)
        for lam, iters in ((.5, 30), (.75, 50), (1., 3)):
            P = router.norm(rng.gamma(.5, size=(57, K)) + 1e-12); target = router.norm(rng.random((1, K)) + .05)[0]
            np.testing.assert_array_equal(router.sinkhorn(P, target, lam, iters), sinkhorn(P, target, lam, iters))

    def test_reasoner_probs_is_floored_shares(self):
        _, _, S, _, _, _, _ = synthetic()
        R = router.reasoner_probs(S, .01)
        np.testing.assert_allclose(R.sum(1), 1); self.assertEqual(R.shape, S.shape)
        np.testing.assert_allclose(R[0], (S[0] + .01) / (1 + .01 * K))
        with self.assertRaises(ValueError):
            router.reasoner_probs(S, 0.)  # an unanswered row with no floor is an empty row

    def test_product_unsent_rows_equal_the_memoriser(self):
        M, _, S, sent, _, _, rng = synthetic()
        R = router.reasoner_probs(S, .03); R[~sent] = np.nan  # whatever R holds on unsent rows must not matter
        P = router.product_fusion(M, R, sent, 1.5)
        np.testing.assert_array_equal(P[~sent], router.norm(M)[~sent])
        np.testing.assert_allclose(P[sent], router.norm(M[sent] * R[sent] ** 1.5), rtol=1e-13)
        self.assertFalse(np.allclose(P[sent], M[sent]))
        np.testing.assert_array_equal(router.product_fusion(2 * M, R, np.zeros(len(M), bool), 1.), router.norm(M))  # nothing sent; M rescaled

    def test_product_second_reasoner_factor(self):
        M, _, S, sent, _, _, rng = synthetic()
        R, R0 = router.reasoner_probs(S, .03), router.norm(rng.random(M.shape) + .1)
        np.testing.assert_array_equal(router.product_fusion(M, R, sent, 1., R0, 0.), router.product_fusion(M, R, sent, 1.))
        P = router.product_fusion(M, R, sent, 1., R0, .5)
        np.testing.assert_allclose(P[sent], router.norm(M[sent] * R0[sent] ** .5 * R[sent]), rtol=1e-13)
        np.testing.assert_array_equal(P[~sent], router.norm(M)[~sent])

    def test_soft_with_beta_zero_is_product(self):
        M, _, S, sent, _, _, rng = synthetic()
        R, g = router.reasoner_probs(S, .01), rng.random(len(M))
        for v in (.25, 1, 2):
            np.testing.assert_allclose(router.soft_fusion(M, R, sent, v, g, 0.), router.product_fusion(M, R, sent, v), rtol=1e-14, atol=0)

    def test_soft_gate_counts_only_on_sent_rows(self):
        M, _, S, sent, _, _, _ = synthetic()
        R = router.reasoner_probs(S, .01)
        P = router.soft_fusion(M, R, sent, 1., np.ones(len(M)), 1.)  # g = 1 everywhere, also where not sent
        np.testing.assert_array_equal(P[~sent], router.norm(M)[~sent])
        np.testing.assert_allclose(P[sent], R[sent], rtol=1e-13)   # beta = g = 1: the memoriser is switched off
        half = router.soft_fusion(M, R, sent, 1., np.full(len(M), .5), .5)
        np.testing.assert_allclose(half[sent], router.norm(M[sent] ** .75 * R[sent]), rtol=1e-13)
        with self.assertRaises(ValueError):
            router.soft_fusion(M, R, sent, 1., np.full(len(M), 1.5), 1.)
        loose = np.where(sent, 1., np.nan)                         # the gate is undefined where nothing was sent
        np.testing.assert_array_equal(router.soft_fusion(M, R, sent, 1., loose, 1.), P)

    def test_hard_route_sent_and_gated_rows_follow_the_reasoner(self):
        M, sub, S, sent, _, _, _ = synthetic()
        R = router.reasoner_probs(S, .01)
        P = router.hard_route(sub, R, sent, np.ones(len(M)))
        np.testing.assert_array_equal(order(P)[sent][:, :3], order(S)[sent][:, :3])   # the reasoner's three, in its order
        for i in np.flatnonzero(sent):                                                 # then the rest as sub ranks them
            rest = [k for k in order(sub[i:i + 1])[0] if S[i, k] == 0]
            self.assertEqual(order(P[i:i + 1])[0][3:].tolist(), rest)
        np.testing.assert_array_equal(P[~sent], sub[~sent])                            # g = 1 but s = 0: sub exactly
        np.testing.assert_allclose(P[sent], R[sent], rtol=2e-3)                        # R itself, up to the tie-break factor
        byM = router.hard_route(sub, R, sent, np.ones(len(M)), tie_break=M)
        i = int(np.flatnonzero(sent)[0])
        self.assertEqual(order(byM[i:i + 1])[0][3:].tolist(), [k for k in order(M[i:i + 1])[0] if S[i, k] == 0])

    def test_hard_route_everything_else_is_sub_exactly(self):
        M, sub, S, sent, _, _, _ = synthetic()
        R = router.reasoner_probs(S, .01)
        np.testing.assert_array_equal(router.hard_route(sub, R, np.zeros(len(M), bool), np.ones(len(M))), sub)  # g = 1, s = 0
        np.testing.assert_array_equal(router.hard_route(sub, R, sent, np.full(len(M), .49)), sub)               # below the threshold
        g = np.where(np.arange(len(M)) % 2 == 0, 1., 0.)
        P = router.hard_route(sub, R, sent, g); routed = sent & (g >= .5)
        np.testing.assert_array_equal(P[~routed], sub[~routed]); self.assertTrue((P[routed] != sub[routed]).any())
        before = sub.copy(); router.hard_route(sub, R, sent, g); np.testing.assert_array_equal(sub, before)     # input untouched

    def test_hard_route_threshold_is_one_half_inclusive(self):
        M, sub, S, sent, _, _, _ = synthetic()
        R = router.reasoner_probs(S, .01)
        at = router.hard_route(sub, R, sent, np.full(len(M), .5))                      # g = 1/2 routes (section 4.2.1: g >= 1/2)
        np.testing.assert_array_equal(at, router.hard_route(sub, R, sent, np.ones(len(M))))
        self.assertTrue((at[sent] != sub[sent]).any())
        loose = np.where(sent, 1., np.nan)                                             # no gate score where nothing was sent
        np.testing.assert_array_equal(router.hard_route(sub, R, sent, loose), at)
        for kw in ({'g': None, 'sub': sub}, {'g': np.ones(len(M)), 'sub': None}):      # a missing gate must not mean "route nothing"
            with self.assertRaises(ValueError):
                router.fuse(M, R, sent, 'hard', **kw)

    def test_outputs_are_row_stochastic_with_the_right_shape(self):
        M, sub, S, sent, _, target, rng = synthetic()
        R, g = router.reasoner_probs(S, .01), rng.random(len(M))
        outs = [router.product_fusion(M, R, sent, 1.), router.soft_fusion(M, R, sent, 1., g, .5), router.hard_route(sub, R, sent, g),
                router.fuse(M, R, sent, 'product', v=1., lam=.5, target=target), router.fuse(M, R, sent, 'soft', v=1., lam=.5, target=target, g=g, beta=1.),
                router.fuse(M, R, sent, 'hard', g=g, sub=sub)]
        for P in outs:
            self.assertEqual(P.shape, (len(M), K)); np.testing.assert_allclose(P.sum(1), 1, atol=1e-12)
            self.assertTrue(np.isfinite(P).all() and (P >= 0).all())
        np.testing.assert_array_equal(outs[3], router.sinkhorn(outs[0], target, .5, 30))
        np.testing.assert_array_equal(router.fuse(M, R, sent, 'product', v=1.), outs[0])  # lam = 0: no balancing
        with self.assertRaises(ValueError):
            router.fuse(M, R, sent, 'average')
        with self.assertRaises(ValueError):
            router.product_fusion(M, R, sent[:-1], 1.)


class GateContracts(unittest.TestCase):
    def test_gate_features_values(self):
        classes = ['A', 'B', 'C', 'NONE']
        sub = np.array([[.7, .1, .1, .1], [.1, .2, .1, .6], [.25, .25, .25, .25]])
        M = np.array([[1., 0., 0., 0.], [.25, .25, .25, .25], [.4, .3, .2, .1]])
        S = np.array([[.6, .25, .15, 0.], [0., .6, .15, .25], [0., 0., 0., 0.]])  # row 2 has no reasoner answer
        fit_codes = {'q1': {'A', 'NONE'}}
        Z = router.gate_features(sub, M, S, np.array([True, True, False]), np.array(['q1', 'q2', 'q1']), fit_codes, classes)
        self.assertEqual(Z.shape, (2, len(router.FEATURES))); self.assertEqual(len(router.FEATURES), 7)
        np.testing.assert_allclose(Z[0], [.7, 0., 1, 0, 1, 0, 0])               # agree on A, A seen on q1, entropy of a point mass
        np.testing.assert_allclose(Z[1], [.6, np.log(4), 0, 1, 0, 1, 0])        # B is an error code and q2 is not in the fit rows
        Z = router.gate_features(sub, M, S, np.array([1, 0]), np.array(['q1', 'q2', 'q1']), fit_codes, classes, flagged=[1, 0, 1])
        np.testing.assert_allclose(Z[:, 6], [0, 1]); np.testing.assert_allclose(Z[:, 0], [.6, .7])  # row numbers keep their order
        S[1] = [0., .25, .15, .6]                                               # the reasoner's top code is NONE on unseen q2
        Z = router.gate_features(sub, M, S, np.array([False, True, False]), np.array(['q1', 'q2', 'q1']), fit_codes, classes)
        np.testing.assert_allclose(Z[0], [.6, np.log(4), 1, 0, 0, 1, 0])        # NONE is never a "new error code"; both tops agree

    def test_gate_features_raise_for_a_row_without_a_reasoner_answer(self):
        M, sub, S, sent, answered, _, _ = synthetic()
        classes = [f'C{i:02d}' for i in range(K - 1)] + ['NONE']
        qid = np.array(['q'] * len(M))
        self.assertEqual(router.gate_features(sub, M, S, sent, qid, {}, classes).shape, (int(sent.sum()), 7))
        self.assertFalse(answered[2])
        for rows in (np.array([2]), answered | (np.arange(len(M)) == 2), np.ones(len(M), bool)):
            with self.assertRaises(ValueError):
                router.gate_features(sub, M, S, rows, qid, {}, classes)

    def test_logistic_fit_is_at_its_optimum(self):
        rng = np.random.default_rng(3)
        Z = rng.normal(size=(200, 4)) * (1, 5, .1, 1) + (0, 3, 0, -2); Z[:, 3] = 1.  # one constant feature
        t = (rng.random(200) < 1 / (1 + np.exp(-(1.5 * Z[:, 0] - .3 * (Z[:, 1] - 3))))).astype(float)
        mu, sd, theta = router.fit_logistic(Z, t, l2=2.)
        X = np.column_stack([np.ones(200), (Z - mu) / sd]); p = 1 / (1 + np.exp(-X @ theta))
        grad = X.T @ (p - t) + 2. * np.r_[0., theta[1:]]
        np.testing.assert_allclose(grad, 0, atol=1e-6)                  # stationary point of the penalised loss
        np.testing.assert_allclose(mu, Z.mean(0)); np.testing.assert_allclose(sd, [Z[:, 0].std(), Z[:, 1].std(), Z[:, 2].std(), 1.])
        self.assertGreater(theta[1], 0); self.assertLess(theta[2], 0); self.assertEqual(theta[4], 0.)
        np.testing.assert_allclose(router.predict_logistic((mu, sd, theta), Z), p)
        mu, sd, theta = router.fit_logistic(Z, np.ones(200))            # one class only: smoothed base rate, no weights
        self.assertTrue((theta[1:] == 0).all()); self.assertAlmostEqual(float(router.predict_logistic((mu, sd, theta), Z)[0]), 200.5 / 201)

    def test_folds_keep_groups_whole(self):
        rng = np.random.default_rng(4)
        groups = rng.integers(0, 23, 300).astype(str)
        fold = router.group_folds(groups, 5, 7)
        self.assertEqual(len({(a, b) for a, b in zip(groups, fold)}), len(set(groups)))
        self.assertEqual(sorted(set(fold)), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(fold, router.group_folds(groups, 5, 7))
        self.assertTrue((fold != router.group_folds(groups, 5, 8)).any())

    def test_crossfit_gate_never_trains_on_the_rows_own_group(self):
        """Group G can be separated only through a feature that is zero outside G. A gate that saw G's rows separates
        them; the cross-fitted gate cannot tell them apart, and G's labels do not move G's own scores."""
        rng = np.random.default_rng(5)
        groups = np.repeat(np.arange(10), 12); G = groups == 3
        t = (rng.random(120) < .5).astype(float); t[G] = np.tile([0., 1.], 6)
        Z = np.column_stack([rng.normal(size=120), np.zeros(120)])
        Z[G, 0] = 0.; Z[G, 1] = 2 * t[G] - 1                            # inside G: the label, and nothing else
        seen = router.predict_logistic(router.fit_logistic(Z, t, 1.), Z)
        self.assertTrue((seen[G & (t == 1)] > .5).all() and (seen[G & (t == 0)] < .5).all())  # fitted on G: separable
        out = router.crossfit_gate(Z, t, groups, n_folds=5, seed=7, l2=1.)
        self.assertTrue(np.isfinite(out).all() and (out > 0).all() and (out < 1).all()); self.assertEqual(out.shape, (120,))
        self.assertLess(np.ptp(out[G]), 1e-12)                          # cross-fitted: G's rows all look alike
        changed = t.copy(); changed[G] = 1.
        again = router.crossfit_gate(Z, changed, groups, n_folds=5, seed=7, l2=1.)
        np.testing.assert_array_equal(again[G], out[G])                 # G's labels never reach G's scores
        fold = router.group_folds(groups, 5, 7)
        self.assertTrue((again[fold != fold[G][0]] != out[fold != fold[G][0]]).any())  # they do reach the other folds
        np.testing.assert_array_equal(router.crossfit_gate(Z, t, groups, folds=fold), out)

    def test_crossfit_gate_rejects_folds_that_split_a_group(self):
        Z, t, groups = np.random.default_rng(6).normal(size=(20, 2)), np.tile([0., 1.], 10), np.repeat(np.arange(5), 4)
        with self.assertRaises(ValueError):
            router.crossfit_gate(Z, t, groups, folds=np.arange(20) % 5)
        with self.assertRaises(ValueError):
            router.crossfit_gate(Z, t, groups, folds=np.zeros(20, int))


@unittest.skipUnless(HAVE_DATA, 'git-ignored dev bundles or answers are absent')
class DevReproduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eval_router
        cls.E = eval_router; cls.D = eval_router.load_dev(); cls.prod = eval_router.product_crossfit(cls.D)

    def test_product_crossfit_reproduces_q38_fuse(self):
        D, prod = self.D, self.prod
        self.assertEqual((len(D.y), int(D.have.sum()), int(D.sent.sum())), (301, 270, 135))
        self.assertAlmostEqual(float(prod['scores'][D.have].mean()), 0.8037037, delta=1e-6)
        self.assertAlmostEqual(prod['in_sample'], 0.8141975, delta=1e-6)
        self.assertEqual(tuple(prod['best']), (.01, 1, .5))
        self.assertEqual([tuple(k) for k in prod['fold_choices']], [(.01, 1, .5), (.01, 1, .5), (.08, 1, .75), (.01, 1, .5), (.08, 1, .75)])
        self.assertAlmostEqual(float(D.sub[D.have].mean()), 0.7746914, delta=1e-6)  # submitted_on_answered in the same file

    def test_router_product_matches_the_fuse_vlm_expression(self):
        check = self.E.formula_check(self.D)
        self.assertLess(check['max_abs_probability_gap'], 1e-12)
        self.assertTrue(check['row_scores_identical_on_every_grid_setting'])

    def test_folds_are_the_fuse_vlm_folds(self):
        np.testing.assert_array_equal(router.group_folds(self.D.qid, 5, 7), self.D.mf)

    def test_unsent_rows_equal_the_memoriser_in_every_mode(self):
        D = self.D; R = router.reasoner_probs(D.S, .01); g = np.ones(len(D.y))
        for P in (router.product_fusion(D.M, R, D.sent, 1.), router.soft_fusion(D.M, R, D.sent, 1., g, 1.)):
            self.assertEqual(P.shape, (301, 76)); np.testing.assert_array_equal(P[~D.sent], router.norm(D.M)[~D.sent])
        P = router.hard_route(D.SUB, R, D.sent, g)
        self.assertEqual(P.shape, (301, 76)); np.testing.assert_array_equal(P[~D.sent], D.SUB[~D.sent])
        np.testing.assert_array_equal(order(P)[D.sent][:, :3], order(D.S)[D.sent][:, :3])

    def test_learned_gate_is_out_of_fold_by_question(self):
        """The benchmark's own gate (eval_router.learned_gate), not only router.crossfit_gate: the groups are the
        questions, no question is split, and a question's gate labels never reach that question's gate scores."""
        import copy
        D, E = self.D, self.E
        Z, label, groups, fold = E.gate_inputs(D)
        np.testing.assert_array_equal(groups, D.qid[D.sent]); np.testing.assert_array_equal(fold, D.mf[D.sent])
        self.assertEqual(len(set(zip(groups.tolist(), fold.tolist()))), len(set(groups.tolist())))
        np.testing.assert_array_equal(label, D.masks['hard'][D.sent].astype(float))
        g = E.learned_gate(D, 1.)
        self.assertTrue((g[~D.sent] == 0).all()); self.assertTrue(((g[D.sent] > 0) & (g[D.sent] < 1)).all())
        np.testing.assert_array_equal(g[D.sent], router.crossfit_gate(Z, label, groups, l2=1., folds=fold))
        counts = {q: int((D.qid[D.sent] == q).sum()) for q in set(D.qid[D.sent].tolist())}
        q = max(sorted(counts), key=counts.get); rows = D.sent & (D.qid == q); own_fold = D.sent & (D.mf == D.mf[rows][0])
        self.assertGreaterEqual(int(rows.sum()), 2)
        D2 = copy.copy(D); D2.masks = dict(D.masks); D2.masks['hard'] = np.where(rows, ~D.masks['hard'], D.masks['hard'])
        g2 = E.learned_gate(D2, 1.)
        np.testing.assert_array_equal(g2[own_fold], g[own_fold])               # its own fold is fitted without it
        self.assertTrue((g2[D.sent & ~own_fold] != g[D.sent & ~own_fold]).all())  # every other fold's model saw the change

    def test_strict_soft_fusion_never_reads_the_scored_folds_gate_labels(self):
        import copy
        D, E = self.D, self.E
        for f in (0, 3):
            held = D.mf == f
            D2 = copy.copy(D); D2.masks = dict(D.masks); D2.masks['hard'] = np.where(held, ~D.masks['hard'], D.masks['hard'])
            g = E.strict_gate(D, 1., f)
            self.assertTrue((g[~D.sent] == 0).all()); self.assertTrue(((g[D.sent] > 0) & (g[D.sent] < 1)).all())
            np.testing.assert_array_equal(E.strict_gate(D2, 1., f), g)          # fold f's labels move no score used for fold f
            other = (f + 1) % 5
            self.assertTrue((E.strict_gate(D2, 1., other)[D.sent] != E.strict_gate(D, 1., other)[D.sent]).any())  # they do reach other folds
            self.assertTrue((E.learned_gate(D2, 1.)[D.sent & ~held] != E.learned_gate(D, 1.)[D.sent & ~held]).any())  # and the loose gate
        cf, chosen = E.soft_strict(D, 1., betas=(0, 1))
        self.assertEqual(cf.shape, (301,)); self.assertEqual(len(chosen), 5); self.assertTrue(all(k[3] in (0, 1) for k in chosen))

    def test_gate_features_exist_only_for_answered_rows(self):
        D = self.D
        Z = router.gate_features(D.SUB, D.M, D.S, D.sent, D.qid, D.fit_codes, D.classes)
        self.assertEqual(Z.shape, (135, 7)); self.assertTrue(np.isfinite(Z).all())
        self.assertLess(Z[:, 0].max(), D.threshold)
        with self.assertRaises(ValueError):
            router.gate_features(D.SUB, D.M, D.S, ~D.have, D.qid, D.fit_codes, D.classes)


if __name__ == '__main__':
    unittest.main()
