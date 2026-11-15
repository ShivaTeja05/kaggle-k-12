"""Fuse ranker v3 (the judge that reads Qwen3.8's description) into the current system, and stage a test file.

current = Sinkhorn(blend x ranker_v1^0.5 x Qwen^1, 0.5) on rows sent to Qwen (submitted top probability < 0.817), the
          file behind the 0.76266 submission (floor 0.01).
new     = Sinkhorn(blend x ranker_v1^u x Qwen^v x (v3 + eps)^w, lambda) on the same sent rows; unsent rows are unchanged.
(u, v, w, eps, lambda) are cross-fitted over 5 question folds on dev; the setting chosen on all dev rows goes to test.

Pass mark (docs/plan_to_084_2026-11-14.md, stage B): new - current >= +0.02 on dev and hard rows >= 0.68.
Never submits. Usage:
  python scripts/fuse_v3.py --dev-v3 runs/deep/q38/train_pod/v3_full/predict_pw_dev/predictions.npz --report reports/v3_fuse.json \
      [--test-v3 'runs/deep/q38/train_pod/v3_full/predict_pw_test_*/predictions.npz' --out artifacts/submission-v3]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals, sinkhorn, unseen_pairs
from k12.core import aligned_bundle, fingerprint, label_space, make_folds, normalize, read_csv, row_scores, save_bundle, topk, write_submission
from score_vlm_dev import shares

ROOT = F.ROOT
RANKER_DEV = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz'
RANKER_TEST = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_test/predictions.npz'
QWEN_DEV = 'runs/deep/q38/dev_answers_merged.jsonl'
QWEN_TEST = 'runs/deep/q38/test_answers_merged.jsonl'
CURRENT = (.01, 1., .5)  # floor, v, lambda of the staged Qwen3.8 file
GRID = [(u, v, w, eps, lam) for u in (0., .5) for v in (0., .5, 1.) for w in (.25, .5, .75, 1., 1.5, 2.) for eps in (1e-4, .01, .03)
        for lam in (.25, .5, .75)]


def bundle_rows(pattern, ids, classes):
    """v3 probabilities aligned to ids; rows without a prediction are all-ones (no information). Returns (R, has)."""
    R = np.ones((len(ids), len(classes))); has = np.zeros(len(ids), bool); at = {t: i for i, t in enumerate(ids)}
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with np.load(path, allow_pickle=True) as b:
            cols = [list(b['classes']).index(c) for c in classes]
            for t, p in zip(b['ids'], b['p']):
                if str(t) in at:
                    R[at[str(t)]] = p[cols]; has[at[str(t)]] = True
    return R, has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev-v3', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--threshold', type=float, default=.817)
    ap.add_argument('--test-v3', help='glob of v3 test bundles'); ap.add_argument('--out')
    ap.add_argument('--all-rows', action='store_true', help='use the new bundle on every row it covers, not only rows sent to Qwen '
                    '(for a model that scores all photos, such as the fine-tuned 27B classification layer)')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    pts = lambda P: row_scores(y, topk(P, classes))
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER_DEV, ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30)
    S, used = shares(ROOT / QWEN_DEV, ids, classes, first_only=True)
    R, has = bundle_rows(a.dev_v3, ids, classes)
    sent = (used > 0) & (SUB.max(1) < a.threshold)
    both = has if a.all_rows else sent & has

    def current(P0, P1, S, sent, target, iters):
        Q = np.where(sent[:, None], normalize(S + CURRENT[0]), 1.)
        return sinkhorn(normalize(P0 * P1 ** .5 * Q ** CURRENT[1]), target, CURRENT[2], iters)

    def new(P0, P1, S, R, sent, both, target, u, v, w, eps, lam, iters):
        base = current(P0, P1, S, sent, target, iters)
        Q = np.where(sent[:, None], normalize(S + CURRENT[0]), 1.); J = normalize(R + eps)
        alt = sinkhorn(normalize(P0 * P1 ** u * Q ** v * J ** w), target, lam, iters)
        return np.where(both[:, None], alt, base)

    cur = pts(current(P0, P1, S, sent, pi, 30))
    score = {k: pts(new(P0, P1, S, R, sent, both, pi, *k, 30)) for k in GRID}
    qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
    cf = cur.copy(); chosen = []
    for f in range(5):
        m = mf != f; k = GRID[int(np.argmax([score[k][m].mean() for k in GRID]))]; chosen.append(k); cf[~m] = score[k][~m]
    best = GRID[int(np.argmax([score[k].mean() for k in GRID]))]
    part = lambda s, m: {'rows': int(m.sum()), 'map3': float(s[m].mean()), 'none': float(s[m & none].mean()),
                         'seen_pair': float(s[m & seen].mean()), 'hard': float(s[m & hard].mean())}
    alone = pts(normalize(R + 1e-4)); everything = np.ones(len(ids), bool)
    diff = boot_ci(cf, cur, qid)
    rep = {'dev_rows': len(ids), 'rows_with_v3': int(has.sum()), 'rows_sent': int(sent.sum()), 'rows_changed_by_v3': int(both.sum()),
           'submitted_old_system': part(pts(SUB), everything), 'current_fused': part(cur, everything), 'new_fused_crossfit': part(cf, everything),
           'new_minus_current': diff, 'on_sent_rows': {'current': part(cur, both), 'new_crossfit': part(cf, both)},
           'v3_alone_on_its_rows': part(alone, has), 'qwen_alone_on_same_rows': part(pts(normalize(S + 1e-4)), has),
           'old_system_on_same_rows': part(pts(SUB), has),
           'fold_choices_u_v_w_eps_lam': [list(k) for k in chosen], 'chosen_all_dev': list(best), 'in_sample_chosen': float(score[best].mean()),
           'pass_mark': {'gain_at_least_0.02': bool(diff[0] >= .02), 'hard_at_least_0.68': bool(cf[hard].mean() >= .68)}}
    if a.test_v3:
        te = read_csv(ROOT / 'data/raw/test.csv'); tids = te.trace_id.tolist()
        PT, P1T = geo(F.TEST, tids, classes), aligned_bundle(ROOT / RANKER_TEST, tids, classes)
        target = marginals(tr.root_code, classes)
        ST, usedT = shares(ROOT / QWEN_TEST, tids, classes, first_only=True)
        RT, hasT = bundle_rows(a.test_v3, tids, classes)
        baseT = fuse(PT, P1T, .5, .75, target, 50)
        sentT = (usedT > 0) & (baseT.max(1) < a.threshold); bothT = hasT if a.all_rows else sentT & hasT
        curT = current(PT, P1T, ST, sentT, target, 50)
        Q = new(PT, P1T, ST, RT, sentT, bothT, target, *best, 50)
        out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
        write_submission(te, Q, classes, out / 'submission.csv')
        save_bundle(out / 'test.npz', tids, classes, Q, kind='qwen38_fused_x_ranker_v3_test')
        a0, a1 = topk(curT, classes), topk(Q, classes)
        staged = topk(aligned_bundle(ROOT / 'artifacts/submission-3way-pw14-q38/test.npz', tids, classes), classes)
        m = {'rows': len(te), 'base': 'artifacts/submission-3way-pw14-q38 (blend x ranker v1 x Qwen3.8 on 1,593 rows)',
             'rebuilt_base_matches_staged_file': bool((a0 == staged).all()),
             'ranker_v3': 'Qwen3-14B yes/no judge reading Qwen3.8 descriptions (scripts/build_ranker_v3.py, src/k12/pointwise_ranker.py)',
             'rows_sent_to_qwen': int(sentT.sum()), 'rows_with_v3': int(hasT.sum()), 'rows_changed_by_v3': int(bothT.sum()),
             'u_v_w_eps_lambda': list(best), 'development': {k: rep[k] for k in ('current_fused', 'new_fused_crossfit', 'new_minus_current', 'pass_mark')},
             'test_top1_changed_vs_staged': float((a0[:, 0] != a1[:, 0]).mean()),
             'test_none_top1_share': float((a1[:, 0] == 'NONE').mean()), 'staged_none_top1_share': float((a0[:, 0] == 'NONE').mean()),
             'script': 'scripts/fuse_v3.py', 'submission_sha256': fingerprint(out / 'submission.csv'), 'submitted': False}
        (out / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n')
        rep['staged'] = m
    (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
