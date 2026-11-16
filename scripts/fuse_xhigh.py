"""Add Qwen3.8-27B maximum-thinking (reasoning_effort xhigh) answers to the current system, and stage a test file.

current = Sinkhorn(blend x ranker_v1^0.5 x Qm, 0.5): Qm = medium-thinking rank shares + floor 0.01 on rows sent to Qwen
          (submitted top probability < 0.817); the file behind the 0.76383 submission.
new     = the same, with Qm x Qx on rows that also have a maximum-thinking answer (Qx = xhigh rank shares + floor 0.01).
Nothing is tuned: floor, weights and lambda are the staged setting, so the dev figure is not fitted to dev.
Never submits. Usage:
  python scripts/fuse_xhigh.py --report reports/q38_xhigh_fuse.json \
      [--test-xhigh 'runs/deep/q38/xhigh_pod/xh_*.jsonl' --out artifacts/submission-3way-pw14-q38-xhigh]
"""
import argparse
import glob
import json

import numpy as np

import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals, sinkhorn, unseen_pairs
from fuse_v3 import QWEN_DEV, QWEN_TEST, RANKER_DEV, RANKER_TEST
from k12.core import aligned_bundle, fingerprint, label_space, make_folds, normalize, read_csv, row_scores, save_bundle, topk, write_submission
from score_vlm_dev import shares

ROOT = F.ROOT
XHIGH_DEV = 'runs/hfjobs/out/base/xhigh_q38_dev_sent158.jsonl'
FLOOR, LAM = .01, .5


def xhigh_shares(pattern, ids, classes):
    S = np.zeros((len(ids), len(classes))); used = np.zeros(len(ids), int)
    for path in sorted(glob.glob(str(ROOT / pattern))):
        s, u = shares(path, ids, classes, first_only=True)
        S[u > 0] = s[u > 0]; used[u > 0] = 1
    return S, used > 0


def build(P0, P1, S, X, sent, hasx, target, iters):
    Q = np.where(sent[:, None], normalize(S + FLOOR), 1.)
    cur = sinkhorn(normalize(P0 * P1 ** .5 * Q), target, LAM, iters)
    Q2 = Q * np.where((sent & hasx)[:, None], normalize(X + FLOOR), 1.)
    return cur, sinkhorn(normalize(P0 * P1 ** .5 * Q2), target, LAM, iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True); ap.add_argument('--threshold', type=float, default=.817)
    ap.add_argument('--test-xhigh'); ap.add_argument('--out')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    pts = lambda P: row_scores(y, topk(P, classes))
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER_DEV, ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30)
    S, used = shares(ROOT / QWEN_DEV, ids, classes, first_only=True)
    X, hasx = xhigh_shares(XHIGH_DEV, ids, classes)
    sent = (used > 0) & (SUB.max(1) < a.threshold)
    cur, new = (pts(P) for P in build(P0, P1, S, X, sent, hasx, pi, 30))
    both = sent & hasx
    part = lambda s, m: {'rows': int(m.sum()), 'map3': float(s[m].mean())}
    # the test run covers only the least-certain rows; the same slice of dev is the fair comparison
    order = np.argsort(SUB.max(1)); least = np.zeros(len(ids), bool); least[order[:max(1, int(round(len(ids) * .025)))]] = True
    rep = {'dev_rows': len(ids), 'rows_sent': int(sent.sum()), 'rows_with_xhigh': int(both.sum()),
           'current_fused': float(cur.mean()), 'with_xhigh': float(new.mean()), 'gain_ci': boot_ci(new, cur, qid),
           'hard': {'current': float(cur[hard].mean()), 'with_xhigh': float(new[hard].mean())},
           'none': {'current': float(cur[none].mean()), 'with_xhigh': float(new[none].mean())},
           'on_xhigh_rows': {'current': part(cur, both), 'with_xhigh': part(new, both)},
           'least_certain_2.5pct_of_dev': {'current': part(cur, least & both), 'with_xhigh': part(new, least & both)}}
    if a.test_xhigh:
        te = read_csv(ROOT / 'data/raw/test.csv'); tids = te.trace_id.tolist()
        PT, P1T = geo(F.TEST, tids, classes), aligned_bundle(ROOT / RANKER_TEST, tids, classes)
        target = marginals(tr.root_code, classes)
        ST, usedT = shares(ROOT / QWEN_TEST, tids, classes, first_only=True)
        XT, hasT = xhigh_shares(a.test_xhigh, tids, classes)
        baseT = fuse(PT, P1T, .5, .75, target, 50); sentT = (usedT > 0) & (baseT.max(1) < a.threshold)
        curT, newT = build(PT, P1T, ST, XT, sentT, hasT, target, 50)
        a0, a1 = topk(curT, classes), topk(newT, classes)
        staged = topk(aligned_bundle(ROOT / 'artifacts/submission-3way-pw14-q38/test.npz', tids, classes), classes)
        m_idx = np.array([classes.index(c) for c in a0[:, 0]]); x_top = XT.argmax(1); bothT = sentT & hasT
        out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
        write_submission(te, newT, classes, out / 'submission.csv')
        save_bundle(out / 'test.npz', tids, classes, newT, kind='qwen38_medium_x_xhigh_test')
        m = {'rows': len(te), 'base': 'artifacts/submission-3way-pw14-q38 (public 0.76383)',
             'rebuilt_base_matches_staged_file': bool((a0 == staged).all()),
             'xhigh': 'Qwen/Qwen3.8-27B-FP8 zero-shot, reasoning_effort xhigh, least-certain test rows first, own pod',
             'rows_with_xhigh': int(bothT.sum()), 'xhigh_first_code_equals_current_top1': float((x_top[bothT] == m_idx[bothT]).mean()),
             'rows_top1_changed': int((a0[:, 0] != a1[:, 0]).sum()), 'rows_top3_changed': int((a0 != a1).any(1).sum()),
             'floor': FLOOR, 'lambda': LAM, 'development': {k: rep[k] for k in ('current_fused', 'with_xhigh', 'gain_ci')},
             'script': 'scripts/fuse_xhigh.py', 'submission_sha256': fingerprint(out / 'submission.csv'), 'submitted': False}
        (out / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n'); rep['staged'] = m
    (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
