"""Fuse a zero-shot vision model's ranked codes into the submitted system, on the rows where that system is least sure.

fused = Sinkhorn(blend x ranker^0.5 x model^v, lambda), where "model" is 1 (no information) on rows that were not sent
to it. Rows are sent when the submitted system's top probability is below --threshold. (floor, v, lambda) are
cross-fitted over 5 question folds on dev; the setting chosen on all dev rows is applied to test.

Never submits. Usage:
  python scripts/fuse_vlm.py --dev runs/deep/q38/pod_out/q38_dev_think.jsonl --report reports/q38_fuse.json \
      [--test 'runs/deep/q38/test_pod_*/q38_test_part_*.jsonl' --out artifacts/submission-3way-pw14-q38]
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
SUBMITTED = 'artifacts/submission-3way-pw14'
GRID = [(floor, v, lam) for floor in (.01, .03, .08) for v in (.25, .5, .75, 1, 1.5, 2) for lam in F.LAMS]


def merged_shares(pattern, ids, classes, scratch):
    """Rank shares from every file matching the pattern (one row per trace_id; the first answer wins)."""
    seen, lines = set(), []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        for line in Path(path).read_text().splitlines():
            trace = json.loads(line)['trace_id']
            if trace not in seen:
                seen.add(trace); lines.append(line)
    scratch.write_text('\n'.join(lines) + '\n')
    return shares(scratch, ids, classes, first_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dev', required=True); ap.add_argument('--threshold', type=float, default=.817)
    ap.add_argument('--report', required=True)
    ap.add_argument('--test', help='glob of test answer files'); ap.add_argument('--out')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    pts = lambda P: row_scores(y, topk(P, classes))
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER_DEV, ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30); sub = pts(SUB)
    S, used = shares(ROOT / a.dev, ids, classes, first_only=True)
    have = used > 0; sent = have & (SUB.max(1) < a.threshold)

    def fused(P0, P1, S, sent, target, floor, v, lam, iters):
        Q = np.where(sent[:, None], normalize(S + floor), 1.)
        return sinkhorn(normalize(P0 * P1 ** .5 * Q ** v), target, lam, iters) if lam else normalize(P0 * P1 ** .5 * Q ** v)

    cache = {k: fused(P0, P1, S, sent, pi, *k, 30) for k in GRID}
    score = {k: pts(v) for k, v in cache.items()}
    qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
    cf = sub.copy(); chosen = []
    for f in range(5):
        m = mf != f; k = GRID[int(np.argmax([score[k][m & have].mean() for k in GRID]))]; chosen.append(k); cf[~m] = score[k][~m]
    best = GRID[int(np.argmax([score[k][have].mean() for k in GRID]))]
    part = lambda s, m: {'rows': int(m.sum()), 'map3': float(s[m].mean()), 'none': float(s[m & none].mean()),
                         'seen_pair': float(s[m & seen].mean()), 'hard': float(s[m & hard].mean())}
    rep = {'threshold': a.threshold, 'dev_rows_answered': int(have.sum()), 'dev_rows_sent': int(sent.sum()),
           'dev_share_sent_of_answered': float(sent.sum() / have.sum()),
           'submitted_on_answered': part(sub, have), 'fused_crossfit_on_answered': part(cf, have),
           'fused_minus_submitted': boot_ci(cf[have], sub[have], qid[have]),
           'fold_choices_floor_v_lam': [list(k) for k in chosen], 'chosen_floor_v_lam_all_dev': list(best),
           'in_sample_chosen': float(score[best][have].mean()),
           'on_sent_rows': {'submitted': float(sub[sent].mean()), 'fused_crossfit': float(cf[sent].mean())},
           'on_unsent_answered_rows': {'submitted': float(sub[have & ~sent].mean()), 'fused_crossfit': float(cf[have & ~sent].mean())}}
    if a.test:
        te = read_csv(ROOT / 'data/raw/test.csv'); tids = te.trace_id.tolist()
        PT, P1T = geo(F.TEST, tids, classes), aligned_bundle(ROOT / RANKER_TEST, tids, classes)
        target = marginals(tr.root_code, classes)
        ST, usedT = merged_shares(a.test, tids, classes, ROOT / 'runs/deep/q38/test_answers_merged.jsonl')
        base = fuse(PT, P1T, .5, .75, target, 50)
        with np.load(ROOT / SUBMITTED / 'test.npz', allow_pickle=False) as b:
            same = bool((topk(base, classes) == topk(aligned_bundle(ROOT / SUBMITTED / 'test.npz', tids, classes), classes)).all())
        sentT = (usedT > 0) & (base.max(1) < a.threshold)
        Q = fused(PT, P1T, ST, sentT, target, *best, 50)
        out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
        write_submission(te, Q, classes, out / 'submission.csv')
        save_bundle(out / 'test.npz', tids, classes, Q, kind='submitted_x_vlm_sinkhorn_test', floor=best[0], v=best[1], lam=best[2])
        a0, a1 = topk(base, classes), topk(Q, classes)
        selected = json.loads((ROOT / 'data/v9/q38_test/ids_least_confident.json').read_text())
        m = {'rows': len(te), 'base': 'the submitted system (artifacts/submission-3way-pw14): 3-way blend x ranker v1^0.5',
             'rebuilt_base_matches_submitted_file': same,
             'vlm': 'Qwen/Qwen3.8-27B-FP8 zero-shot, thinking (effort medium), vLLM 0.33.0, offline on our own pods',
             'vlm_rows_selected': len(selected), 'vlm_rows_answered': int((usedT > 0).sum()), 'vlm_rows_used': int(sentT.sum()),
             'threshold': a.threshold, 'floor': best[0], 'v': best[1], 'lambda': best[2],
             'post_processing': 'Sinkhorn toward the 1,500-row train root_code marginals (+0.5), 50 iterations, label-free on test',
             'development': {k: rep[k] for k in ('dev_rows_answered', 'submitted_on_answered', 'fused_crossfit_on_answered', 'fused_minus_submitted')},
             'test_top1_changed_vs_submitted': float((a0[:, 0] != a1[:, 0]).mean()),
             'test_top3_set_changed_vs_submitted': float(np.mean([set(x) != set(z) for x, z in zip(a0, a1)])),
             'test_none_top1_share': float((a1[:, 0] == 'NONE').mean()), 'submitted_none_top1_share': float((a0[:, 0] == 'NONE').mean()),
             'script': 'scripts/fuse_vlm.py', 'submission_sha256': fingerprint(out / 'submission.csv'), 'submitted': False}
        (out / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n')
        rep['staged'] = m
    (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
