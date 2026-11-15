"""Score a dev probability bundle (ids, classes, p[, final_ok]) from the Mac 2B rehearsal: MAP@3 and top-1 overall and by
slice, and the `final_ok` output against the teacher flag `final_answer_correct`.

Usage (repo root, PYTHONPATH=src): python runs/mac_2b/score_bundle.py <bundle.npz> [--out report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from k12.core import label_space, read_csv, row_scores, topk

ROOT = Path(__file__).resolve().parents[2]


def auc(score, y):
    """Probability that a random positive row scores above a random negative row (ties count half)."""
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def boot(values, n=2000, seed=7):
    rng = np.random.default_rng(seed); means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bundle'); ap.add_argument('--out'); a = ap.parse_args()
    classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv')); train = read_csv(ROOT / 'data/raw/train.csv')
    truth = dict(zip(train.trace_id, train.root_code)); ok = dict(zip(train.trace_id, train.final_answer_correct.astype(bool)))
    slices = json.loads((ROOT / 'data/v7/dev_slices.json').read_text())
    with np.load(a.bundle, allow_pickle=True) as b:
        ids = [str(t) for t in b['ids']]; cols = [list(b['classes']).index(c) for c in classes]; P = b['p'][:, cols].astype(float)
        final = b['final_ok'].astype(float) if 'final_ok' in b.files else None
    y = np.array([truth[t] for t in ids]); sl = np.array([slices[t] for t in ids])
    s = row_scores(y, topk(P, classes)); top1 = topk(P, classes)[:, 0] == y
    groups = {'all': np.ones(len(ids), bool), 'none': sl == 'none', 'seen_pair': sl == 'seen_pair', 'new_pair': sl == 'new_pair',
              'unseen_q': sl == 'unseen_q', 'hard': np.isin(sl, ['new_pair', 'unseen_q'])}
    rep = {'bundle': a.bundle, 'rows': len(ids), 'dev_rows_missing': len(set(slices) - set(ids)),
           'map3': {k: {'rows': int(m.sum()), 'map3': float(s[m].mean()), 'top1': float(top1[m].mean())} for k, m in groups.items()},
           'map3_all_ci95': boot(s),
           'predicted_top1_distinct_codes': int(len(set(topk(P, classes)[:, 0]))),
           'predicted_top1_none_share': float((topk(P, classes)[:, 0] == 'NONE').mean()), 'true_none_share': float((y == 'NONE').mean()),
           'mean_true_class_probability': float(P[np.arange(len(ids)), [classes.index(c) for c in y]].mean())}
    # baselines on the same rows: always the class prior of the fit rows
    fit = train[~train.trace_id.isin(set(slices))]; prior = fit.root_code.value_counts(normalize=True)
    PP = np.tile(np.array([prior.get(c, 0.) for c in classes]), (len(ids), 1)); sp = row_scores(y, topk(PP, classes))
    rep['prior_only_baseline'] = {k: float(sp[m].mean()) for k, m in groups.items()}
    if final is not None:
        t = np.array([ok[x] for x in ids]); p_none = P[:, classes.index('NONE')]
        rep['final_ok'] = {'teacher_true_share': float(t.mean()), 'accuracy_at_0.5': float(((final > .5) == t).mean()),
                           'majority_baseline_accuracy': float(max(t.mean(), 1 - t.mean())), 'auc': auc(final, t),
                           'mean_output_when_true': float(final[t].mean()), 'mean_output_when_false': float(final[~t].mean()),
                           'auc_of_p_none_for_the_same_flag': auc(p_none, t), 'accuracy_of_top1_is_none_for_the_flag': float(((topk(P, classes)[:, 0] == 'NONE') == t).mean()),
                           'flag_equals_truth_is_none_share': float((t == (y == 'NONE')).mean()),
                           'auc_of_final_ok_for_truth_is_none': auc(final, y == 'NONE'), 'auc_of_p_none_for_truth_is_none': auc(p_none, y == 'NONE'),
                           'correlation_final_ok_with_p_none': float(np.corrcoef(final, p_none)[0, 1])}
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')


if __name__ == '__main__':
    main()
