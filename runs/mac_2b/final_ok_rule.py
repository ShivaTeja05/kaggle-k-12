"""How much could a "final answer correct" signal add on top of the current fused system (dev, 301 rows)?

current fused P is rebuilt exactly as scripts/fuse_v3.py does. Rule: multiply p(NONE) by exp(+a) when the signal says the
final answer is correct and by exp(-b) when it says it is wrong, then renormalise. (a, b[, threshold]) are cross-fitted over
5 question folds (same folds as fuse_v3). Signals: the teacher's own flag (ceiling: a perfect predictor), and, when a head
bundle is given, the 2B head's final_ok output.

Usage (repo root, PYTHONPATH=src:scripts): python runs/mac_2b/final_ok_rule.py [--bundle head_dev.npz] [--out report.json]
"""
import argparse, json
from pathlib import Path
import numpy as np
import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals, sinkhorn, unseen_pairs
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from score_vlm_dev import shares
import fuse_v3 as V

ROOT = F.ROOT
ap = argparse.ArgumentParser(); ap.add_argument('--bundle'); ap.add_argument('--out'); a = ap.parse_args()
tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy(); flag = dev.final_answer_correct.to_numpy().astype(bool)
pts = lambda P: row_scores(y, topk(P, classes))
newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
P1 = aligned_bundle(ROOT / V.RANKER_DEV, ids, classes); SUB = fuse(P0, P1, .5, .75, pi, 30)
S, used = shares(ROOT / V.QWEN_DEV, ids, classes, first_only=True); sent = (used > 0) & (SUB.max(1) < .817)
Q = np.where(sent[:, None], normalize(S + V.CURRENT[0]), 1.)
CUR = sinkhorn(normalize(P0 * P1 ** .5 * Q ** V.CURRENT[1]), pi, V.CURRENT[2], 30); cur = pts(CUR); ni = classes.index('NONE')
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs); g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
part = lambda s: {'map3': float(s.mean()), 'none': float(s[none].mean()), 'seen_pair': float(s[seen].mean()), 'hard': float(s[hard].mean())}


def apply(signal_true, known, up, down):
    P = CUR.copy(); P[known & signal_true, ni] *= np.exp(up); P[known & ~signal_true, ni] *= np.exp(-down); return normalize(P)


def crossfit(scores):
    keys = list(scores); cf = cur.copy(); chosen = []
    for f in range(5):
        m = mf != f; k = keys[int(np.argmax([scores[k][m].mean() for k in keys]))]; chosen.append(k); cf[~m] = scores[k][~m]
    best = keys[int(np.argmax([scores[k].mean() for k in keys]))]
    return {'crossfit': part(cf), 'gain_crossfit_ci95': boot_ci(cf, cur, qid), 'in_sample_best': [list(best), float(scores[best].mean())],
            'fold_choices': [list(k) for k in chosen], 'rows_improved': int((cf > cur).sum()), 'rows_worse': int((cf < cur).sum())}


UPS, DOWNS = (0., .5, 1., 2., 4.), (0., .5, 1., 2., 4.)
rep = {'current_fused': part(cur), 'teacher_flag_true_share': float(flag.mean()), 'flag_equals_truth_none': float((flag == none).mean()),
       'current_top1_is_none_vs_flag_agreement': float(((topk(CUR, classes)[:, 0] == 'NONE') == flag).mean())}
everything = np.ones(len(ids), bool)
rep['ceiling_with_teacher_flag'] = crossfit({(u, d): pts(apply(flag, everything, u, d)) for u in UPS for d in DOWNS})
if a.bundle:
    with np.load(a.bundle, allow_pickle=True) as b:
        at = {str(t): i for i, t in enumerate(b['ids'])}; fo = np.array([b['final_ok'][at[t]] if t in at else np.nan for t in ids])
        pn = np.array([b['p'][at[t]][list(b['classes']).index('NONE')] if t in at else np.nan for t in ids])
    known = ~np.isnan(fo); rep['rows_with_head'] = int(known.sum())
    rep['with_2b_final_ok'] = crossfit({(t, u, d): pts(apply(np.nan_to_num(fo) > t, known, u, d)) for t in (.3, .4, .5, .6, .7) for u in UPS for d in DOWNS})
    qn = np.quantile(pn[known], [.5, .6, .67, .75, .85])
    rep['with_2b_p_none_instead'] = crossfit({(float(t), u, d): pts(apply(np.nan_to_num(pn) > t, known, u, d)) for t in qn for u in UPS for d in DOWNS})
print(json.dumps(rep, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
