"""Score the Stage 1 second-opinion arms on the dev rows they cover, against the current fused system.

For each arm (ranked codes from Qwen3.8 on the least-confident dev rows) it reports, on the rows the arm answered:
the submitted system, the current fused system (one Qwen answer), the arm used alone, and the arm fused in three ways:
replacing the first answer, multiplied with it, or (for a repeat sample) averaged with it. Fixed settings (floor 0.01,
weight 1, lambda 0.5); the sample is too small to tune more. Usage:
  python scripts/score_stage1.py rerank_ex=runs/deep/q38/stage1_pod/rerank_ex.jsonl ... --report reports/q38_stage1.json
"""
import argparse
import json

import numpy as np

import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals, sinkhorn, unseen_pairs
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from score_vlm_dev import shares

ROOT = F.ROOT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('arms', nargs='+'); ap.add_argument('--report')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    pts = lambda P: row_scores(y, topk(P, classes))
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz', ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30)
    S1, u1 = shares(ROOT / 'runs/deep/q38/pod_out/q38_dev_think.jsonl', ids, classes, True)
    sent = (u1 > 0) & (SUB.max(1) < .817); agree = S1.argmax(1) == SUB.argmax(1)

    def fused(Q, rows):  # Q is rows x classes of multipliers; rows not in `rows` keep the current treatment
        base = np.where(sent[:, None], normalize(S1 + .01), 1.)
        return sinkhorn(normalize(P0 * P1 ** .5 * np.where(rows[:, None], Q, base)), pi, .5, 30)

    cur = pts(fused(np.ones_like(P0), np.zeros(len(ids), bool)))
    part = lambda s, m: {'rows': int(m.sum()), 'map3': round(float(s[m].mean()), 4), 'none': round(float(s[m & none].mean()), 3) if (m & none).any() else None,
                         'seen_pair': round(float(s[m & seen].mean()), 3) if (m & seen).any() else None, 'hard': round(float(s[m & hard].mean()), 3) if (m & hard).any() else None}
    rep = {'rows_sent': int(sent.sum()), 'rows_disagree': int((sent & ~agree).sum())}
    for spec in a.arms:
        name, path = spec.split('=', 1)
        S, u = shares(ROOT / path, ids, classes, True); m = u > 0
        R = normalize(S + .01)
        variants = {'arm_alone': pts(normalize(S + 1e-6)),
                    'replace_first_answer': pts(fused(R, m)),
                    'multiply_with_first_answer': pts(fused(normalize(S1 + .01) * R, m)),
                    'average_with_first_answer': pts(fused(normalize((S1 + S) / 2 + .01), m)),
                    'arm_squared': pts(fused(R ** 2, m))}
        entry = {'answered': int(m.sum()), 'submitted_system': part(pts(SUB), m), 'current_fused': part(cur, m)}
        for k, s in variants.items():
            entry[k] = part(s, m); entry[k]['minus_current'] = [round(x, 4) for x in (lambda d: (d[0], *d[1]))(boot_ci(s[m], cur[m], qid[m]))]
            entry[k]['on_disagreement_rows'] = round(float(s[m & ~agree].mean()), 3) if (m & ~agree).any() else None
            entry[k]['whole_dev_270_gain'] = round(float((s - cur)[m].sum() / (u1 > 0).sum()), 4)
        entry['current_fused']['on_disagreement_rows'] = round(float(cur[m & ~agree].mean()), 3) if (m & ~agree).any() else None
        top = np.asarray(classes)[S.argmax(1)]
        entry['arm_top1_right'] = round(float((top == y)[m].mean()), 3); entry['first_answer_top1_right'] = round(float((np.asarray(classes)[S1.argmax(1)] == y)[m].mean()), 3)
        entry['arm_top_equals_first_answer'] = round(float((S.argmax(1) == S1.argmax(1))[m].mean()), 3); entry['arm_top_equals_ours'] = round(float((S.argmax(1) == SUB.argmax(1))[m].mean()), 3)
        rep[name] = entry
    if a.report:
        (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
