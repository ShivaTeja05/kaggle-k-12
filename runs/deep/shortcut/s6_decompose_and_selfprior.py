"""(6) How much of the new-pair gap is the shortcut itself, and does a label-free per-question self-prior help?

Part A (oracle decomposition, O): on new-pair rows, delete the question's fit codes from each model's ranking
(only possible because we know the row is new-pair) and rescore. The gain is the most that perfectly suppressing
the shortcut could give with the model's OWN remaining ranking; the rest of the gap to Claude is competence
(not recognising the error type), which de-biasing cannot fix.
Part B (deployable, M): per-question batch calibration (Zhou et al. 2023 'Batch Calibration'; Zhao et al. 2021
contextual calibration) - divide each row by the mean prediction of the OTHER rows of the same question:
p_i * (mean_{j!=i, q(j)=q(i)} p_j)^(-a), a cross-fitted on 5 question folds. Dev rows only (no fit-only test bundles).
Part C: question-bootstrap CIs for the within-question homogeneity numbers of s2.
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from k12.core import topk, row_scores, normalize, read_csv
from sc_common import load, dev_preds, test_preds, claude_dev, slices, OUT, ROOT, r, boot_ci

tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
S, qc = slices(dev, fit)
y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
P = dev_preds(dev, classes)
PC, _ = claude_dev(dev, classes); P['claude'] = PC
res = {'A_oracle_shortcut_removal_new_pair': {}}
m = S == 'new_pair'
for n in ['m32', 'm8', 'vl', 'blend', 'claude']:
    base = row_scores(y, topk(P[n], classes))
    Pm = P[n].copy()
    for i in np.where(m)[0]:
        for c in qc[qid[i]]:
            Pm[i, ci[c]] = 1e-12
    sc = row_scores(y, topk(Pm, classes))
    top1_fit = np.array([topk(P[n], classes)[i, 0] in qc[qid[i]] for i in np.where(m)[0]])
    res['A_oracle_shortcut_removal_new_pair'][n] = dict(map3=r(base[m].mean()), map3_fit_codes_removed=r(sc[m].mean()),
                                                         gain_per_301=r((sc - base)[m].sum() / 301),
                                                         top1_is_fit_code_rows=int(top1_fit.sum()),
                                                         map3_rows_top1_fit_code=r(base[m][top1_fit].mean()) if top1_fit.any() else None,
                                                         map3_rows_top1_not_fit_code=r(base[m][~top1_fit].mean()))
b = res['A_oracle_shortcut_removal_new_pair']
gap = (b['claude']['map3'] - b['blend']['map3']) * m.sum() / 301
res['A_summary'] = dict(new_pair_gap_per_301=r(gap), recoverable_by_perfect_shortcut_removal_per_301=b['blend']['gain_per_301'],
                        share_of_new_pair_gap_that_is_shortcut=r(b['blend']['gain_per_301'] / gap),
                        residual_competence_gap_per_301=r(gap - b['blend']['gain_per_301']),
                        note='oracle: requires knowing which rows are new-pair; blend after removal vs Claude untouched')
# where does truth rank in the blend on new-pair rows, after removing fit codes
rk = []
for i in np.where(m)[0]:
    order = list(topk(P['blend'], classes, k=76)[i])
    rk.append(order.index(y[i]) + 1)
res['A_blend_truth_rank_new_pair'] = dict(median=float(np.median(rk)), share_rank1=r(np.mean(np.array(rk) == 1)),
                                          share_rank_le3=r(np.mean(np.array(rk) <= 3)), share_rank_le10=r(np.mean(np.array(rk) <= 10)))

# Part B: per-question self-prior (batch calibration) on dev
groups = dev.groupby('question_id').indices
multi = np.array([len(groups[q]) >= 2 for q in qid])
qs = dev.question_id.unique(); rr = np.random.default_rng(2026); rr.shuffle(qs); qf = {q: i % 5 for i, q in enumerate(qs)}
fold = dev.question_id.map(qf).to_numpy()
grid = [0, .1, .2, .3, .5, .75, 1.]
res['B_self_prior'] = {'rows_with_other_rows_same_q': int(multi.sum())}
for n in ['m32', 'm8', 'vl', 'blend']:
    mean_other = np.ones_like(P[n]) / len(classes)
    for i in np.where(multi)[0]:
        o = [j for j in groups[qid[i]] if j != i]
        mean_other[i] = P[n][o].mean(0)
    sc = {a: row_scores(y, topk(normalize(P[n] * np.clip(mean_other, 1e-6, 1) ** (-a)), classes)) for a in grid}
    out = np.zeros(len(dev)); ch = []
    for f in range(5):
        a = max(grid, key=lambda a: sc[a][(fold != f) & multi].mean()); ch.append(a); out[fold == f] = sc[a][fold == f]
    res['B_self_prior'][n] = dict(in_sample_multi_rows={str(a): r(sc[a][multi].mean()) for a in grid}, chosen=ch,
                                  crossfit_delta_multi_rows=dict(zip(['delta', 'ci95'], boot_ci(out[multi], sc[0][multi], qid[multi]))),
                                  crossfit_delta_new_pair_multi=r((out - sc[0])[multi & m].mean()) if (multi & m).any() else None)

# Part C: CIs for within-question homogeneity
pairs = [(i, j) for q, g in dev.groupby('question_id') for i, j in combinations(g.index, 2)]
pq = np.array([qid[i] for i, _ in pairs])
st = np.array([y[i] == y[j] for i, j in pairs]); be = np.array([y[i] != 'NONE' and y[j] != 'NONE' for i, j in pairs])
t1 = {n: topk(P[n], classes)[:, 0] for n in ['blend', 'claude', 'm32', 'm8', 'vl']}
same = {n: np.array([t1[n][i] == t1[n][j] for i, j in pairs], float) for n in t1}
k = be & ~st
res['C_dev_same_top1_truth_differs_both_error'] = {n: r(same[n][k].mean()) for n in same} | {
    'blend_minus_claude': dict(zip(['delta', 'ci95'], boot_ci(same['blend'][k], same['claude'][k], pq[k]))), 'pairs': int(k.sum())}
res['C_dev_simpson'] = {n: r(same[n].mean()) for n in same} | {'truth': r(st.mean()),
                                                               'blend_minus_truth': dict(zip(['delta', 'ci95'], boot_ci(same['blend'], st.astype(float), pq)))}
# Test: blend top-1 Simpson vs train truth Simpson, question bootstrap over shared questions
test = read_csv(ROOT / 'data/raw/test.csv')
PT = test_preds(test, classes)
tq = test.question_id.to_numpy(); trq = tr.question_id.to_numpy(); yt = tr.root_code.to_numpy()
def per_q_simpson(codes, qarr):
    out = {}
    for q, idx in pd.Series(range(len(qarr))).groupby(qarr).groups.items():
        idx = list(idx)
        if len(idx) >= 2:
            out[q] = (np.mean([codes[i] == codes[j] for i, j in combinations(idx, 2)]), len(idx) * (len(idx) - 1) / 2)
    return out
truth_q = per_q_simpson(yt, trq)
res['C_test'] = {}
for n in ['m32', 'm8', 'vl', 'blend']:
    pred_q = per_q_simpson(topk(PT[n], classes)[:, 0], tq)
    common = sorted(set(truth_q) & set(pred_q))
    a = np.array([pred_q[q][0] for q in common]); b = np.array([truth_q[q][0] for q in common]); w = np.array([pred_q[q][1] for q in common])
    rng = np.random.default_rng(2026); boots = []
    for _ in range(4000):
        s = rng.integers(0, len(common), len(common))
        boots.append(np.average(a[s], weights=w[s]) - np.average(b[s], weights=w[s]))
    res['C_test'][n] = dict(questions=len(common), pred_simpson=r(np.average(a, weights=w)), truth_simpson=r(np.average(b, weights=w)),
                            delta=r(np.average(a, weights=w) - np.average(b, weights=w)), ci95=[r(x) for x in np.quantile(boots, [.025, .975])])
    # Error-only version: drop rows predicted/labelled NONE
    pe = topk(PT[n], classes)[:, 0]
    ke = pe != 'NONE'; kt = yt != 'NONE'
    pq_e = per_q_simpson(pe[ke], tq[ke]); tq_e = per_q_simpson(yt[kt], trq[kt])
    ce = sorted(set(pq_e) & set(tq_e))
    w = np.array([pq_e[q][1] for q in ce])
    res['C_test'][n]['error_only'] = dict(questions=len(ce), pred_simpson=r(np.average([pq_e[q][0] for q in ce], weights=w)),
                                          truth_simpson=r(np.average([tq_e[q][0] for q in ce], weights=w)))
json.dump(res, open(OUT / 's6_decompose_and_selfprior.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
