"""(2) Within-question prediction diversity vs truth diversity.

A memorising model gives every student on a question the same codes. For all pairs of rows on the same question:
- Simpson index S = P(two rows share top-1), for truth, each model, Claude.
- P(model top-1 equal | truths differ) and | truths equal), and the AUC with which the pairwise JS similarity of
  the model's distributions separates same-truth from different-truth pairs ("does the model look at the student?").
Dev: 301 rows (pairs within the dev fold). Test, label-free: Simpson of top-1 predictions over test pairs vs the
truth Simpson over train pairs of the same questions (the test models trained on all of train).
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from k12.core import read_csv, topk
from sc_common import load, dev_preds, test_preds, claude_dev, slices, OUT, ROOT, r

tr, tax, classes, dev, fit = load()
S, qc = slices(dev, fit)
P = dev_preds(dev, classes)
PC, _ = claude_dev(dev, classes)
P['claude'] = PC
y = dev.root_code.to_numpy()


def js(p, q):
    p = np.clip(p, 1e-9, 1); q = np.clip(q, 1e-9, 1); m = (p + q) / 2
    return .5 * (p * np.log(p / m)).sum() + .5 * (q * np.log(q / m)).sum()


pairs = [(i, j) for q, g in dev.groupby('question_id') for i, j in combinations(g.index, 2)]
same_truth = np.array([y[i] == y[j] for i, j in pairs])
both_err = np.array([y[i] != 'NONE' and y[j] != 'NONE' for i, j in pairs])
res = {'dev': {'questions_with_2plus_rows': int((dev.question_id.value_counts() >= 2).sum()),
               'rows_in_them': int(dev.question_id.map(dev.question_id.value_counts()).ge(2).sum()),
               'pairs': len(pairs), 'pairs_truth_differs': int((~same_truth).sum()),
               'pairs_both_error': int(both_err.sum()), 'pairs_both_error_truth_differs': int((both_err & ~same_truth).sum()),
               'truth_simpson': r(same_truth.mean()), 'truth_simpson_both_error': r(same_truth[both_err].mean())}}
for n in ['m32', 'm8', 'vl', 'blend', 'claude']:
    t1 = topk(P[n], classes)[:, 0]
    t3 = topk(P[n], classes)
    same = np.array([t1[i] == t1[j] for i, j in pairs])
    jac = np.array([len(set(t3[i]) & set(t3[j])) / len(set(t3[i]) | set(t3[j])) for i, j in pairs])
    sim = -np.array([js(P[n][i], P[n][j]) for i, j in pairs])
    d = dict(top1_simpson=r(same.mean()), top1_simpson_both_error=r(same[both_err].mean()),
             same_top1_given_truth_differs=r(same[~same_truth].mean()),
             same_top1_given_truth_differs_both_error=r(same[both_err & ~same_truth].mean()),
             same_top1_given_truth_same=r(same[same_truth].mean()),
             top3_jaccard_truth_differs=r(jac[~same_truth].mean()), top3_jaccard_truth_same=r(jac[same_truth].mean()),
             auc_similarity_separates_same_truth=r(roc_auc_score(same_truth, sim)),
             auc_similarity_separates_same_truth_both_error=r(roc_auc_score(same_truth[both_err], sim[both_err])))
    res['dev'][n] = d
# Distinct top-1 codes per question vs distinct truths (questions with >=3 dev rows)
big = dev.question_id.value_counts(); big = big[big >= 3].index
res['dev']['questions_3plus'] = int(len(big))
for n in ['truth', 'm32', 'm8', 'vl', 'blend', 'claude']:
    col = y if n == 'truth' else topk(P[n], classes)[:, 0]
    res['dev'].setdefault('distinct_top1_per_q_3plus', {})[n] = r(np.mean([len(set(col[dev.question_id == q])) for q in big]), 3)

# Test, label-free
test = read_csv(ROOT / 'data/raw/test.csv')
PT = test_preds(test, classes)
tpairs = [(i, j) for q, g in test.groupby('question_id') for i, j in combinations(g.index, 2)]
trpairs = [(i, j) for q, g in tr.groupby('question_id') for i, j in combinations(g.index, 2)]
yt = tr.root_code.to_numpy()
# weight train pairs so each question counts as in the test pairs (test question pair counts)
tq = test.question_id.to_numpy(); trq = tr.question_id.to_numpy()
tp_q = pd.Series([tq[i] for i, _ in tpairs]).value_counts()
tr_same = pd.DataFrame({'q': [trq[i] for i, _ in trpairs], 's': [yt[i] == yt[j] for i, j in trpairs]}).groupby('q').s.mean()
common = tr_same.index.intersection(tp_q.index)
res['test'] = {'pairs': len(tpairs), 'questions_with_pairs_in_both': int(len(common)),
               'truth_simpson_train_pairs_weighted_by_test_pairs': r(np.average(tr_same[common], weights=tp_q[common])),
               'truth_simpson_train_pairs_unweighted': r(np.mean([yt[i] == yt[j] for i, j in trpairs]))}
for n in ['m32', 'm8', 'vl', 'blend']:
    t1 = topk(PT[n], classes)[:, 0]
    s = pd.DataFrame({'q': [tq[i] for i, _ in tpairs], 's': [t1[i] == t1[j] for i, j in tpairs]}).groupby('q').s.mean()
    res['test'][n] = dict(top1_simpson=r(np.mean([t1[i] == t1[j] for i, j in tpairs])),
                          top1_simpson_common_q_weighted=r(np.average(s[common], weights=tp_q[common])))
    # distinct top-1 codes per question on test rows vs distinct truth codes in train, for questions with >=5 rows in both
    qs = [q for q in common if (tq == q).sum() >= 5 and (trq == q).sum() >= 5]
    res['test'][n]['distinct_top1_per_q_5plus'] = r(np.mean([len(set(t1[tq == q][:5])) for q in qs]), 3)
res['test']['distinct_truth_per_q_5plus_train_first5'] = r(np.mean([len(set(yt[trq == q][:5])) for q in qs]), 3)
res['test']['questions_5plus_both'] = len(qs)
# NONE share of predictions vs train NONE share, for context
res['test']['none_top1_share'] = {n: r((topk(PT[n], classes)[:, 0] == 'NONE').mean()) for n in ['m32', 'm8', 'vl', 'blend']}
res['test']['train_none_share'] = r((tr.root_code == 'NONE').mean())
json.dump(res, open(OUT / 's2_within_question.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
