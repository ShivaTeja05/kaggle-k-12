"""CPU check: does simply dividing out the question prior (blend * prior^-a) recover unseen-code rows? a cross-fitted within dev (5 question-grouped folds)."""
import json
import numpy as np
import pandas as pd
from k12.core import read_csv, make_folds, label_space, aligned_bundle, row_scores, topk, normalize
from k12.baseline import HierarchicalPrior
train = read_csv('data/raw/train.csv'); folds = make_folds(train); classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
fit, dev = train.loc[folds != 0], train.loc[folds == 0].reset_index(drop=True); ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
B = [aligned_bundle(p, ids, classes) for p in ['runs/pod/text32/validation_tx_dev_epoch_1.npz', 'runs/pod/tx/text_tx_dev_e3.npz', 'runs/pod/dev-distill/validation_epoch_2.npz']]
blend = normalize(np.exp(.35 * np.log(B[0]) + .30 * np.log(B[1]) + .35 * np.log(B[2])))
prior = HierarchicalPrior(classes).fit(fit).predict_proba(dev)
grid = [0, .1, .2, .3, .5, .75, 1.]
adj = lambda a: normalize(blend * np.clip(prior, 1e-6, None) ** (-a))
qs = dev.question_id.unique(); rng = np.random.default_rng(2026); rng.shuffle(qs)
qf = {q: i % 5 for i, q in enumerate(qs)}; f = dev.question_id.map(qf).to_numpy()
scores = {a: row_scores(y, topk(adj(a), classes)) for a in grid}
out = np.zeros(len(dev)); chosen = []
for k in range(5):
    best = max(grid, key=lambda a: scores[a][f != k].mean()); chosen.append(best)
    out[f == k] = scores[best][f == k]
base = scores[0]
g = pd.DataFrame({'q': dev.question_id, 'd': out - base}).groupby('q').d.agg(['sum', 'size']).to_numpy()
s = np.random.default_rng(2026).integers(0, len(g), (4000, len(g))); b = g[s, 0].sum(1) / g[s, 1].sum(1)
res = {'in_sample_by_a': {str(a): round(float(scores[a].mean()), 4) for a in grid}, 'crossfit_map3': round(float(out.mean()), 4),
       'chosen_a_per_fold': chosen, 'delta_vs_0718': round(float((out - base).mean()), 4), 'ci95': [round(float(x), 4) for x in np.quantile(b, [.025, .975])]}
json.dump(res, open('runs/arch/teacher-gap/debias_cf.json', 'w'), indent=1); print(json.dumps(res, indent=1))
