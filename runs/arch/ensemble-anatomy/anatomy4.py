"""Part 4: how often is a row's code already seen for its question (LOO over all 1,500 train) + CIs of gap per group."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from k12.core import read_csv, make_folds
train = read_csv('data/raw/train.csv'); test = read_csv('data/raw/test.csv')
g = train.groupby('question_id')
loo_in, loo_seen = [], []
for q, code in zip(train.question_id, train.root_code):
    others = train[(train.question_id == q)]
    loo_seen.append(len(others) > 1)
    loo_in.append((others.root_code == code).sum() > 1)
loo_in, loo_seen = np.array(loo_in), np.array(loo_seen)
rows_per_q_train = train.question_id.value_counts()
tq = test.question_id.map(rows_per_q_train).fillna(0)
out = {'train_rows_question_has_other_rows': float(loo_seen.mean()),
       'train_rows_code_seen_in_other_rows_of_same_question': float(loo_in.mean()),
       'given_seen_code_seen': float(loo_in[loo_seen].mean()),
       'test_rows_question_in_train': float((tq > 0).mean()),
       'test_mean_train_rows_per_question': float(tq.mean()),
       'dev_mean_fit_rows_per_question_(1199)': None}
d = read_csv('runs/arch/ensemble-anatomy/rows.csv') if False else pd.read_csv('runs/arch/ensemble-anatomy/rows.csv', keep_default_na=False, dtype={'question_id': str})
folds = make_folds(train); fit = train[folds != 0]
fq = fit.question_id.value_counts()
out['dev_mean_fit_rows_per_question_(1199)'] = float(d.question_id.map(fq).fillna(0).mean())
qcodes = fit.groupby('question_id').root_code.apply(set).to_dict()
grp = np.array(['unseen' if q not in qcodes else ('in' if t in qcodes[q] else 'notin') for q, t in zip(d.question_id, d.truth)])
rng = np.random.default_rng(3)
uq = d.question_id.unique(); qi = {q: i for i, q in enumerate(uq)}
idx = d.question_id.map(qi).to_numpy()
gap = (d.claude - d.blend).to_numpy()
ci = {}
for name in ['in', 'notin', 'unseen']:
    m = grp == name
    contrib = np.zeros(len(uq)); sizes = np.bincount(idx, minlength=len(uq)).astype(float)
    np.add.at(contrib, idx[m], gap[m])
    s = rng.integers(0, len(uq), (4000, len(uq)))
    b = contrib[s].sum(1) / sizes[s].sum(1)
    ci[name] = {'gap_map_contrib': round(float(gap[m].sum() / len(d)), 4), 'ci95': [round(float(x), 4) for x in np.quantile(b, [.025, .975])]}
contrib = np.zeros(len(uq)); np.add.at(contrib, idx, gap); sizes = np.bincount(idx).astype(float)
s = rng.integers(0, len(uq), (4000, len(uq))); b = contrib[s].sum(1) / sizes[s].sum(1)
ci['total'] = {'gap': round(float(gap.mean()), 4), 'ci95': [round(float(x), 4) for x in np.quantile(b, [.025, .975])]}
out['gap_ci_by_group'] = ci
print(json.dumps(out, indent=1))
Path('runs/arch/ensemble-anatomy/anatomy4.json').write_text(json.dumps(out, indent=2) + '\n')
