"""How strongly do our models' rankings stick to codes already seen for the same question? Can the blend's confidence flag the risky rows?"""
import json
import numpy as np
import pandas as pd
from k12.core import read_csv, make_folds, label_space, aligned_bundle, map3
from k12.baseline import HierarchicalPrior
OUT = 'runs/arch/teacher-gap/'
train = read_csv('data/raw/train.csv'); folds = make_folds(train); classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
fit, dev = train.loc[folds != 0], train.loc[folds == 0].reset_index(drop=True)
d = pd.read_csv(OUT + 'rows.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
assert (d.trace_id == dev.trace_id).all()
qc = {q: set(g.root_code) for q, g in fit.groupby('question_id')}
seen = lambda q, c: c in qc.get(q, set())
res = {}
for name, col in [('blend', 'blend'), ('claude', 'claude'), ('m32', 'm32'), ('m8', 'm8'), ('vl', 'mvl')]:
    slots = [[seen(q, c) for c in s.split()] for q, s in zip(d.question_id, d[col])]
    res[f'{name}_top1_seen_for_q'] = round(float(np.mean([s[0] for s, q in zip(slots, d.question_id) if q in qc])), 3)
    res[f'{name}_top3_slots_seen_for_q'] = round(float(np.mean([np.mean(s) for s, q in zip(slots, d.question_id) if q in qc])), 3)
res['truth_seen_for_q'] = round(float(np.mean([seen(q, t) for q, t in zip(d.question_id, d.truth) if q in qc])), 3)
res['truth_seen_for_q_error_rows'] = round(float(np.mean([seen(q, t) for q, t in zip(d.question_id, d.truth) if q in qc and t != 'NONE'])), 3)
prior = HierarchicalPrior(classes).fit(fit).predict_proba(dev)
res['hierarchical_prior_only_map3'] = round(map3(dev.root_code, prior, classes), 4)
blend = None
ids = dev.trace_id.tolist()
B = [aligned_bundle(p, ids, classes) for p in ['runs/pod/text32/validation_tx_dev_epoch_1.npz', 'runs/pod/tx/text_tx_dev_e3.npz', 'runs/pod/dev-distill/validation_epoch_2.npz']]
blend = np.exp(.35 * np.log(B[0]) + .30 * np.log(B[1]) + .35 * np.log(B[2])); blend /= blend.sum(1, keepdims=True)
top = blend.max(1)
e = (d.truth != 'NONE').to_numpy()
unseen = np.array([not seen(q, t) for q, t in zip(d.question_id, d.truth)])
from sklearn.metrics import roc_auc_score
res['blend_top1_prob_mean'] = {'code_seen_rows': round(float(top[e & ~unseen].mean()), 3), 'code_unseen_rows': round(float(top[e & unseen].mean()), 3)}
res['auc_low_blend_confidence_flags_unseen_code'] = round(float(roc_auc_score(unseen[e], -top[e])), 3)
top1_seen = np.array([seen(q, s.split()[0]) for q, s in zip(d.question_id, d.blend)])
res['blend_top1_correct_when_top1_is_seen_code'] = round(float((d.blend.str.split().str[0] == d.truth)[top1_seen].mean()), 3)
res['blend_top1_correct_when_top1_is_unseen_code'] = round(float((d.blend.str.split().str[0] == d.truth)[~top1_seen].mean()), 3)
res['rows_blend_top1_seen_code'] = int(top1_seen.sum())
json.dump(res, open(OUT + 'memorisation.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
