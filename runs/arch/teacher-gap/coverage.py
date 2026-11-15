"""Does Claude-labelled unlabelled data cover the (question, code) pairs the blend misses, and did more of it help there?"""
import json
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from k12.core import read_csv, make_folds, label_space, aligned_bundle, topk, row_scores
OUT = 'runs/arch/teacher-gap/'
train = read_csv('data/raw/train.csv'); folds = make_folds(train); classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
fit = train.loc[folds != 0]
d = pd.read_csv(OUT + 'rows.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
ids, truth = d.trace_id.tolist(), d.truth.to_numpy()
qc = {q: set(g.root_code) for q, g in fit.groupby('question_id')}
d['bin'] = np.where(~d.question_id.isin(qc), 'q_unseen', np.where([t in qc.get(q, set()) for t, q in zip(d.truth, d.question_id)], 'code_seen_for_q', 'code_unseen_for_q'))
unl = read_csv('data/semisupervised/competition_data/unlabelled.csv')[['trace_id', 'question_id']]
uq = dict(zip(unl.trace_id, unl.question_id))
def pairs(path, key):
    s = defaultdict(Counter)
    for line in open(path):
        r = json.loads(line)
        if key == 'answer':
            a = r.get('answer') or {}
            codes = [a.get('first_code'), a.get('second_code'), a.get('third_code')]
        else:
            codes = sorted(r['soft'], key=r['soft'].get, reverse=True)
        if r['trace_id'] in uq and codes[0]:
            s[uq[r['trace_id']]][codes[0]] += 1
    return s
all_v4 = pairs('runs/claude/unl_all_v4.jsonl', 'answer')
used_v4 = pairs('data/v3/claude_unl_v4.jsonl', 'soft')
res = {'unlabelled_questions_overlap_dev': float(d.question_id.isin(set(uq.values())).mean())}
for name, s in [('claude_unl_all_v4_26k', all_v4), ('claude_unl_v4_11k_used_in_training', used_v4)]:
    cov = [t in s.get(q, {}) for t, q in zip(d.truth, d.question_id)]
    d['cov_' + name] = cov
    e = d[d.truth != 'NONE']
    res[name] = {b: {'rows': int((e.bin == b).sum()), 'truth_code_is_claude_top1_for_some_unl_photo_of_q': round(float(e['cov_' + name][e.bin == b].mean()), 3)} for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen']}
bundles = {'scale_0_tx': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz', 'scale_25_tx': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
           'scale_100_tx': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', 'scale_0_gold': 'runs/pod/scaling/scale_0/validation_gold_epoch_1.npz',
           'scale_100_gold': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz', 'text32_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
           'text8_tx_e3': 'runs/pod/tx/text_tx_dev_e3.npz', 'vl_distill': 'runs/pod/dev-distill/validation_epoch_2.npz',
           'fewshot235': 'runs/pod/probe235/fewshot235/validation.npz'}
tab = {}
for name, path in bundles.items():
    try:
        s = row_scores(truth, topk(aligned_bundle(path, ids, classes), classes))
    except Exception as ex:
        tab[name] = str(ex); continue
    tab[name] = {'all': round(s.mean(), 4), 'NONE_rows': round(s[truth == 'NONE'].mean(), 3)} | {b: round(s[(d.bin == b).to_numpy() & (truth != 'NONE')].mean(), 3) for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen']}
tab['blend'] = {'all': round(d.s_blend.mean(), 4), 'NONE_rows': round(d.s_blend[truth == 'NONE'].mean(), 3)} | {b: round(d.s_blend[(d.bin == b) & (d.truth != 'NONE')].mean(), 3) for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen']}
tab['claude'] = {'all': round(d.s_claude.mean(), 4), 'NONE_rows': round(d.s_claude[truth == 'NONE'].mean(), 3)} | {b: round(d.s_claude[(d.bin == b) & (d.truth != 'NONE')].mean(), 3) for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen']}
res['map3_by_bin'] = tab
res['bin_rows_error_only'] = d[d.truth != 'NONE'].bin.value_counts().to_dict()
# How common is 'code unseen for its question' under cross-fitting? (5 folds, train only) -> proxy for test with full train
rates = []
for f in range(5):
    fq = {q: set(g.root_code) for q, g in train.loc[folds != f].groupby('question_id')}
    v = train.loc[(folds == f) & (train.root_code != 'NONE')]
    rates.append(float(np.mean([t not in fq.get(q, set()) for t, q in zip(v.root_code, v.question_id)])))
res['error_rows_code_unseen_for_question_rate_by_fold'] = [round(r, 3) for r in rates]
# test: how many training rows per test question (full train) vs dev question (fit)
test = read_csv('data/raw/test.csv')
nq = train.groupby('question_id').size()
res['train_rows_per_test_question_median'] = float(test.question_id.map(nq).fillna(0).median())
res['fit_rows_per_dev_question_median'] = float(d.question_id.map(fit.groupby('question_id').size()).fillna(0).median())
res['distinct_codes_per_question_in_train_median'] = float(train.groupby('question_id').root_code.nunique().median())
json.dump(res, open(OUT + 'coverage.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
