"""Approach A+B data recipe (CPU, $0): rewrite only the `weight` field of data/v5/cand_train.jsonl into
runs/deep/approach-W5/cand_train_w5.jsonl. Claude rows: label-shift weight pi_gold(c)/pi_v5(c), divided by sqrt of the
(question, code) count among v5 Claude rows, generic codes x0.7, then NONE vs error mass reset to the gold NONE share.
Gold rows on questions the Claude pool never covers: x3. Original file untouched."""
import json, collections
import numpy as np, pandas as pd
from k12.core import read_csv, make_folds
ROOT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'; OUT = ROOT + '/runs/deep/approach-W5'
GENERIC = {'CAL', 'WOP', 'ANS', 'OTH', 'NOT', 'FRM', 'VIS', 'CHK', 'UNT', 'REL'}
isgen = lambda c: c != 'NONE' and c.split('-')[0].upper() in GENERIC
tr = read_csv(ROOT + '/data/raw/train.csv'); fit = tr[make_folds(tr) != 0]
unl = pd.read_csv(ROOT + '/data/semisupervised/competition_data/unlabelled.csv', usecols=['trace_id', 'question_id']).set_index('trace_id').question_id
qmap = dict(zip(tr.trace_id, tr.question_id)); qmap.update(unl.to_dict()); poolq = set(unl)
rows = [json.loads(l) for l in open(ROOT + '/data/v5/cand_train.jsonl')]
truth = lambda r: r['codes'][r['letters'].index(r['target'].rsplit('ANSWER:', 1)[1].strip())]
cl = [r for r in rows if r['source'] == 'claude']
pg = fit.root_code.value_counts(normalize=True); cv = collections.Counter(truth(r) for r in cl)
pv = {c: n / len(cl) for c, n in cv.items()}; nqc = collections.Counter((qmap[r['trace_id']], truth(r)) for r in cl)
w = {}
for r in cl:
    c = truth(r); x = (pg.get(c, .5 / len(fit)) / pv[c]) / np.sqrt(nqc[(qmap[r['trace_id']], c)])
    w[r['trace_id']] = x * (.7 if isgen(c) else 1.)
v = np.array(list(w.values())); v = np.minimum(v, np.quantile(v, .99)); w = dict(zip(w, v))
isN = {t: truth(r) == 'NONE' for t, r in ((r['trace_id'], r) for r in cl)}
mN = sum(x for t, x in w.items() if isN[t]); mE = sum(x for t, x in w.items() if not isN[t]); tgt = float(pg['NONE'])
tot0 = sum(r['weight'] for r in cl)
for t in w: w[t] *= (tgt / mN if isN[t] else (1 - tgt) / mE) * tot0
st = collections.defaultdict(float)
with open(OUT + '/cand_train_w5.jsonl', 'w') as f:
    for r in rows:
        r = dict(r)
        if r['source'] == 'claude': r['weight'] = round(float(w[r['trace_id']]), 5)
        elif qmap[r['trace_id']] not in poolq: r['weight'] = 3.0
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
        c = truth(r); st[(r['source'], 'none' if c == 'NONE' else 'gen' if isgen(c) else 'spec')] += r['weight']
ww = np.array([w[r['trace_id']] for r in cl])
cE = [r for r in cl if truth(r) != 'NONE']
out = dict(claude_rows=len(cl), claude_weight_total_before=round(tot0, 1), claude_weight_total_after=round(float(ww.sum()), 1),
           claude_ess_before=round(len(cl), 1), claude_ess_after=round(float(ww.sum() ** 2 / (ww ** 2).sum()), 1),
           claude_none_share_after=round(st[('claude', 'none')] / sum(v for k, v in st.items() if k[0] == 'claude'), 3),
           claude_generic_share_of_err_before=round(np.mean([isgen(truth(r)) for r in cE]), 3),
           claude_generic_share_of_err_after=round(st[('claude', 'gen')] / (st[('claude', 'gen')] + st[('claude', 'spec')]), 3),
           gold_generic_share_of_err=round(float(fit.root_code[fit.root_code != 'NONE'].map(isgen).mean()), 3), gold_none_share=round(tgt, 3),
           gold_rows_upweighted_x3=sum(1 for r in rows if r['source'] == 'gold' and qmap[r['trace_id']] not in poolq),
           max_over_median_claude_weight=round(float(ww.max() / np.median(ww)), 2))
print(json.dumps(out, indent=1)); json.dump(out, open(OUT + '/cand_train_w5_stats.json', 'w'), indent=1)
