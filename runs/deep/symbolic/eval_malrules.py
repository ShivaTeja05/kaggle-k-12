"""How predictive are mal-rule / correct-path tokens of the gold root code?
Gold transcripts of all 1,500 train rows (rules have no fitted parameters; the path pool is leave-one-out),
plus machine transcripts of dev.  Writes out/tokens_{gold,machine_dev,claude_unl}.json"""
import sys, json, collections, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load
from malrules import row_tokens

tr, tax, classes, folds, dev, fit = load()
R = lambda p: {r['trace_id']: r for r in map(json.loads, open(p))}
gold, mdev = R(HERE / 'out/gold_train.jsonl'), R(HERE / 'out/machine_dev.jsonl')
unl = {}
for k in range(8):
    p = HERE / f'out/unl_shard{k}.jsonl'
    if p.exists():
        unl.update(R(p))
meta = {}
for l in open(ROOT / 'data/claude_dataset/labels.jsonl'):
    r = json.loads(l)
    if r['split'] == 'unlabelled':
        meta[r['trace_id']] = (r['question_id'], r['first_code'], r['correct_answer'])

MAP = {'SIGN': {'INT-02', 'INT-03', 'INT-04', 'ALG-06', 'OOO-02'}, 'SLIP': {'CAL-01'}, 'ANS-01': {'ANS-01', 'ANS-02'}}


def codes_of(tok):
    c = tok.split('_')[-1]
    return MAP.get(c, {c})


def vals(rec):
    return set(v for x in rec['lines'] for v in x['values'])


# correct-path pool: values in NONE transcripts per question (gold train NONE rows + Claude-NONE unlabelled)
pool_rows = collections.defaultdict(list)
for r in tr.itertuples():
    if r.root_code == 'NONE':
        pool_rows[r.question_id].append((r.trace_id, vals(gold[r.trace_id])))
for t, rec in unl.items():
    q, c, _ = meta.get(t, (None, None, None))
    if c == 'NONE':
        pool_rows[q].append((t, vals(rec)))


def last_in_path(rec, q, exclude, allowed=None):
    lv = [v for x in rec['lines'][-2:] for v in x['values'] if abs(v) > 1]
    if not lv:
        return False
    for tid, S in pool_rows.get(q, []):
        if tid == exclude or (allowed is not None and tid not in allowed and tid not in unl):
            continue
        if any(abs(v - s) <= 1e-3 * max(1, abs(s)) for v in lv for s in S):
            return True
    return False


def tokens_for(rec, q, tid, allowed=None):
    fw = rec['fac'] is not None and rec['fac'] < 1
    return row_tokens(rec, fw, last_in_path(rec, q, tid, allowed) if fw else False)


TG = {r.trace_id: tokens_for(gold[r.trace_id], r.question_id, r.trace_id) for r in tr.itertuples()}
fit_ids = set(fit.trace_id)
TM = {r.trace_id: tokens_for(mdev[r.trace_id], r.question_id, r.trace_id, fit_ids) for r in dev.itertuples()}
TU = {t: tokens_for(rec, meta.get(t, (None,))[0], t) for t, rec in unl.items()}
json.dump(TG, open(HERE / 'out/tokens_gold.json', 'w')); json.dump(TM, open(HERE / 'out/tokens_machine_dev.json', 'w'))
json.dump(TU, open(HERE / 'out/tokens_claude_unl.json', 'w'))


def table(T, frame, label_col='root_code'):
    out = {}
    base = frame[label_col].value_counts(normalize=True)
    for tok in sorted({k for v in T.values() for k in v if not k.startswith('first_')}):
        rows = frame[[tok in T.get(t, {}) for t in frame.trace_id]]
        if len(rows) == 0:
            continue
        cs = codes_of(tok)
        prec = float(rows[label_col].isin(cs).mean())
        out[tok] = dict(fires=int(len(rows)), precision=round(prec, 3), base_rate=round(float(sum(base.get(c, 0) for c in cs)), 3),
                        none_share=round(float((rows[label_col] == 'NONE').mean()), 3),
                        top_codes=rows[label_col].value_counts().head(4).to_dict())
    return out


res = dict(gold_all_train=table(TG, tr), machine_dev=table(TM, dev))
ul = pd.DataFrame([(t, meta[t][1]) for t in TU if t in meta], columns=['trace_id', 'claude_code'])
res['claude_unl_vs_claude_label'] = table(TU, ul, 'claude_code')
res['coverage'] = dict(gold_rows_any_mal=round(np.mean([any(k.startswith('mal_') for k in v) for v in TG.values()]), 3),
                       gold_rows_any_path=round(np.mean([('path_ANS-01' in v) for v in TG.values()]), 3),
                       mdev_rows_any_mal=round(np.mean([any(k.startswith('mal_') for k in v) for v in TM.values()]), 3),
                       mdev_rows_any_path=round(np.mean([('path_ANS-01' in v) for v in TM.values()]), 3),
                       n_unl=len(TU))
json.dump(res, open(HERE / 'malrules.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
