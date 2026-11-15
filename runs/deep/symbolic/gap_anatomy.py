"""Where could symbolic checking matter?  Dev error rows by category (seen pair / unseen pair / unseen question):
blend MAP@3, Claude MAP@3, symbolic flag coverage and localisation hit on machine transcripts, and the
share of root codes whose nature is 'computational' (detectable by checking arithmetic)."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load, blend
from k12.core import row_scores, topk
from symutil import root_lines, map_span, first_of

tr, tax, classes, folds, dev, fit = load()
nature = dict(zip(tax.id, tax.nature))
# codes whose definition is an arithmetic/algebraic slip that a step checker can in principle see
COMPUTATIONAL = {'OPS-01', 'OPS-02', 'OOO-01', 'OOO-02', 'INT-02', 'INT-03', 'INT-04', 'FRA-01', 'FRA-03', 'FRA-04', 'FRA-05',
                 'Dec-02', 'EXP-01', 'EXP-02', 'EXP-03', 'EXP-05', 'EXP-07', 'RAT-02', 'ALG-03', 'ALG-04', 'ALG-05', 'ALG-06',
                 'CAL-01', 'LOG-02', 'QUA-02', 'SIM-01'}
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
pts_b = row_scores(y, topk(blend(ids, classes), classes))
cl = {r['trace_id']: r['answer'] for r in map(json.loads, open(ROOT / 'runs/claude/dev_opus.jsonl')) if r.get('answer')}
pts_c = row_scores(y, [[cl[t]['first_code'], cl[t]['second_code'], cl[t]['third_code']] if t in cl else [] for t in ids])
mdev = {r['trace_id']: r for r in map(json.loads, open(HERE / 'out/machine_dev.jsonl'))}
txm = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))
fit_pairs = set(zip(fit.question_id, fit.root_code)); fit_q = set(fit.question_id)
cat = np.array(['NONE' if c == 'NONE' else 'seen_pair' if (q, c) in fit_pairs else 'unseen_pair' if q in fit_q else 'unseen_q'
                for q, c in zip(dev.question_id, y)])
flag = np.array([first_of(mdev[t]['lines'], {'arith', 'eq', 'cont'}) is not None for t in ids])
comp = np.array([c in COMPUTATIONAL for c in y])
hit = np.full(len(ids), np.nan)
for i, r in enumerate(dev.itertuples()):
    g, span = root_lines(r.misconceptions)
    if not g:
        continue
    ln, sc = map_span(span, txm[r.trace_id])
    if ln is None or sc < .6:
        continue
    f = first_of(mdev[r.trace_id]['lines'], {'arith', 'eq', 'cont'})
    hit[i] = float(f == ln) if f is not None else 0.0
res = {}
for k in ['seen_pair', 'unseen_pair', 'unseen_q', 'NONE']:
    m = cat == k
    res[k] = dict(n=int(m.sum()), blend_map3=round(float(pts_b[m].mean()), 3), claude_map3=round(float(pts_c[m].mean()), 3),
                  lost_points_vs_claude=round(float((pts_c[m] - pts_b[m]).sum()), 2),
                  symbolic_flag_rate=round(float(flag[m].mean()), 3),
                  computational_code_share=round(float(comp[m].mean()), 3) if k != 'NONE' else None,
                  symbolic_hits_root_line=round(float(np.nanmean(hit[m])), 3) if np.isfinite(hit[m]).any() else None)
err = cat != 'NONE'
res['err_rows_computational_vs_not'] = {
    nm: dict(n=int((err & mm).sum()), blend=round(float(pts_b[err & mm].mean()), 3), claude=round(float(pts_c[err & mm].mean()), 3),
             lost_vs_claude=round(float((pts_c - pts_b)[err & mm].sum()), 2), flag_rate=round(float(flag[err & mm].mean()), 3))
    for nm, mm in [('computational', comp), ('not_computational', ~comp)]}
res['unseen_pair_codes_top'] = pd.Series(y[cat == 'unseen_pair']).value_counts().head(15).to_dict()
res['nature_of_root_codes_dev_err'] = pd.Series([nature.get(c, '') for c in y[err]]).value_counts().to_dict()
json.dump(res, open(HERE / 'gap_anatomy.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
