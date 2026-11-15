"""Do our models fail on the same rows where Claude mislocates the root line? (difficulty control)"""
import sys, json, re
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/reading-impact')
import numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT + 'scripts')
from evaluate_claude import claude_scores
R = pd.read_csv(ROOT + 'runs/arch/reading-impact/per_row_reading.csv', keep_default_na=False, dtype={'trace_id': str})
P['claude'] = claude_scores(ROOT + 'runs/claude/dev_opus.jsonl', ids, classes)[0]
ref = rs(geo([('32_tx', .35), ('8_tx_e3', .30), ('vl', .35)]))
isroot = (R.root_step != '').values
def contains(rs_, pred):
    m = re.match(r'L(\d+)(?:\s*-\s*L?(\d+))?', rs_); p = re.match(r'L(\d+)', str(pred))
    if not m or not p: return False
    a = int(m.group(1)); b = int(m.group(2) or a); return a <= int(p.group(1)) <= b
loc = np.array([contains(a, b) if r else False for a, b, r in zip(R.root_step, R.c_root_line_pred, isroot)])
res = {}
for n, m in (('claude localised', isroot & loc), ('claude not localised', isroot & ~loc)):
    res[n] = dict(n=int(m.sum()), blend=round(float(ref[m].mean()), 3), claude=round(float(rs(P['claude'])[m].mean()), 3),
                  s32_gold=round(float(rs(P['32_gold'])[m].mean()), 3), s100_gold=round(float(rs(P['s100_gold'])[m].mean()), 3),
                  claude_minus_blend_points=round(float((rs(P['claude']) - ref)[m].sum()), 1))
res['claude_minus_blend_points_total_root_rows'] = round(float((rs(P['claude']) - ref)[isroot].sum()), 1)
res['claude_minus_blend_points_NONE_rows'] = round(float((rs(P['claude']) - ref)[~isroot].sum()), 1)
# among rows machine read root span exactly: Claude vs blend gap (pure non-reading gap)
ok = isroot & (pd.to_numeric(R.m_span_dist, errors='coerce') == 0).values
res['gap_on_rows_span_read_exactly'] = dict(n=int(ok.sum()), points=round(float((rs(P['claude']) - ref)[ok].sum()), 1),
                                           map_units=round(float((rs(P['claude']) - ref)[ok].sum() / len(ids)), 4))
print(json.dumps(res, indent=1))
json.dump(res, open(ROOT + 'runs/arch/reading-impact/localisation.json', 'w'), indent=1)
