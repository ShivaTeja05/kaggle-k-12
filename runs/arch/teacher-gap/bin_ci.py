"""Paired question-bootstrap CIs for Claude-minus-blend within each coverage bin, and the share of the total gap per bin."""
import json
import numpy as np
import pandas as pd
from k12.core import read_csv, make_folds
OUT = 'runs/arch/teacher-gap/'
train = read_csv('data/raw/train.csv'); folds = make_folds(train); fit = train.loc[folds != 0]
d = pd.read_csv(OUT + 'rows.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
qc = {q: set(g.root_code) for q, g in fit.groupby('question_id')}
d['bin'] = np.where(d.truth == 'NONE', 'NONE', np.where(~d.question_id.isin(qc), 'q_unseen', np.where([t in qc.get(q, set()) for t, q in zip(d.truth, d.question_id)], 'code_seen_for_q', 'code_unseen_for_q')))
def boot(v, q, n=4000):
    g = pd.DataFrame({'q': q, 'd': v}).groupby('q').d.agg(['sum', 'size']).to_numpy()
    s = np.random.default_rng(2026).integers(0, len(g), (n, len(g)))
    b = g[s, 0].sum(1) / g[s, 1].sum(1)
    return [round(float(x), 3) for x in np.quantile(b, [.025, .975])]
res = {}
delta = (d.s_claude - d.s_blend).to_numpy()
for b, v in d.groupby('bin'):
    dd = (v.s_claude - v.s_blend).to_numpy()
    res[b] = {'rows': len(v), 'blend': round(v.s_blend.mean(), 3), 'claude': round(v.s_claude.mean(), 3), 'claude_minus_blend': round(dd.mean(), 3),
              'ci95': boot(dd, v.question_id.to_numpy()), 'share_of_total_gap': round(float(dd.sum() / delta.sum()), 3)}
json.dump(res, open(OUT + 'bin_ci.json', 'w'), indent=1); print(json.dumps(res, indent=1))
