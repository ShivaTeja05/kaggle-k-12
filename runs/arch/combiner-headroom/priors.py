"""Question/topic priors: gold (fit rows only) and Claude-labelled unlabelled photos (no dev/test photos)."""
import numpy as np, pandas as pd
from common import *
from k12.baseline import HierarchicalPrior

def gold_prior(a=2.0, b=8.0, frame=None, rows=None):
    frame = fit if frame is None else frame
    rows = dev if rows is None else rows
    return HierarchicalPrior(classes, a, b).fit(frame).predict_proba(rows)

_lab = None
def claude_unl():
    global _lab
    if _lab is None:
        lab = pd.read_csv(R+'data/claude_dataset/labels.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
        lab = lab[(lab.split == 'unlabelled')]
        lab = lab[lab.first_code.isin(ci)].drop_duplicates('trace_id', keep='last')
        _lab = lab
    return _lab

def claude_prior(w2=0.5, w3=0.25, a=5.0, rows=None, base=None):
    """P(code | question) from Claude first/second/third codes on unlabelled photos, shrunk to `base` (row-wise)."""
    rows = dev if rows is None else rows
    lab = claude_unl()
    counts = {}
    for q, g in lab.groupby('question_id'):
        v = np.zeros(K)
        for col, w in [('first_code', 1.), ('second_code', w2), ('third_code', w3)]:
            for c in g[col]:
                if c in ci: v[ci[c]] += w
        counts[q] = v
    base = gold_prior(rows=rows) if base is None else base
    out = []
    for i, q in enumerate(rows.question_id):
        v = counts.get(q, np.zeros(K))
        out.append((v + a*base[i]) / (v.sum() + a))
    return np.asarray(out)
