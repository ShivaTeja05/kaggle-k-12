"""Reference: how often does Claude Opus' own root_line hit the gold root (both mapped to Claude's transcript)?"""
import sys, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load
from symutil import root_lines, map_span, first_of
tr, tax, classes, folds, dev, fit = load()
cl = {r['trace_id']: r['answer'] for r in map(json.loads, open(ROOT / 'runs/claude/dev_opus.jsonl')) if r.get('answer')}
cdev = {r['trace_id']: r for r in map(json.loads, open(HERE / 'out/claude_dev.jsonl'))}
hit = cov = n = 0; hits_sym = []; hits_cl = []
for r in dev.itertuples():
    g, span = root_lines(r.misconceptions)
    if not g or r.trace_id not in cl:
        continue
    a = cl[r.trace_id]; ln, sc = map_span(span, a['transcription'])
    if ln is None or sc < .6:
        continue
    n += 1
    m = re.match(r'\s*L(\d+)', a.get('root_line') or '')
    if m:
        cov += 1; hit += int(m.group(1)) == ln
    L = cdev[r.trace_id]['lines']
    s = first_of(L, {'arith', 'eq', 'cont'}) or (L[-1]['line'] if L else None)
    hits_sym.append(s == ln); hits_cl.append(bool(m) and int(m.group(1)) == ln)
res = dict(n_mapped=n, claude_rootline_coverage=round(cov / n, 3), claude_hit_given_flag=round(hit / max(cov, 1), 3),
           claude_hit_overall=round(hit / n, 3), symbolic_core_cont_else_last_hit_same_rows=round(float(np.mean(hits_sym)), 3))
print(res); json.dump(res, open(HERE / 'claude_localisation.json', 'w'), indent=1)
