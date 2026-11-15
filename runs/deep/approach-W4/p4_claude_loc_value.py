"""Does correct localisation go with correct codes (Claude dev, no API)? Split Claude MAP on dev error rows by
whether its root_line equals the gold root (fuzzy-mapped), overall and on rows with an annotated secondary code."""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT/'runs/deep/approach-W4'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'runs/arch/label-structure'), str(ROOT/'runs/deep/symbolic')]
from common import load, blend
from k12.core import row_scores
from symutil import root_lines, map_span
tr, tax, classes, folds, dev, fit = load()
first = lambda s: (int(re.findall(r'L(\d+)', str(s))[0]) if re.findall(r'L(\d+)', str(s or '')) else None)
A = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(ROOT/'runs/claude/dev_opus.jsonl')}
ids = dev.trace_id.tolist(); P = blend(ids, classes)
out = {'all': {'loc': [], 'miss': []}, 'sec': {'loc': [], 'miss': []}}
for i, r in enumerate(dev.itertuples()):
    if r.root_code == 'NONE': continue
    a = A.get(r.trace_id, {}); s, span = root_lines(r.misconceptions)
    tgt, sc = map_span(span, a.get('transcription', '')) if s else (None, 0)
    cl = first(a.get('root_line'))
    if tgt is None or sc < .6 or cl is None: continue
    sC = row_scores([r.root_code], [[a.get(k) for k in ('first_code', 'second_code', 'third_code')]])[0]
    sB = row_scores([r.root_code], [[classes[j] for j in np.argsort(-P[i])[:3]]])[0]
    try: sec = any(not m.get('root') and m.get('category') not in (None, r.root_code) for m in json.loads(r.misconceptions))
    except Exception: sec = False
    k = 'loc' if cl == tgt else 'miss'
    out['all'][k].append((sC, sB))
    if sec: out['sec'][k].append((sC, sB))
res = {g: {k: {'n': len(v), 'claude': round(float(np.mean([x[0] for x in v])), 3), 'blend': round(float(np.mean([x[1] for x in v])), 3)}
           for k, v in d.items() if v} for g, d in out.items()}
json.dump(res, open(OUT/'p4_claude_loc_value.json', 'w'), indent=1); print(json.dumps(res))
