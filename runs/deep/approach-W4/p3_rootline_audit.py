"""Audit of ROOT_LINE supervision (CPU). Is Claude's root_line biased toward late / last lines relative to gold?
(i) gold train error rows: relative position of the root line; (ii) Claude train-explain root_line vs gold step;
(iii) Claude 26k pool error rows: relative position; (iv) Claude dev root_line vs gold (fuzzy-mapped span)."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT/'runs/deep/approach-W4'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'runs/arch/label-structure'), str(ROOT/'runs/deep/symbolic')]
from common import load
from symutil import root_lines, map_span
tr, tax, classes, folds, dev, fit = load()
ids_of = lambda tx: [int(m.group(1)) for m in (re.match(r'\s*L(\d+)', l) for l in str(tx).splitlines() if l.strip()) if m]
first = lambda s: (int(re.findall(r'L(\d+)', str(s))[0]) if re.findall(r'L(\d+)', str(s or '')) else None)
def pos_stats(pairs):  # (root, line_ids)
    rel = [(L.index(r)) / max(len(L) - 1, 1) for r, L in pairs if r in L and len(L) > 1]
    last = [r == L[-1] for r, L in pairs if r in L]
    return {'n': len(last), 'share_last_line': round(float(np.mean(last)), 3), 'mean_rel_pos': round(float(np.mean(rel)), 3)}
res = {}
g = []
for r in tr.itertuples():
    if r.root_code == 'NONE': continue
    s, _ = root_lines(r.misconceptions)
    if s: g.append((min(s), ids_of(r.transcription_steps)))
res['gold_train'] = pos_stats(g)
ex = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(ROOT/'runs/claude/train_explain_opus.jsonl')}
hit = []; trs = tr.set_index('trace_id')
for t, a in ex.items():
    if t not in trs.index or trs.loc[t, 'root_code'] == 'NONE': continue
    s, span = root_lines(trs.loc[t, 'misconceptions']); cl = first(a.get('root_line'))
    if not s or cl is None: continue
    tgt, sc = map_span(span, a.get('transcription', ''))
    if tgt is None or sc < .6: continue
    hit.append((cl == tgt, cl > tgt, cl < tgt))
h = np.array(hit)
res['claude_train_explain_vs_gold'] = {'n_mapped': len(h), 'exact': round(float(h[:, 0].mean()), 3),
                                       'claude_later': round(float(h[:, 1].mean()), 3), 'claude_earlier': round(float(h[:, 2].mean()), 3)}
pool = []
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    x = json.loads(l); a = x.get('answer') or {}
    if not a or a.get('first_code') in (None, 'NONE'): continue
    cl = first(a.get('root_line'))
    if cl is not None: pool.append((cl, ids_of(a.get('transcription', ''))))
res['claude_pool_error_rows'] = pos_stats(pool)
dv = []; dpos = []
for l in open(ROOT/'runs/claude/dev_opus.jsonl'):
    x = json.loads(l); a = x['answer']; t = x['trace_id']
    if t not in trs.index or trs.loc[t, 'root_code'] == 'NONE': continue
    s, span = root_lines(trs.loc[t, 'misconceptions']); cl = first(a.get('root_line'))
    tgt, sc = map_span(span, a.get('transcription', '')) if s else (None, 0)
    if cl is None or tgt is None or sc < .6: continue
    dv.append((cl == tgt, cl > tgt, cl < tgt)); dpos.append((cl, ids_of(a.get('transcription', ''))))
d = np.array(dv)
res['claude_dev_vs_gold'] = {'n_mapped': len(d), 'exact': round(float(d[:, 0].mean()), 3),
                             'claude_later': round(float(d[:, 1].mean()), 3), 'claude_earlier': round(float(d[:, 2].mean()), 3)}
res['claude_dev_error_rows_position'] = pos_stats(dpos)
json.dump(res, open(OUT/'p3_rootline_audit.json', 'w'), indent=1); print(json.dumps(res, indent=1))

# ---- why is the pool last-line heavy? split by code; and what the v5 student actually trains on
pc = Counter(); pl = Counter()
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    x = json.loads(l); a = x.get('answer') or {}
    if not a or a.get('first_code') in (None, 'NONE'): continue
    cl = first(a.get('root_line')); L = ids_of(a.get('transcription', ''))
    if cl is None or cl not in L: continue
    pc[a['first_code']] += 1; pl[a['first_code']] += int(cl == L[-1])
gc = Counter(); gl = Counter()
for r in tr.itertuples():
    if r.root_code == 'NONE': continue
    s, _ = root_lines(r.misconceptions); L = ids_of(r.transcription_steps)
    if s and min(s) in L: gc[r.root_code] += 1; gl[r.root_code] += int(min(s) == L[-1])
top = [c for c, _ in pc.most_common(12)]
res['pool_last_line_by_code'] = {c: [pc[c], round(pl[c]/pc[c], 3), gc[c], round(gl[c]/max(gc[c], 1), 3)] for c in top}
ex_ans = [c for c in pc if not c.startswith('ANS')]
res['pool_last_line_excl_ANS'] = round(sum(pl[c] for c in ex_ans)/sum(pc[c] for c in ex_ans), 3)
res['gold_last_line_excl_ANS'] = round(sum(gl[c] for c in gc if not c.startswith('ANS'))/sum(gc[c] for c in gc if not c.startswith('ANS')), 3)
# code-mix-reweighted pool last-line share (reweight pool codes to gold code mix)
w = {c: gc[c]/sum(gc.values()) for c in gc}
res['pool_last_line_reweighted_to_gold_code_mix'] = round(sum(w[c]*pl[c]/pc[c] for c in w if pc[c]), 3)
v5 = Counter()
for l in open(ROOT/'data/v5/cand_train.jsonl'):
    x = json.loads(l)
    if x['source'] != 'claude' or x['target'].startswith('ROOT_LINE: -'): continue
    rl = int(re.match(r'ROOT_LINE: L(\d+)', x['target']).group(1)) if re.match(r'ROOT_LINE: L(\d+)', x['target']) else None
    L = ids_of(x['prompt'].split('Student working (line by line):\n')[1].split('\n\nCandidate')[0])
    if rl in L: v5['n'] += 1; v5['last'] += int(rl == L[-1])
res['v5_claude_rows_last_line_share'] = [v5['n'], round(v5['last']/max(v5['n'], 1), 3)]
json.dump(res, open(OUT/'p3_rootline_audit.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('pool_last_line_by_code', 'pool_last_line_excl_ANS', 'gold_last_line_excl_ANS', 'pool_last_line_reweighted_to_gold_code_mix', 'v5_claude_rows_last_line_share')}))

# ---- length control: last-line share by number of lines (gold train vs Claude pool vs v5 Claude rows)
def by_len(pairs):
    out = {}
    for lo, hi in ((1, 3), (4, 6), (7, 10), (11, 99)):
        sel = [r == L[-1] for r, L in pairs if r in L and lo <= len(L) <= hi]
        out[f'{lo}-{hi}'] = [len(sel), round(float(np.mean(sel)), 3) if sel else None]
    return out
res['len_control'] = {'gold_train': by_len(g), 'claude_pool': by_len(pool),
                      'mean_lines_gold': round(float(np.mean([len(L) for _, L in g])), 2),
                      'mean_lines_pool': round(float(np.mean([len(L) for _, L in pool])), 2)}
json.dump(res, open(OUT/'p3_rootline_audit.json', 'w'), indent=1); print(json.dumps(res['len_control']))
