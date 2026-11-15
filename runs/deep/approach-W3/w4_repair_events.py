"""W3 test 4: at the level of single wrong student equalities 'A = B' (gold line, sympy says A != B),
does the machine transcript keep the student's wrong B, or replace it with the arithmetically correct value of A
(an LM-prior 'repair')? Same for Claude's transcript. Dev rows, plus fold-wise split by error vs NONE.
Run with runs/deep/symbolic/venv/bin/python (sympy)."""
import json, re, sys
from collections import Counter
import pandas as pd
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/symbolic')
from symutil import body
import sympy as sp
ROOT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'
w1 = pd.read_csv(f'{ROOT}/runs/deep/approach-W3/w1_rows.csv', keep_default_na=False).set_index('trace_id')
dev_ids = set(w1.index)
tx = json.load(open(f'{ROOT}/runs/pod/tx/tx_dev.json'))
cl = {}
for l in open(f'{ROOT}/runs/claude/dev_opus.jsonl'):
    j = json.loads(l); a = j.get('answer') or {}
    if a.get('transcription'): cl[j['trace_id']] = a['transcription']
gold = {r['trace_id']: r['transcription_steps'] for r in pd.read_csv(f'{ROOT}/data/raw/train.csv', keep_default_na=False).to_dict('records')}
NUM = re.compile(r'\d+(?:[.,]\d+)*')
def nums(s):
    out = set()
    for m in NUM.findall(str(s)):
        v = m.replace('.', '').replace(',', '.') if re.fullmatch(r'\d{1,3}(\.\d{3})+', m) else m.replace(',', '.')
        try: out.add(round(float(v), 4))
        except ValueError: pass
    return out
def val(e):
    try:
        x = sp.sympify(e.replace('sqrt', 'sqrt'), rational=True); x = complex(sp.N(x))
        return round(x.real, 4) if abs(x.imag) < 1e-9 else None
    except Exception: return None
def pretty(v): return {round(v, 4), round(abs(v), 4)}
ev = []
for l in open(f'{ROOT}/runs/deep/symbolic/out/gold_train.jsonl'):
    j = json.loads(l); t = j['trace_id']
    if t not in dev_ids: continue
    g_all = nums(gold[t])
    for x in j['lines']:
        for rs in x['reasons']:
            m = re.fullmatch(r'(.+?) != (.+)', rs)
            if not m or 'contradicts' in rs: continue
            a, b = val(m.group(1)), val(m.group(2))
            if a is None or b is None or abs(a - b) < 1e-6: continue
            # only clean events: correct value of A never written by the student anywhere
            if pretty(a) & g_all: continue
            for k, T in [('machine', tx.get(t, '')), ('claude', cl.get(t, ''))]:
                tn = nums(T)
                has_b, has_a = bool(pretty(b) & tn), bool(pretty(a) & tn)
                kind = 'kept_wrong' if has_b and not has_a else 'repaired' if has_a and not has_b else 'both' if has_a else 'neither'
                ev.append(dict(trace_id=t, line=x['line'], pair=rs, A=a, B=b, reader=k, kind=kind, grp=w1.grp[t], truth=w1.truth[t]))
d = pd.DataFrame(ev)
print('clean wrong-equality events on dev gold:', d[d.reader == 'machine'].shape[0], 'rows', d.trace_id.nunique())
tab = pd.crosstab([d.reader, d.grp], d.kind); tab['repair_share'] = (tab.get('repaired', 0) + tab.get('both', 0)) / tab.sum(1)
print(tab.to_string())
print(pd.crosstab(d.reader, d.kind, normalize='index').round(3).to_string())
print('\nexamples machine repaired:'); print(d[(d.reader == 'machine') & (d.kind == 'repaired')][['trace_id', 'line', 'pair', 'grp']].to_string())
d.to_csv(f'{ROOT}/runs/deep/approach-W3/w4_events.csv', index=False)
json.dump({'by_reader': pd.crosstab(d.reader, d.kind).to_dict(), 'by_reader_grp': {f'{a}|{b}': v for (a, b), v in pd.crosstab([d.reader, d.grp], d.kind).T.to_dict().items()}},
          open(f'{ROOT}/runs/deep/approach-W3/w4_repair_events.json', 'w'), indent=1)
