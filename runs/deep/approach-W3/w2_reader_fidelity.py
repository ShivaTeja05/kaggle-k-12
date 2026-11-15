"""W3 test 2: how faithful is each reader on the root span? Gold vs Qwen3-VL-8B machine transcript vs Claude Opus transcript (dev, 301 rows).
Also: does the machine reader 'repair' arithmetic the student got wrong (gold line flagged invalid by the sympy checker, machine has no invalid line)?"""
import json, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/symbolic')
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/approach-W3')
from common import load, ROOT, boot_ci
from symutil import root_lines, body, norm, map_span
tr, tax, classes, folds, dev, fit = load()
w1 = pd.read_csv(ROOT/'runs/deep/approach-W3/w1_rows.csv', keep_default_na=False).set_index('trace_id')
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
cl = {}
for l in open(ROOT/'runs/claude/dev_opus.jsonl'):
    j = json.loads(l); a = j.get('answer') or {}
    if a.get('transcription'): cl[j['trace_id']] = a['transcription']
NUM = re.compile(r'\d+(?:[.,]\d+)*')
def nums(s):
    out = []
    for m in NUM.findall(str(s)):
        v = m.replace('.', '').replace(',', '.') if re.fullmatch(r'\d{1,3}(\.\d{3})+', m) else m.replace(',', '.')
        try: out.append(str(float(v)).rstrip('0').rstrip('.'))
        except ValueError: pass
    return out
def lines(t): return [l for l in str(t).strip().strip('"').splitlines() if l.strip()]
def num_f1(g, t):
    G, T = Counter(nums(g)), Counter(nums(t)); tp = sum((G & T).values())
    return 2 * tp / max(1, sum(G.values()) + sum(T.values()))
def span_line_sim(span, t):
    ln, s = map_span(span, t); return s
def chk(path):
    out = {}
    for l in open(path):
        j = json.loads(l); out[j['trace_id']] = j
    return out
CG = chk(ROOT/'runs/deep/symbolic/out/gold_train.jsonl'); CM = chk(ROOT/'runs/deep/symbolic/out/machine_dev.jsonl'); CC = chk(ROOT/'runs/deep/symbolic/out/claude_dev.jsonl')
def n_invalid(j): return sum(int(x['invalid']) for x in j['lines']) if j else None
def n_checks(j): return sum(x['n_checks'] for x in j['lines']) if j else None
VERD = re.compile(r'\[(?:[^\]]*\b(?:benar|salah|correct|incorrect|wrong|right|answered)\b[^\]]*)\]', re.I)
rows = []
for r in dev.itertuples(index=False):
    rl, span = root_lines(r.misconceptions)
    g, m, c = str(r.transcription_steps), tx.get(r.trace_id, ''), cl.get(r.trace_id)
    rec = dict(trace_id=r.trace_id, q=r.question_id, truth=r.root_code, grp=w1.grp[r.trace_id])
    for k, t in [('m', m), ('c', c)]:
        if t is None: continue
        rec[f'{k}_numf1'] = num_f1(g, t)
        rec[f'{k}_verdict'] = bool(VERD.search(t))
        if span:
            rec[f'{k}_span_in'] = norm(span) in norm(''.join(body(l) for l in lines(t)))
            sn = Counter(nums(span)); T = Counter(nums(t))
            rec[f'{k}_spannum_recall'] = (sum((sn & T).values()) / sum(sn.values())) if sn else np.nan
            rec[f'{k}_span_sim'] = span_line_sim(span, t)
    rec['g_verdict'] = bool(VERD.search(g))
    gi, mi, ci_ = n_invalid(CG.get(r.trace_id)), n_invalid(CM.get(r.trace_id)), n_invalid(CC.get(r.trace_id))
    rec.update(g_inv=gi, m_inv=mi, c_inv=ci_, m_chk=n_checks(CM.get(r.trace_id)), c_chk=n_checks(CC.get(r.trace_id)))
    rec['m_repair'] = (gi or 0) > 0 and mi == 0
    rec['c_repair'] = (gi or 0) > 0 and ci_ == 0 if ci_ is not None else np.nan
    rec['m_newflag'] = (gi or 0) == 0 and (mi or 0) > 0
    rec['c_newflag'] = (gi or 0) == 0 and (ci_ or 0) > 0 if ci_ is not None else np.nan
    rows.append(rec)
d = pd.DataFrame(rows)
for c_ in ['m_span_in','c_span_in','m_verdict','c_verdict','g_verdict','m_repair','c_repair','m_newflag','c_newflag']: d[c_] = pd.to_numeric(d[c_].map(lambda v: np.nan if v is None or (isinstance(v,float) and np.isnan(v)) else float(v)))
print('claude transcripts available:', d.c_numf1.notna().sum(), 'of', len(d))
cols = ['m_numf1', 'c_numf1', 'm_span_in', 'c_span_in', 'm_spannum_recall', 'c_spannum_recall', 'm_span_sim', 'c_span_sim', 'm_verdict', 'c_verdict', 'g_verdict']
err = d.truth != 'NONE'
de = d[err].copy()
summ = de.groupby('grp')[cols].mean().round(3); summ['n'] = de.groupby('grp').size()
print('ERROR rows (root span exists):'); print(summ.to_string())
print('ALL error rows:', de[cols].mean().round(3).to_dict())
# repair rates among rows where gold has >=1 invalid arithmetic line
gi = d.g_inv.fillna(0) > 0
print('\nrows whose GOLD transcript has an invalid arithmetic line:', int(gi.sum()), '(error rows', int((gi & err).sum()), ')')
for g_ in ['M6', 'other miss', 'OK(top1 right)']:
    s = gi & (d.grp == g_)
    print(f'  {g_:15s} n={int(s.sum()):3d}  machine repairs (0 invalid) {d.m_repair[s].mean():.3f}   claude repairs {pd.to_numeric(d.c_repair[s]).mean():.3f}')
print('  ALL            n=%d  machine repairs %.3f  claude repairs %.3f' % (gi.sum(), d.m_repair[gi].mean(), pd.to_numeric(d.c_repair[gi]).mean()))
print('new invalid lines where gold has none: machine %.3f claude %.3f' % (d.m_newflag[~gi].mean(), pd.to_numeric(d.c_newflag[~gi]).mean()))
# paired comparison claude vs machine span recall on error rows, question-grouped CI
s = de.dropna(subset=['m_spannum_recall', 'c_spannum_recall'])
print('\nspan-number recall claude - machine (error rows w/ numeric span, n=%d):' % len(s), boot_ci(s.c_spannum_recall.to_numpy(), s.m_spannum_recall.to_numpy(), s.q.to_numpy()))
s6 = s[s.grp == 'M6']; print('   on M6 rows (n=%d):' % len(s6), boot_ci(s6.c_spannum_recall.to_numpy(), s6.m_spannum_recall.to_numpy(), s6.q.to_numpy()))
s2 = de.dropna(subset=['m_span_in', 'c_span_in'])
print('exact span present claude - machine (error rows n=%d):' % len(s2), boot_ci(s2.c_span_in.astype(float).to_numpy(), s2.m_span_in.astype(float).to_numpy(), s2.q.to_numpy()))
d.to_csv(ROOT/'runs/deep/approach-W3/w2_rows.csv', index=False)
json.dump(dict(error_rows_by_group=summ.reset_index().to_dict('records')), open(ROOT/'runs/deep/approach-W3/w2_reader_fidelity.json', 'w'), indent=1, default=str)
