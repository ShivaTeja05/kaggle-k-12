"""Literature-lens probe 2 (PAL / PoT / CAS-checking idea, CPU, aggregate only).
A deterministic arithmetic checker over transcript lines: finds 'a OP b = c' (OP in + - x * : / ^)
and flags lines where the stated result is wrong. Measures (a) how often the flag fires on each gold code
family, (b) whether the first flagged line matches the gold root step, and (c) cross-fitted fusion value
of boosting CAL-01 (and generic arithmetic codes) when the flag fires, on machine transcripts."""
import json, re, numpy as np, pandas as pd
from common import load, blend, boot_ci, ROOT
from k12.core import row_scores, topk, normalize
tr, tax, classes, folds, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
NUM = r'(-?\d+(?:[.,]\d+)?)'
OPS = {'+': lambda a, b: a + b, '-': lambda a, b: a - b, '−': lambda a, b: a - b, 'x': lambda a, b: a * b, '×': lambda a, b: a * b,
       '*': lambda a, b: a * b, '.': None, ':': lambda a, b: a / b if b else None, '/': lambda a, b: a / b if b else None,
       '÷': lambda a, b: a / b if b else None, '^': lambda a, b: a ** b if abs(b) < 10 else None}
PAT = re.compile(NUM + r'\s*([+\-−x×*:/÷^])\s*' + NUM + r'\s*=\s*' + NUM + r'(?![\d/^])')
def num(s):
    s = s.replace(',', '.')
    return float(s)
def check_line(t):
    bad = good = 0
    for m in PAT.finditer(t):
        a, op, b, c = m.groups()
        f = OPS.get(op)
        if f is None: continue
        try:
            v = f(num(a), num(b))
        except Exception:
            continue
        if v is None: continue
        if abs(v - num(c)) <= 1e-6 + 1e-3 * abs(v): good += 1
        else: bad += 1
    return bad, good
def lines(text):
    out = []
    for ln in str(text).split('\n'):
        if '|' in ln:
            k, t = ln.split('|', 1); out.append((k.strip(), t))
    return out
def scan(text):
    first_bad = None; nb = ng = 0
    for k, t in lines(text):
        b, g = check_line(t); nb += b; ng += g
        if b and first_bad is None: first_bad = k
    return nb, ng, first_bad
def root_step(ms):
    try:
        for m in json.loads(ms):
            if m.get('root'): return m.get('step')
    except Exception: pass
    return None
res = {}
for src in ['gold', 'machine']:
    recs = []
    for _, r in dev.iterrows():
        text = r.transcription_steps if src == 'gold' else tx.get(r.trace_id, '')
        nb, ng, fb = scan(text)
        recs.append(dict(code=r.root_code, fam=r.root_code.split('-')[0], bad=nb > 0, checked=(nb + ng) > 0, first_bad=fb, root=root_step(r.misconceptions)))
    d = pd.DataFrame(recs)
    out = {'rows_with_checkable_arith': int(d.checked.sum()), 'rows_flagged': int(d.bad.sum())}
    out['flag_rate_by_group'] = {
        'CAL-01': [round(float(d[d.code == 'CAL-01'].bad.mean()), 3), int((d.code == 'CAL-01').sum())],
        'NONE': [round(float(d[d.code == 'NONE'].bad.mean()), 3), int((d.code == 'NONE').sum())],
        'other_error': [round(float(d[(d.code != 'NONE') & (d.code != 'CAL-01')].bad.mean()), 3), int(((d.code != 'NONE') & (d.code != 'CAL-01')).sum())]}
    fl = d[d.bad]
    out['gold_code_dist_when_flagged_top8'] = fl.code.value_counts().head(8).to_dict()
    out['first_flag_line_equals_gold_root_step'] = round(float((fl.first_bad == fl.root).mean()), 3) if len(fl) else None
    res[src] = out
    if src == 'machine': dm = d
# cross-fitted fusion on machine transcripts: P' = P * exp(b * flag) for CAL-01 (and optionally OPS/INT/Dec families)
P = blend(dev.trace_id.tolist(), classes); y = dev.root_code.to_numpy()
base = row_scores(y, topk(P, classes))
flag = dm.bad.to_numpy().astype(float)
targets = {'CAL-01_only': ['CAL-01'], 'arith_codes': ['CAL-01', 'OPS-01', 'OPS-02', 'INT-04', 'Dec-02', 'OOO-01']}
qs = dev.question_id.unique(); rng = np.random.default_rng(2026); rng.shuffle(qs)
qf = {q: i % 5 for i, q in enumerate(qs)}; f = dev.question_id.map(qf).to_numpy()
grid = [0, .25, .5, 1, 1.5, 2, 3]
res['fusion'] = {}
for name, codes in targets.items():
    idx = [ci[c] for c in codes if c in ci]
    sc = {}
    for b in grid:
        L = np.log(P); L[:, idx] += b * flag[:, None]
        sc[b] = row_scores(y, topk(normalize(np.exp(L - L.max(1, keepdims=True))), classes))
    out = np.zeros(len(dev)); ch = []
    for k in range(5):
        bb = max(grid, key=lambda b: sc[b][f != k].mean()); ch.append(bb); out[f == k] = sc[bb][f == k]
    dlt, ci95 = boot_ci(out, base, dev.question_id.to_numpy())
    res['fusion'][name] = {'in_sample': {str(b): round(float(sc[b].mean()), 4) for b in grid}, 'crossfit': round(float(out.mean()), 4),
                           'delta': round(dlt, 4), 'ci95': [round(x, 4) for x in ci95], 'chosen': ch}
res['blend_ref'] = round(float(base.mean()), 4)
json.dump(res, open(ROOT/'runs/deep/literature/p2_arith_checker.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
