"""Refined checker: evaluate the whole numeric left-hand side of each '=' (maximal chain of numbers/ops/parens),
not a 2-operand window, to remove chain false positives. Same measurements as p2."""
import json, re, ast, operator as O, numpy as np, pandas as pd
from common import load, blend, boot_ci, ROOT
from k12.core import row_scores, topk, normalize
tr, tax, classes, folds, dev, fit = load()
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
SEG = re.compile(r'([0-9][0-9\s.,+\-−x×*:/÷()]*[0-9)])\s*=\s*(-?[0-9]+(?:[.,][0-9]+)?)(?![0-9/^a-zA-Z])')
BIN = {ast.Add: O.add, ast.Sub: O.sub, ast.Mult: O.mul, ast.Div: O.truediv}
def ev(n):
    if isinstance(n, ast.Expression): return ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in BIN: return BIN[type(n.op)](ev(n.left), ev(n.right))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
    raise ValueError
def norm_expr(s):
    s = s.replace('−', '-').replace('×', '*').replace('÷', '/').replace(':', '/')
    s = re.sub(r'(?<=\d)\s*x\s*(?=\d)', '*', s)
    s = re.sub(r'(?<=\d)\.(?=\d{3}(?!\d))', '', s)   # Indonesian thousands separator 1.000
    s = s.replace(',', '.')
    if re.search(r'[a-zA-Z]', s): raise ValueError
    return s
def check_line(t):
    bad = good = 0
    for m in SEG.finditer(t):
        lhs, rhs = m.groups()
        if not re.search(r'[+\-−x×*:/÷]', lhs.strip().lstrip('-')): continue
        try:
            v = ev(ast.parse(norm_expr(lhs), mode='eval')); c = float(norm_expr(rhs))
        except Exception:
            continue
        if abs(v - c) <= 1e-6 + 2e-3 * abs(v) or abs(round(v) - c) < 1e-9 and abs(v) > 50: good += 1
        else: bad += 1
    return bad, good
def scan(text):
    nb = ng = 0; fb = None
    for ln in str(text).split('\n'):
        if '|' not in ln: continue
        k, t = ln.split('|', 1); b, g = check_line(t); nb += b; ng += g
        if b and fb is None: fb = k.strip()
    return nb, ng, fb
def root_step(ms):
    try:
        for m in json.loads(ms):
            if m.get('root'): return m.get('step')
    except Exception: pass
res = {}
for src in ['gold', 'machine']:
    d = pd.DataFrame([dict(code=r.root_code, **dict(zip(['nb', 'ng', 'fb'], scan(r.transcription_steps if src == 'gold' else tx.get(r.trace_id, '')))), root=root_step(r.misconceptions)) for _, r in dev.iterrows()])
    d['bad'] = d.nb > 0
    res[src] = {'rows_checkable': int(((d.nb + d.ng) > 0).sum()), 'rows_flagged': int(d.bad.sum()),
                'flag_rate_CAL01': [round(float(d[d.code == 'CAL-01'].bad.mean()), 3), int((d.code == 'CAL-01').sum())],
                'flag_rate_NONE': round(float(d[d.code == 'NONE'].bad.mean()), 3),
                'flag_rate_other_err': round(float(d[(d.code != 'NONE') & (d.code != 'CAL-01')].bad.mean()), 3),
                'flagged_code_top6': d[d.bad].code.value_counts().head(6).to_dict(),
                'first_flag_eq_root_step': round(float((d[d.bad].fb == d[d.bad].root).mean()), 3) if d.bad.any() else None,
                'share_flagged_rows_with_gold_error': round(float((d[d.bad].code != 'NONE').mean()), 3) if d.bad.any() else None}
    if src == 'machine': dm = d
# gold misconception spans tagged CAL-01 anywhere (root or not) - how often are slips non-root?
cal_any = cal_root = 0
for ms in dev.misconceptions:
    try: L = json.loads(ms)
    except Exception: continue
    if any(m.get('category') == 'CAL-01' for m in L): cal_any += 1
    if any(m.get('category') == 'CAL-01' and m.get('root') for m in L): cal_root += 1
res['dev_rows_with_any_CAL01_span'] = cal_any; res['dev_rows_with_root_CAL01'] = cal_root
P = blend(dev.trace_id.tolist(), classes); y = dev.root_code.to_numpy(); base = row_scores(y, topk(P, classes))
flag = dm.bad.to_numpy().astype(float); ci_ = classes.index('CAL-01'); none_ = classes.index('NONE')
qs = dev.question_id.unique(); rng = np.random.default_rng(2026); rng.shuffle(qs); qf = {q: i % 5 for i, q in enumerate(qs)}; f = dev.question_id.map(qf).to_numpy()
grid = [(a, b) for a in [0, .5, 1, 2] for b in [0, -.5, -1]]
sc = {}
for a, b in grid:
    L = np.log(P); L[:, ci_] += a * flag; L[:, none_] += b * flag
    sc[(a, b)] = row_scores(y, topk(normalize(np.exp(L - L.max(1, keepdims=True))), classes))
out = np.zeros(len(dev)); ch = []
for k in range(5):
    bb = max(grid, key=lambda g: sc[g][f != k].mean()); ch.append(bb); out[f == k] = sc[bb][f == k]
dlt, c95 = boot_ci(out, base, dev.question_id.to_numpy())
res['fusion_CAL_up_NONE_down'] = {'in_sample_best': max(((str(g), round(float(s.mean()), 4)) for g, s in sc.items()), key=lambda t: t[1]),
                                  'crossfit': round(float(out.mean()), 4), 'delta': round(dlt, 4), 'ci95': [round(x, 4) for x in c95], 'chosen': [str(c) for c in ch]}
json.dump(res, open(ROOT/'runs/deep/literature/p2b_arith_checker_chain.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
