"""W4 test T2 (CPU): approach A's data gate on the existing v5 candidate data, without rebuilding the GPU run.
(i)  gold-row secondary coverage if the annotated non-root codes are passed as `extra` (train_candidates puts extra
     right after truth and NONE, so coverage is by construction; verified by counting slots).
(ii) positional importance weights on Claude error rows -> weighted last-line share per length bucket vs gold train.
(iii) Kish ESS of error rows with focal gamma=2 (shortcut lens s9 recipe, copied) x positional weights; LR_unseen.
(iv) ANS-01 share of weighted Claude error rows vs current.
Writes runs/deep/test-W4/t2_gate.json only."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT / 'runs/deep/test-W4'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(ROOT / 'runs/deep/shortcut'), str(ROOT / 'runs/deep/symbolic')]
from sc_common import load, claude_unl
from symutil import root_lines
tr, tax, classes, dev, fit = load()
ids_of = lambda tx: [int(m.group(1)) for m in (re.match(r'\s*L(\d+)', l) for l in str(tx).splitlines() if l.strip()) if m]
BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 999)); bname = lambda n: next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= n <= hi)
GOLD_SHARE = [0.421, 0.211, 0.201, 0.112]   # gold train last-line share by bucket (p3, recomputed in t1)
# ---- focal p_q (s9 recipe, verbatim logic)
cl, unl = claude_unl(); cl_ok = cl[~cl.trace_id.isin(set(dev.trace_id))]
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
qcount = {}
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount.setdefault(q, Counter())[c] += 1
g = Counter(fit.root_code) + Counter(cl_ok.root_code); tot = sum(g.values())
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))
L = 'ABCDEFGHIJ'
trs = tr.set_index('trace_id')
rows = []
for line in open(ROOT / 'data/v5/cand_train.jsonl'):
    x = json.loads(line); truth = x['codes'][L.index(x['target'][-1])]
    cnt = Counter(qcount.get(qmap[x['trace_id']], Counter())); cnt[own.get(x['trace_id'])] -= 1
    n = sum(v for v in cnt.values() if v > 0)
    p = (max(cnt.get(truth, 0), 0) + 2 * g[truth] / tot) / (n + 2)
    m = re.match(r'ROOT_LINE: L(\d+)', x['target']); rl = int(m.group(1)) if m else None
    Ls = ids_of(x['prompt'].split('Student working (line by line):\n')[1].split('\n\nCandidate')[0])
    sec = []
    if x['source'] == 'gold' and truth != 'NONE':
        ms = json.loads(trs.at[x['trace_id'], 'misconceptions'] or '[]')
        sec = list(dict.fromkeys(a['category'] for a in ms if not a.get('root') and a.get('category') not in (truth, None, '')))
    rows.append(dict(src=x['source'], none=truth == 'NONE', truth=truth, w0=float(x['weight']), p=p,
                     seen_t=cnt.get(truth, 0) > 0, seen_d=[cnt.get(c, 0) > 0 for c in x['codes'] if c not in (truth, 'NONE')],
                     rl=rl, L=Ls, codes=x['codes'], sec=sec))
res = {}
# ---- (i) secondary coverage
gs = [r for r in rows if r['src'] == 'gold' and r['sec']]
cur = sum(sum(c in r['codes'] for c in r['sec'][:3]) for r in gs); totsec = sum(len(r['sec'][:3]) for r in gs)
slots = Counter(min(len(r['sec'][:3]), 8) for r in gs)
res['i_secondary_coverage'] = {'gold_rows_with_secondary': len(gs), 'secondary_codes(<=3/row)': totsec,
                               'current_in_candidates': cur, 'current_share': round(cur / totsec, 3),
                               'with_extra': 'by construction 1.0: extra is added right after truth and NONE (<=5 of k=10 slots)',
                               'unknown_codes': sum(c not in classes for r in gs for c in r['sec']),
                               'rows_by_n_secondary': dict(slots)}
# ---- (ii) positional weights on Claude error rows
ce = [r for r in rows if r['src'] == 'claude' and not r['none'] and r['rl'] is not None and r['rl'] in r['L']]
for r in ce:
    r['b'] = bname(len(r['L'])); r['last'] = r['rl'] == r['L'][-1]
def pos_weights(rs, exclude_ans=False, clip=(0.3, 3.0)):
    sub = [r for r in rs if not (exclude_ans and r['truth'].startswith('ANS'))]
    pool = {b: np.average([r['last'] for r in sub if r['b'] == b], weights=[r['w0'] for r in sub if r['b'] == b]) for b in range(4)}
    w = {}
    for r in rs:
        if exclude_ans and r['truth'].startswith('ANS'):
            w[id(r)] = 1.0; continue
        t, s = GOLD_SHARE[r['b']], pool[r['b']]
        w[id(r)] = float(np.clip(t / s if r['last'] else (1 - t) / (1 - s), *clip))
    tot0 = sum(r['w0'] for r in rs); tot1 = sum(r['w0'] * w[id(r)] for r in rs)
    return {k: v * tot0 / tot1 for k, v in w.items()}, pool
def report(wpos, gamma, label):
    def wf(r):
        f = (1 - r['p']) ** gamma if not r['none'] else 1.
        return r['w0'] * f * (wpos.get(id(r), 1.) if wpos else 1.)
    e = [r for r in rows if not r['none']]
    w = np.array([wf(r) for r in e]); st = np.array([r['seen_t'] for r in e]); sd = np.array([np.mean(r['seen_d']) for r in e])
    pt = np.average(st, weights=w); pd_ = np.average(sd, weights=w)
    wc = np.array([wf(r) for r in ce])
    by = {}
    for b in range(4):
        m = np.array([r['b'] == b for r in ce]); lst = np.array([r['last'] for r in ce])
        by[f'{BUCKETS[b][0]}-{BUCKETS[b][1]}'] = [int(m.sum()), round(float(np.average(lst[m], weights=wc[m])), 3), GOLD_SHARE[b]]
    lst = np.array([r['last'] for r in ce]); ans = np.array([r['truth'] == 'ANS-01' for r in ce])
    return {'label': label, 'claude_err_last_share_overall': round(float(np.average(lst, weights=wc)), 3),
            'claude_err_last_by_bucket[n,weighted,gold]': by,
            'max_abs_bucket_dev_from_gold': round(max(abs(v[1] - v[2]) for v in by.values()), 3),
            'kish_ess_error_rows': int(w.sum() ** 2 / (w ** 2).sum()), 'error_rows': len(e),
            'kish_ess_claude_err_rows': int(wc.sum() ** 2 / (wc ** 2).sum()), 'claude_err_rows': len(ce),
            'LR_unseen': round(float((1 - pt) / (1 - pd_)), 4),
            'ANS-01_share_weighted_claude_err': round(float(np.average(ans, weights=wc)), 4)}
wA, poolA = pos_weights(ce); wB, poolB = pos_weights(ce, exclude_ans=True); wC, _ = pos_weights(ce, clip=(0.0, 1e9))
res['pool_bucket_last_share_v5_claude_err'] = {f'{BUCKETS[b][0]}-{BUCKETS[b][1]}': round(float(poolA[b]), 3) for b in range(4)}
res['variants'] = [report(None, 0, 'current (no focal, no positional)'), report(None, 2, 'focal g2 only'),
                   report(wA, 0, 'positional only, clip[0.3,3]'), report(wA, 2, 'focal g2 x positional clip[0.3,3]'),
                   report(wB, 2, 'focal g2 x positional, ANS-* excluded'), report(wC, 2, 'focal g2 x positional unclipped')]
base_ans = res['variants'][0]['ANS-01_share_weighted_claude_err']
for v in res['variants']:
    v['ANS-01_rel_change_vs_current'] = round(v['ANS-01_share_weighted_claude_err'] / base_ans - 1, 3)
    v['gate_pass'] = {'ii_bucket_within_0.03': v['max_abs_bucket_dev_from_gold'] <= 0.03, 'iii_ess>=4000': v['kish_ess_error_rows'] >= 4000,
                      'iii_LR_0.45-0.6': 0.45 <= v['LR_unseen'] <= 0.6, 'iv_ANS01_within_20pct': abs(v['ANS-01_rel_change_vs_current']) <= 0.2}
json.dump(res, open(OUT / 't2_gate.json', 'w'), indent=1); print(json.dumps(res, indent=1))

# ---- variant: positional weights stratified by (bucket, ANS-vs-rest), targets = gold train shares in the same cell,
# each cell's total weight preserved (so the code mix, incl. ANS-01, is unchanged by the positional part)
gcell = {}
for r in tr.itertuples():
    if r.root_code == 'NONE': continue
    s, _ = root_lines(r.misconceptions); Lg = ids_of(r.transcription_steps)
    if s and min(s) in Lg:
        k = (bname(len(Lg)), r.root_code.startswith('ANS')); gcell.setdefault(k, []).append(min(s) == Lg[-1])
gshare = {k: float(np.mean(v)) for k, v in gcell.items()}
wS = {}
for k in gshare:
    cell = [r for r in ce if (r['b'], r['truth'].startswith('ANS')) == k]
    if not cell: continue
    s = np.average([r['last'] for r in cell], weights=[r['w0'] for r in cell]); t = gshare[k]
    raw = {id(r): float(np.clip(t / s if r['last'] else (1 - t) / (1 - s), 0.3, 3.0)) for r in cell}
    z = sum(r['w0'] for r in cell) / sum(r['w0'] * raw[id(r)] for r in cell)
    wS.update({i: v * z for i, v in raw.items()})
res['gold_cell_last_share(bucket,isANS)'] = {f'{BUCKETS[k[0]][0]}-{BUCKETS[k[0]][1]}|ANS={k[1]}': [len(gcell[k]), round(gshare[k], 3)] for k in sorted(gshare)}
for gam in (0, 2):
    v = report(wS, gam, f'focal g{gam} x positional stratified by (bucket, ANS)')
    v['ANS-01_rel_change_vs_current'] = round(v['ANS-01_share_weighted_claude_err'] / base_ans - 1, 3)
    v['gate_pass'] = {'ii_bucket_within_0.03': v['max_abs_bucket_dev_from_gold'] <= 0.03, 'iii_ess>=4000': v['kish_ess_error_rows'] >= 4000,
                      'iii_LR_0.45-0.6': 0.45 <= v['LR_unseen'] <= 0.6, 'iv_ANS01_within_20pct': abs(v['ANS-01_rel_change_vs_current']) <= 0.2}
    res['variants'].append(v)
json.dump(res, open(OUT / 't2_gate.json', 'w'), indent=1)
print(json.dumps(res['gold_cell_last_share(bucket,isANS)']))
for v in res['variants']:
    print(v['label'], v['claude_err_last_share_overall'], v['max_abs_bucket_dev_from_gold'], v['kish_ess_error_rows'], v['LR_unseen'], v['ANS-01_rel_change_vs_current'], all(v['gate_pass'].values()))
