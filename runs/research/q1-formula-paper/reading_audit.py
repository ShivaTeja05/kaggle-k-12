"""Q1 audit: how good is our Qwen3-VL-8B reading, and where does reading loss show up in diagnosis? Aggregates only."""
import json, re, unicodedata
from pathlib import Path
import numpy as np
from k12.core import read_csv

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
tx = json.loads((ROOT / 'runs/pod/tx/tx_dev.json').read_text())
train = read_csv(ROOT / 'data/raw/train.csv').set_index('trace_id')

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def strip_ids(t):
    return [re.sub(r'^L\d+\s*\|\s*', '', l).strip() for l in str(t).splitlines() if l.strip()]

def norm(s):
    s = unicodedata.normalize('NFKC', s).lower()
    s = s.replace('−', '-').replace('×', 'x').replace('·', '*').replace('÷', ':').replace('²', '^2').replace('³', '^3')
    return re.sub(r'\s+', '', s)

MATH = re.compile(r'\d+|[a-z]|[=+\-*/:^()<>%.,]')
def mtoks(s):
    return MATH.findall(norm(s))

rows = []
for tid, text in tx.items():
    gold = str(train.loc[tid, 'transcription_steps'])
    g_lines, m_lines = strip_ids(gold), strip_ids(text)
    g_all, m_all = norm(' '.join(g_lines)), norm(' '.join(m_lines))
    cer_raw = lev(text, gold) / max(1, len(gold))
    cer_norm = lev(m_all, g_all) / max(1, len(g_all))
    gt, mt = mtoks(' '.join(g_lines)), mtoks(' '.join(m_lines))
    ter = lev(mt, gt) / max(1, len(gt))
    # root span recovery
    mis = train.loc[tid, 'misconceptions']
    try:
        mis = json.loads(mis) if isinstance(mis, str) else []
    except Exception:
        mis = []
    roots = [m for m in mis if m.get('root')]
    span_hit = None; span_cer = None; line_cer = None
    if roots:
        sp = norm(roots[0].get('span', ''))
        if sp:
            span_hit = sp in m_all
            # best substring match CER of span against any machine line (and adjacent pair)
            cands = [norm(l) for l in m_lines] + [norm(a + b) for a, b in zip(m_lines, m_lines[1:])]
            best = 1.0
            for c in cands:
                # approximate substring distance: sliding window of len(sp)
                L = len(sp)
                if len(c) <= L:
                    best = min(best, lev(c, sp) / L)
                else:
                    for k in range(0, len(c) - L + 1, max(1, L // 4)):
                        best = min(best, lev(c[k:k + L], sp) / L)
            span_cer = best
        step = roots[0].get('step', '')
        mm = re.match(r'L(\d+)', str(step))
        if mm:
            i = int(mm.group(1)) - 1
            if 0 <= i < len(g_lines):
                gl = norm(g_lines[i])
                line_cer = min((lev(norm(l), gl) / max(1, len(gl)) for l in m_lines), default=1.0)
    rows.append(dict(id=tid, cer_raw=cer_raw, cer_norm=cer_norm, ter=ter, span_hit=span_hit,
                     span_cer=span_cer, line_cer=line_cer, root=str(train.loc[tid, 'root_code']),
                     nchar=len(g_all)))

def q(a):
    a = np.array(a, float); return dict(mean=round(float(a.mean()), 4), median=round(float(np.median(a)), 4),
                                         p90=round(float(np.percentile(a, 90)), 4), n=len(a))

out = {'rows': len(rows),
       'cer_raw_with_line_ids': q([r['cer_raw'] for r in rows]),
       'cer_normalized_no_ids_no_space': q([r['cer_norm'] for r in rows]),
       'math_token_error_rate': q([r['ter'] for r in rows]),
       'root_span_exact_found_normalized': None, 'root_span_best_window_cer': None, 'root_line_best_match_cer': None}
sh = [r['span_hit'] for r in rows if r['span_hit'] is not None]
out['root_span_exact_found_normalized'] = dict(rate=round(float(np.mean(sh)), 4), n=len(sh))
out['root_span_best_window_cer'] = q([r['span_cer'] for r in rows if r['span_cer'] is not None])
sc = np.array([r['span_cer'] for r in rows if r['span_cer'] is not None])
out['root_span_window_cer_le_0.2_rate'] = round(float((sc <= 0.2).mean()), 4)
out['root_line_best_match_cer'] = q([r['line_cer'] for r in rows if r['line_cer'] is not None])

# downstream: 32B epoch-1 reasoner gold vs machine transcripts
def map3(p, classes, ids):
    s = []
    for i, tid in enumerate(ids):
        y = str(train.loc[tid, 'root_code']); y = y if y and y != 'nan' else 'NONE'
        top = [classes[j] for j in np.argsort(-p[i])[:3]]
        s.append(1 / (top.index(y) + 1) if y in top else 0.0)
    return np.array(s)
G = np.load(ROOT / 'runs/pod/text32/validation_gold_epoch_1.npz'); M = np.load(ROOT / 'runs/pod/text32/validation_tx_dev_epoch_1.npz')
assert list(G['ids']) == list(M['ids'])
sg, sm = map3(G['p'], list(G['classes']), list(G['ids'])), map3(M['p'], list(M['classes']), list(M['ids']))
out['reasoner32b_e1_map3_gold'] = round(float(sg.mean()), 4)
out['reasoner32b_e1_map3_machine'] = round(float(sm.mean()), 4)
rng = np.random.default_rng(0); d = sg - sm
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
out['gold_minus_machine_bootstrap95'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
out['rows_gold_better'] = int((d > 0).sum()); out['rows_machine_better'] = int((d < 0).sum())
byid = {r['id']: r for r in rows}
cers = np.array([byid[t]['cer_norm'] for t in G['ids']])
for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 9)]:
    m = (cers >= lo) & (cers < hi)
    out[f'cer_norm_bin_{lo}-{hi}'] = dict(n=int(m.sum()), map_gold=round(float(sg[m].mean()), 4) if m.any() else None,
                                          map_machine=round(float(sm[m].mean()), 4) if m.any() else None)
# span recovered vs not, machine-score
hit = np.array([byid[t]['span_cer'] is not None and byid[t]['span_cer'] <= 0.2 for t in G['ids']])
has = np.array([byid[t]['span_cer'] is not None for t in G['ids']])
out['root_rows_span_read_ok'] = dict(n=int((hit & has).sum()), map_gold=round(float(sg[hit & has].mean()), 4), map_machine=round(float(sm[hit & has].mean()), 4))
out['root_rows_span_misread'] = dict(n=int((~hit & has).sum()), map_gold=round(float(sg[~hit & has].mean()), 4), map_machine=round(float(sm[~hit & has].mean()), 4))
out['meta'] = dict(gold=str(G['meta']), machine=str(M['meta']))
print(json.dumps(out, indent=1))
(ROOT / 'runs/research/q1-formula-paper/reading_audit.json').write_text(json.dumps(out, indent=1))
