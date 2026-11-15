"""Per-row reading fidelity of machine (Qwen3-VL-8B tx) and Claude transcripts vs gold, dev fold."""
import sys, json, re, unicodedata
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/reading-impact')
import numpy as np, pandas as pd
from common import ROOT, dev

def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    for a, b in [('−', '-'), ('–', '-'), ('—', '-'), ('×', 'x'), ('·', '.'), ('÷', ':'), ('\\', '')]:
        s = s.replace(a, b)
    s = re.sub(r'\[[^\]]*\]', '', s)          # annotation markers like [circled]
    return re.sub(r'\s+', '', s)

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def substr_dist(pat, text):
    """min edit distance of pat to any substring of text (semi-global)."""
    if not pat: return 0
    prev = [0] * (len(text) + 1)
    for i, cp in enumerate(pat, 1):
        cur = [i]
        for j, ct in enumerate(text, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cp != ct)))
        prev = cur
    return min(prev)

def step_range(s):
    m = re.match(r'L(\d+)\s*-\s*L?(\d+)', s)
    if m:
        return [f'L{i}' for i in range(int(m.group(1)), int(m.group(2)) + 1)]
    return [s.strip()]

def lines(t):
    out = {}
    for ln in str(t).splitlines():
        m = re.match(r'\s*(L\d+)\s*\|\s?(.*)', ln)
        if m: out[m.group(1)] = m.group(2)
    return out

tx = json.load(open(ROOT + 'runs/pod/tx/tx_dev.json'))
cl = {}
for l in open(ROOT + 'runs/claude/dev_opus.jsonl'):
    r = json.loads(l); cl[r['trace_id']] = r['answer']

rows = []
for r in dev.itertuples(index=False):
    g = lines(r.transcription_steps)
    gpage = norm(' '.join(g.values()))
    mis = json.loads(r.misconceptions) if r.misconceptions else []
    root = [m for m in mis if m.get('root')]
    root = root[0] if root else None
    rec = dict(trace_id=r.trace_id, y=r.root_code, n_gold_lines=len(g), gold_chars=len(gpage),
               root_step=root['step'] if root else None, root_span=root['span'] if root else None)
    for name, src in (('m', tx.get(r.trace_id, '')), ('c', cl.get(r.trace_id, {}).get('transcription', ''))):
        L = lines(src)
        page = norm(' '.join(L.values()))
        rec[f'{name}_page_cer'] = lev(gpage, page) / max(1, len(gpage))
        rec[f'{name}_nlines_diff'] = len(L) - len(g)
        if root:
            steps = step_range(root['step'])
            gl = norm(' '.join(g.get(k, '') for k in steps))
            same = norm(' '.join(L.get(k, '') for k in steps))
            rec[f'{name}_rootline_cer_same'] = lev(gl, same) / max(1, len(gl))
            keys = list(L)
            wins = [' '.join(L[k] for k in keys[i:i + len(steps)]) for i in range(max(1, len(keys) - len(steps) + 1))]
            rec[f'{name}_rootline_cer_best'] = min(lev(gl, norm(w)) / max(1, len(gl)) for w in wins)
            sp = norm(root['span'])
            rec['span_len'] = len(sp)
            rec[f'{name}_span_dist'] = substr_dist(sp, page) / max(1, len(sp))
            # non-root part of page: remove root line from both and compare
            gr = norm(' '.join(v for k, v in g.items() if k not in steps))
            # best-matching machine line to the root line removed
            if L:
                bi = min(range(len(wins)), key=lambda i: lev(gl, norm(wins[i])))
                drop = set(keys[bi:bi + len(steps)])
                mr = norm(' '.join(v for k, v in L.items() if k not in drop))
            else:
                mr = ''
            rec[f'{name}_rest_cer'] = lev(gr, mr) / max(1, len(gr))
    if r.trace_id in cl and root:
        rec['c_root_line_pred'] = cl[r.trace_id].get('root_line')
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_csv(ROOT + 'runs/arch/reading-impact/per_row_reading.csv', index=False)
print(df.describe().T.to_string())
