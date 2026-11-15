"""W1 kill test (A1, error-statement bottleneck), CPU only, adversarial.

Questions:
 Q0  Was the earlier 'Claude reasoning -> code' transfer (t1: pair0 0.522) inflated by code IDs left in the text?
 Q1  (spec kill test) A TF-IDF+LR readout trained on ERROR lines *extracted* from Claude pool error rows:
     score on dev gold root texts and on Claude's extracted dev ERROR lines, by (question, code) pair-count bin.
     Kill if pair0 <= 0.454 (blend+Sinkhorn) or new-minus-seen gap not flatter than -0.405.
 Q2  Is the readout just a lossy copy of the generator's own letter? (readout vs Claude's own codes, same rows)
 Q3  Fusion value, cross-fitted on dev (5 question-grouped folds): blend x readout^w (+ Sinkhorn lambda),
     with and without the generator's own letter. Upper bound because the generator here is Claude (dev 0.819).
 Q4  Degradation sweep: generator writes Claude's ERROR line with prob q, else the statement of the blend's
     top-1 code (shortcut-following generator). Break-even q for a positive fused gain.
No API, no GPU, no data leaves the machine. Outputs only under runs/deep/test-W1/.
"""
import json, re, sys, time
from collections import Counter
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, boot_ci, ROOT, sinkhorn
from k12.core import row_scores, topk, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

T0 = time.time()
OUT = ROOT / 'runs/deep/test-W1'
D = get_all(); dev, fit, y, classes, P, PS = D['dev'], D['fit'], D['y'], D['classes'], D['P'], D['PS']
ci = {c: i for i, c in enumerate(classes)}
g = dev.question_id.to_numpy()
pair = Counter(zip(fit.question_id, fit.root_code)); qseen = set(fit.question_id)
def binof(q, c):
    if c == 'NONE': return 'NONE'
    if q not in qseen: return 'q_unseen'
    k = pair[(q, c)]; return 'pair0' if k == 0 else 'pair1' if k == 1 else 'pair2' if k == 2 else 'pair3+'
bins = np.array([binof(q, c) for q, c in zip(dev.question_id, dev.root_code)])
err = y != 'NONE'; NEW = np.isin(bins, ['pair0', 'q_unseen']); SEEN = np.isin(bins, ['pair1', 'pair2', 'pair3+'])
ref_pts = row_scores(y, D['preds']['blend_sk']); cl_pts = row_scores(y, D['preds']['claude'])
print('ref blend_sk dev MAP', ref_pts.mean(), 'claude', cl_pts.mean())

# ---------------- extraction rules ----------------
ID = re.compile(r'\b[A-Za-z]{2,3}-\d{2}\b')
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\'"‘“])')
KEY = re.compile(r"\b(instead|error|errors|wrong|wrongly|slip|mistake|mistaken|incorrect|incorrectly|misread|misreads|misinterpret\w*|"
                 r"confus\w*|treat\w*|fail\w*|forg[eo]t\w*|omit\w*|never|missing|ignor\w*|swap\w*|revers\w*|invalid|rather than|as if|"
                 r"but|should|not)\b", re.I)
ROOTS = re.compile(r'\b(the root( error| cause| misconception)? is|root error|is the root|taken as the root|is taken as root)\b', re.I)
CODEY = re.compile(r'\b(code|codes|fits? .{0,30}best|primary choice|weaker alternative|close alternative)\b', re.I)
NOERR = re.compile(r'\bno (major |real |substantive )?(error|mistake|misconception)', re.I)

def sents(t):
    return [s.strip() for s in SPLIT.split(str(t or '')) if s.strip()]
def clean(s):
    s = ID.sub(' ', s); s = re.sub(r'\bNONE\b', ' ', s); return re.sub(r'\s+', ' ', s).strip()
def keep_sents(t):
    # drop sentences about the code choice (they carry the hidden ID decision)
    return [s for s in sents(t) if not ID.search(s) and not CODEY.search(s) and not re.search(r'\bNONE\b', s)]

def ex_noid(t):            # full reasoning, code IDs and code-choice sentences removed
    return ' '.join(keep_sents(t))
def ex_raw(t):             # full reasoning as t1 used it (IDs kept)
    return str(t or '')
def ex_e1(t, none=False):  # spec rule: first error-describing sentence
    ss = keep_sents(t)
    if none:
        z = [s for s in ss if NOERR.search(s)]; return clean(z[0] if z else (ss[-1] if ss else 'No major error.'))
    z = [s for s in ss if KEY.search(s)]
    return clean(z[0] if z else (ss[0] if ss else ''))
def ex_e2(t, none=False):  # prefer the root-summary sentence, else E1
    if none: return ex_e1(t, True)
    ss = keep_sents(t); z = [s for s in ss if ROOTS.search(s)]
    return clean(z[-1]) if z else ex_e1(t)
EX = {'raw': lambda t, n=False: ex_raw(t), 'noid': lambda t, n=False: ex_noid(t), 'e1': ex_e1, 'e2': ex_e2}

# ---------------- data ----------------
pool = []
for l in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') in ci and a.get('reasoning'):
        pool.append((r['trace_id'], a['reasoning'], a['first_code']))
perr = [p for p in pool if p[2] != 'NONE']; pnone = [p for p in pool if p[2] == 'NONE']
rng = np.random.default_rng(2026); pnone_s = [pnone[i] for i in rng.choice(len(pnone), 4000, replace=False)]
cdev = {}
for l in open(ROOT / 'runs/claude/dev_opus.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}; cdev[r['trace_id']] = (a.get('reasoning', ''), a.get('first_code'))
def gold_root(m):
    try: ms = json.loads(m)
    except Exception: return ''
    return ' '.join(x.get('text', '') for x in ms if x.get('root'))
gtext = np.array([gold_root(m) for m in dev.misconceptions])

class Mapper:
    def __init__(self, C=5.):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
        self.lr = LogisticRegression(C=C, max_iter=3000)
    def fit(self, X, yy):
        self.lr.fit(hstack([self.w.fit_transform(X), self.c.fit_transform(X)]).tocsr(), yy); return self
    def proba(self, X):
        p = self.lr.predict_proba(hstack([self.w.transform(X), self.c.transform(X)]).tocsr())
        F = np.full((len(X), len(classes)), 1e-6); F[:, [ci[c] for c in self.lr.classes_]] = p
        return normalize(F)

def binrep(s, mask):
    return {b: [round(float(s[(bins == b) & mask].mean()), 3), int(((bins == b) & mask).sum())]
            for b in ['q_unseen', 'pair0', 'pair1', 'pair2', 'pair3+', 'NONE'] if ((bins == b) & mask).any()}
def gap(s, mask=None):
    m = np.ones(len(s), bool) if mask is None else mask
    return round(float(s[NEW & m].mean() - s[SEEN & m].mean()), 3)

res = {'n_pool_error_rows': len(perr), 'n_pool_none_sampled': len(pnone_s),
       'reference': {'blend_sk_dev_map3': round(float(ref_pts.mean()), 4), 'claude_dev_map3': round(float(cl_pts.mean()), 4),
                     'blend_sk_bins': binrep(ref_pts, err), 'claude_bins': binrep(cl_pts, err),
                     'blend_sk_new_minus_seen': gap(ref_pts), 'claude_new_minus_seen': gap(cl_pts)}}
# extraction diagnostics
ex_stats = {}
for k in ['e1', 'e2']:
    xs = [EX[k](t) for _, t, _ in perr]
    ex_stats[k] = {'empty_frac': round(float(np.mean([not x for x in xs])), 4),
                   'mean_words': round(float(np.mean([len(x.split()) for x in xs])), 1),
                   'id_left_frac': float(np.mean([bool(ID.search(x)) for x in xs])),
                   'examples': [x for x in xs[:5]]}
ex_stats['raw_id_frac'] = round(float(np.mean([bool(ID.search(t)) for _, t, _ in perr])), 4)
res['extraction'] = ex_stats
print(json.dumps(ex_stats, indent=1)[:2500])

# ---------------- Q0/Q1/Q2: error-only readouts ----------------
ide = np.where(err)[0]
cdev_err_text = {k: [EX[k](cdev[t][0], cdev[t][1] == 'NONE') for t in dev.trace_id[ide]] for k in EX}
q1 = {}; readout_err = {}
for k in ['raw', 'noid', 'e1', 'e2']:
    t = time.time()
    m = Mapper().fit([EX[k](tx) for _, tx, _ in perr], np.array([c for _, _, c in perr]))
    out = {}
    for src, X in [('claude_dev_lines', cdev_err_text[k]), ('gold_dev_root_text', list(gtext[ide]))]:
        pr = m.proba(X); pred = topk(pr, classes); s = np.zeros(len(y)); s[ide] = row_scores(y[ide], pred)
        pb = np.where(bins == 'pair0')[0]; pu = np.where(bins == 'q_unseen')[0]
        out[src] = dict(bins=binrep(s, err), err_rows_map=round(float(s[err].mean()), 4), new_minus_seen=gap(s, err),
                        pair0_vs_blend_sk=boot_ci(s[pb], ref_pts[pb], g[pb]), pair0_vs_claude=boot_ci(s[pb], cl_pts[pb], g[pb]),
                        q_unseen_vs_blend_sk=boot_ci(s[pu], ref_pts[pu], g[pu]),
                        newrows_vs_blend_sk=boot_ci(s[NEW], ref_pts[NEW], g[NEW]))
        if src == 'claude_dev_lines':
            readout_err[k] = s
            c1 = np.array([cdev[t][1] for t in dev.trace_id[ide]])
            out[src]['top1_agrees_with_claude_first_code'] = round(float((pred[:, 0] == c1).mean()), 3)
            out[src]['readout_minus_claude_on_err_rows'] = boot_ci(s[err], cl_pts[err], g[err])
            out[src]['readout_minus_claude_on_pair0'] = boot_ci(s[pb], cl_pts[pb], g[pb])
    q1[k] = out
    print(k, round(time.time() - t), 's', json.dumps({s: (o['bins'].get('pair0'), o['new_minus_seen']) for s, o in out.items()}))
res['Q0_Q1_Q2_error_rows'] = q1
kill = {}
for k in ['e1', 'e2']:
    for src in ['claude_dev_lines', 'gold_dev_root_text']:
        o = q1[k][src]; kill[f'{k}/{src}'] = dict(pair0=o['bins']['pair0'][0], gap=o['new_minus_seen'],
                                                  kill_pair0=o['bins']['pair0'][0] <= 0.454, kill_gap=o['new_minus_seen'] <= -0.405)
res['kill_criteria'] = kill

# ---------------- Q3/Q4: fusion (readout over all rows incl. NONE) ----------------
K = 'e2' if q1['e2']['claude_dev_lines']['bins']['pair0'][0] >= q1['e1']['claude_dev_lines']['bins']['pair0'][0] else 'e1'
res['fusion_extraction'] = K
tr_rows = perr + pnone_s
mall = Mapper().fit([EX[K](tx, c == 'NONE') for _, tx, c in tr_rows], np.array([c for _, _, c in tr_rows]))
cl_lines = [EX[K](cdev[t][0], cdev[t][1] == 'NONE') for t in dev.trace_id]
R_claude = mall.proba(cl_lines)
# generator's own letter as a soft distribution (Claude gives ranked codes only)
L_claude = np.full((len(y), len(classes)), 1e-3)
for i, cs in enumerate(D['preds']['claude']):
    for c, w in zip(cs, [.6, .25, .1]):
        L_claude[i, ci[c]] += w
L_claude = normalize(L_claude)
res['readout_all_rows_alone_map3'] = round(float(row_scores(y, topk(R_claude, classes)).mean()), 4)

pi_fit = D['pi_fit']
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
gf = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([gf[q] for q in dev.question_id])
WG = [0, .25, .5, .75, 1.]; LG = [0, .25, .5, .75, 1.]
def crossfit(comps):
    """comps: list of (name, matrix). Geometric fusion with the blend, then Sinkhorn; grid chosen per fold."""
    import itertools
    grids = list(itertools.product(*([WG] * len(comps)), LG)); cache = {}
    for gr in grids:
        lp = np.log(P) + sum(w * np.log(M) for w, (_, M) in zip(gr[:-1], comps))
        F = normalize(np.exp(lp - lp.max(1, keepdims=True)))
        cache[gr] = row_scores(y, topk(sinkhorn(F, pi_fit, gr[-1]), classes))
    pts = np.empty(len(y)); ch = []
    for f in range(5):
        tr_ = mf != f
        best = max(grids, key=lambda gr: (cache[gr][tr_].mean(), -sum(gr[:-1])))
        ch.append(best); pts[~tr_] = cache[best][~tr_]
    best_in = max(grids, key=lambda gr: cache[gr].mean())
    d, c95 = boot_ci(pts, ref_pts, g)
    return pts, dict(crossfit_map3=round(float(pts.mean()), 4), delta_vs_0737=round(d, 4), ci95=[round(x, 4) for x in c95],
                     choices=[list(map(float, c)) for c in ch], in_sample_best=[list(map(float, best_in)), round(float(cache[best_in].mean()), 4)],
                     new_rows=round(float(pts[NEW].mean()), 3), seen_rows=round(float(pts[SEEN].mean()), 3), none_rows=round(float(pts[~err].mean()), 3),
                     new_delta=boot_ci(pts[NEW], ref_pts[NEW], g[NEW]), seen_delta=boot_ci(pts[SEEN], ref_pts[SEEN], g[SEEN]))
q3 = {}
_, q3['blend_x_readout(claude lines)'] = crossfit([('R', R_claude)])
_, q3['blend_x_claude_letter'] = crossfit([('L', L_claude)])
p_both, q3['blend_x_claude_letter_x_readout'] = crossfit([('L', L_claude), ('R', R_claude)])
_, q3['blend_only_crossfit_sinkhorn'] = crossfit([])
# does the readout add anything once the generator's own letter is in? (paired, same folds)
p_L, _ = crossfit([('L', L_claude)])
q3['readout_increment_over_letter'] = boot_ci(p_both, p_L, g)
res['Q3_fusion_upper_bound'] = q3
print(json.dumps(q3, indent=1)[:3000])

# Q4 degradation: shortcut-following generator
by_code = {}
for tid, tx, c in tr_rows:
    by_code.setdefault(c, []).append(EX[K](tx, c == 'NONE'))
btop = D['preds']['blend'][:, 0]
R_short = mall.proba([by_code[c][np.random.default_rng(i).integers(len(by_code[c]))] if c in by_code else 'No major error.'
                      for i, c in enumerate(btop)])
q4 = {}
for q in [0., .25, .5, .6, .7, .8, .9, 1.]:
    vals, ds = [], []
    for seed in range(5):
        u = np.random.default_rng(100 + seed).random(len(y)) < q
        Rq = np.where(u[:, None], R_claude, R_short)
        pts, info = crossfit([('R', Rq)])
        vals.append(info['crossfit_map3']); ds.append(info['delta_vs_0737'])
    # CI reported for the median seed
    q4[str(q)] = dict(crossfit_map3_mean=round(float(np.mean(vals)), 4), delta_mean=round(float(np.mean(ds)), 4),
                      delta_min=round(float(np.min(ds)), 4), delta_max=round(float(np.max(ds)), 4))
    print('q', q, q4[str(q)])
res['Q4_degradation_sweep'] = q4
# what q does the readout need on pair0/new rows? fraction of new rows where blend top1 != truth and Claude top1 == truth
c1 = D['preds']['claude'][:, 0]; b1 = D['preds']['blend'][:, 0]
res['headroom_rows'] = dict(new_rows=int(NEW.sum()), blend_top1_wrong_claude_right_new=int(((b1 != y) & (c1 == y) & NEW).sum()),
                            blend_right_claude_wrong_new=int(((b1 == y) & (c1 != y) & NEW).sum()),
                            blend_top1_wrong_claude_right_all=int(((b1 != y) & (c1 == y)).sum()),
                            blend_right_claude_wrong_all=int(((b1 == y) & (c1 != y)).sum()))
res['runtime_s'] = round(time.time() - T0)
json.dump(res, open(OUT / 'w1_readout_kill.json', 'w'), indent=1, default=float)
print('done', res['runtime_s'])
