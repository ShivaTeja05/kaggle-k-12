"""W5 CPU test: does reweighting the Claude pool toward the gold label mix (label-shift importance weights,
optionally question/pair-balanced) help a cheap proxy student, alone and fused with the blend?

Proxy = char_wb 2-5 TF-IDF over (topic + question + transcript) + multinomial logistic regression.
Train: 1,199 fit gold rows (gold transcripts, weight 1) + Claude pool rows (Claude transcripts) under a weighting scheme.
Eval: 301 dev rows (fold 0) on machine transcripts. Question-grouped paired bootstrap CIs.
Run: PYTHONPATH=src:scripts:runs/arch/label-structure runs/deep/approach-W5/venv/bin/python runs/deep/approach-W5/w5_proxy.py
"""
import json, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import load, blend, boot_ci
from k12.core import normalize, topk, row_scores

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT / 'runs/deep/approach-W5'
GENERIC = {'CAL', 'WOP', 'ANS', 'OTH', 'NOT', 'FRM', 'VIS', 'CHK', 'UNT', 'REL'}
fam = lambda c: c.split('-')[0].upper()
isgen = lambda c: c != 'NONE' and fam(c) in GENERIC
noL = lambda t: re.sub(r'(?m)^\s*L\d+\s*\|\s*', ' ', str(t or ''))
R_POOL = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0   # pool weight mass / gold weight mass

tr, tax, classes, folds, dev, fit = load(); ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
unl = pd.read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
pool = []
for line in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); a = r.get('answer') or {}
    t = r['trace_id']
    if t in unl.index and a.get('first_code') in ci and (a.get('transcription') or '').strip():
        u = unl.loc[t]
        pool.append(dict(trace_id=t, question_id=u.question_id, topic=u.topic_name, q=u.question_text,
                         tx=a['transcription'][-1500:], y=a['first_code']))
pool = pd.DataFrame(pool)
v5 = {}
for line in open(ROOT / 'data/v5/cand_train.jsonl'):
    r = json.loads(line)
    if r['source'] == 'claude': v5[r['trace_id']] = r['weight']
txdev = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))

def text(topic, q, tx): return f"{topic} || {q} || {noL(tx)}".lower()
Xfit_t = [text(a, b, c) for a, b, c in zip(fit.topic_name, fit.question_text, fit.transcription_steps)]
Xpool_t = [text(a, b, c) for a, b, c in zip(pool.topic, pool.q, pool.tx)]
Xdev_t = [text(a, b, txdev.get(t, '')[-1500:]) for a, b, t in zip(dev.topic_name, dev.question_text, dev.trace_id)]
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)
vec.fit(Xfit_t + Xpool_t)
Xfit, Xpool, Xdev = vec.transform(Xfit_t), vec.transform(Xpool_t), vec.transform(Xdev_t)
yfit = fit.root_code.map(ci).to_numpy(); ypool = pool.y.map(ci).to_numpy()

# ---------------- label mixes (the W5 mechanism) ----------------
pi_fit = np.bincount(yfit, minlength=K) + .5; pi_fit /= pi_fit.sum()
pi_pool = np.bincount(ypool, minlength=K) + .5; pi_pool /= pi_pool.sum()
inv5 = pool.trace_id.isin(v5).to_numpy()
pi_v5 = np.bincount(ypool[inv5], minlength=K) + .5; pi_v5 /= pi_v5.sum()
def mix(y):
    y = list(y); e = [c for c in y if c != 'NONE']
    return dict(n=len(y), none=round(sum(c == 'NONE' for c in y) / len(y), 3), generic_of_err=round(np.mean([isgen(c) for c in e]), 3),
                top_err=Counter(e).most_common(5))

# Claude's own NONE/generic habit, measured on dev (aggregate only; used as a diagnostic, never as a training target)
cl = {}
for line in open(ROOT / 'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); cl[r['trace_id']] = (r['answer'] or {}).get('first_code')
gy = dev.root_code.tolist(); cy = [cl.get(t) or 'NONE' for t in dev.trace_id]
gN = np.array([c == 'NONE' for c in gy]); cN = np.array([c == 'NONE' for c in cy])
tpr, fpr = cN[gN].mean(), cN[~gN].mean()   # P(claude NONE | gold NONE), P(claude NONE | gold error)
mu = (pool.y == 'NONE').mean(); q_true = (mu - fpr) / (tpr - fpr)            # BBSE (Lipton et al. 2018), binary NONE/error
err = ~gN
gen_cl = np.mean([isgen(c) for c, g in zip(cy, gy) if g != 'NONE' and c != 'NONE'])
gen_gold = np.mean([isgen(g) for g, c in zip(gy, cy) if g != 'NONE' and c != 'NONE'])
diag = dict(claude_dev_none_rate=round(float(cN.mean()), 3), gold_dev_none_rate=round(float(gN.mean()), 3),
            p_claudeNONE_given_goldNONE=round(float(tpr), 3), p_claudeNONE_given_goldERR=round(float(fpr), 3),
            pool_claude_none_share=round(float(mu), 3), bbse_corrected_pool_none_share=round(float(q_true), 3),
            dev_err_rows_both_nonNONE_generic_share=dict(claude=round(float(gen_cl), 3), gold=round(float(gen_gold), 3)),
            mix_gold_fit=mix(fit.root_code), mix_pool=mix(pool.y), mix_v5_claude=mix(pool.y[inv5]), mix_dev_gold=mix(dev.root_code))

# question coverage
pq = pool.question_id.value_counts()
nq = pool.question_id.map(pq).to_numpy().astype(float)
nqc = pool.groupby(['question_id', 'y']).trace_id.transform('size').to_numpy().astype(float)
covered = dev.question_id.isin(pq.index).to_numpy()

def scheme(name):
    w = np.zeros(len(pool))
    if name == 'gold_only': return w
    if name == 'uniform_all': w[:] = 1
    elif name == 'v5_as_built': w = np.array([v5.get(t, 0.) for t in pool.trace_id])
    elif name == 'v5_labelshift': w = np.where(inv5, (pi_fit / pi_v5)[ypool], 0.)
    elif name == 'all_labelshift': w = (pi_fit / pi_pool)[ypool]
    elif name == 'all_labelshift_qbal': w = (pi_fit / pi_pool)[ypool] / np.sqrt(nq)
    elif name == 'all_labelshift_pairbal': w = (pi_fit / pi_pool)[ypool] / np.sqrt(nqc)
    elif name == 'all_labelshift_pairbal_gen': w = (pi_fit / pi_pool)[ypool] / np.sqrt(nqc)  # + generic down-weight below
    w = np.minimum(w, np.quantile(w[w > 0], .99))  # clip extreme importance weights
    if name.endswith('_gen'):
        w = w * np.where(pool.y.map(isgen).to_numpy(), .7, 1.)
    return w * (R_POOL * len(fit) / w.sum())

def ess(w): w = w[w > 0]; return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.

def fitpred(w):
    m = w > 0
    from scipy.sparse import vstack
    X = vstack([Xfit, Xpool[m]]); y = np.concatenate([yfit, ypool[m]]); sw = np.concatenate([np.ones(len(fit)), w[m]])
    lr = LogisticRegression(C=4.0, max_iter=400, tol=1e-3)
    lr.fit(X, y, sample_weight=sw)
    P = np.full((Xdev.shape[0], K), 1e-6); P[:, lr.classes_] += lr.predict_proba(Xdev)
    return normalize(P), {classes[k]: float(v) for k, v in zip(*np.unique(y, return_counts=True))}

def sinkhorn(P, target, lam, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(target / Q.mean(0))
    return normalize(P * np.exp(lam * b))

ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
B = blend(ids, classes); ref = row_scores(y, topk(B, classes)); refS = row_scores(y, topk(sinkhorn(B, pi_fit, .5), classes))
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
fitq = fit.groupby('question_id').root_code.apply(set).to_dict()
sl = np.array(['none' if t == 'NONE' else 'unseen_q' if q not in fitq else 'seen_pair' if t in fitq[q] else 'new_pair'
               for q, t in zip(dev.question_id, dev.root_code)])
GRID = [0, .1, .2, .3, .5, .75, 1.]

def crossfit_fuse(P):
    cache = {w: normalize(B * P ** w) for w in GRID}; R = np.empty_like(B); ch = []
    for f in range(5):
        m = mf != f
        sc = [row_scores(y[m], topk(cache[w][m], classes)).mean() for w in GRID]
        w = GRID[int(np.argmax(sc))]; ch.append(w); R[~m] = cache[w][~m]
    return R, ch

def report(P):
    pts = row_scores(y, topk(P, classes)); t1 = np.array([classes[i] for i in P.argmax(1)])
    e = t1 != 'NONE'
    d = dict(map3=round(float(pts.mean()), 4), **{f'map3_{s}': round(float(pts[sl == s].mean()), 4) for s in ['none', 'seen_pair', 'new_pair', 'unseen_q']},
             map3_err_covered=round(float(pts[(y != 'NONE') & covered].mean()), 4), map3_err_uncovered=round(float(pts[(y != 'NONE') & ~covered].mean()), 4),
             pred_none_rate=round(float((~e).mean()), 3), pred_generic_share_of_err_top1=round(float(np.mean([isgen(c) for c in t1[e]])), 3))
    return d, pts

res = dict(diagnostics=diag, R_POOL=R_POOL, dev_covered_rows=int(covered.sum()), dev_err_uncovered_rows=int(((y != 'NONE') & ~covered).sum()),
           blend=report(B)[0], blend_sinkhorn=report(sinkhorn(B, pi_fit, .5))[0], schemes={})
pts_by = {}
for name in ['gold_only', 'uniform_all', 'v5_as_built', 'v5_labelshift', 'all_labelshift', 'all_labelshift_qbal', 'all_labelshift_pairbal', 'all_labelshift_pairbal_gen']:
    t0 = time.time(); w = scheme(name); P, cnt = fitpred(w)
    alone, pa = report(P)
    F, ch = crossfit_fuse(P); fused, pf = report(F)
    FS = sinkhorn(F, pi_fit, .5); fusedS, pfs = report(FS)
    pool_w = pd.Series(w, index=pool.y).groupby(level=0).sum()
    wm = pool_w / pool_w.sum() if pool_w.sum() > 0 else pool_w
    res['schemes'][name] = dict(
        pool_rows_used=int((w > 0).sum()), ess_pool=round(ess(w), 1),
        weighted_pool_none_share=round(float(wm.get('NONE', 0)), 3),
        weighted_pool_generic_share_of_err=round(float(sum(v for c, v in wm.items() if isgen(c)) / max(1e-9, sum(v for c, v in wm.items() if c != 'NONE'))), 3),
        weighted_pool_q_effective=round(float(np.exp(-(lambda p: (p * np.log(p)).sum())((lambda s: s[s > 0] / s.sum())(pd.Series(w, index=pool.question_id).groupby(level=0).sum())))), 1) if w.sum() > 0 else 0,
        alone=alone, fused_crossfit=fused, fused_w_choices=ch, fused_then_sinkhorn=fusedS,
        fused_minus_blend=boot_ci(pf, ref, qid), fusedS_minus_blendS=boot_ci(pfs, refS, qid), secs=round(time.time() - t0))
    pts_by[name] = (pa, pf, pfs)
    print(name, json.dumps(res['schemes'][name])[:600], flush=True)
base = 'v5_as_built'
for name in pts_by:
    if name == base: continue
    s = res['schemes'][name]
    s['alone_minus_v5'] = boot_ci(pts_by[name][0], pts_by[base][0], qid)
    s['alone_minus_v5_err_rows'] = boot_ci(pts_by[name][0][y != 'NONE'], pts_by[base][0][y != 'NONE'], qid[y != 'NONE'])
    s['alone_minus_v5_newpair_unseen'] = boot_ci(pts_by[name][0][np.isin(sl, ['new_pair', 'unseen_q'])], pts_by[base][0][np.isin(sl, ['new_pair', 'unseen_q'])], qid[np.isin(sl, ['new_pair', 'unseen_q'])])
    s['fusedS_minus_v5_fusedS'] = boot_ci(pts_by[name][2], pts_by[base][2], qid)
json.dump(res, open(OUT / f'w5_proxy_R{R_POOL:g}.json', 'w'), indent=1, default=str)
print(json.dumps(res['diagnostics'], indent=1, default=str))
