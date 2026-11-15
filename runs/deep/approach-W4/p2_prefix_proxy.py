"""W4 CPU proxy (TF-IDF + LR, fit rows only -> dev fold 0).
(A) Prefix-scan localiser: a 'prefix contains the root error' classifier trained with step labels derived
    from gold root lines (process supervision), vs an outcome-only classifier read as an implicit PRM
    (largest log-odds jump), vs position baselines. Gold and machine dev transcripts.
(B) Line-marginalised code classifier: p(c) = sum_k q(k) g(c | question, line k [+ prev line]) vs a
    whole-transcript classifier; top-1 = annotated secondary rate; cross-fitted fusion with the blend."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT/'runs/deep/approach-W4'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'runs/arch/label-structure'), str(ROOT/'runs/deep/symbolic')]
from common import load, blend, boot_ci
from k12.core import row_scores
from symutil import map_span, root_lines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

tr, tax, classes, folds, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
def lines(tx):
    out = []
    for i, l in enumerate([l for l in str(tx).splitlines() if l.strip()]):
        m = re.match(r'\s*L(\d+)\s*\|\s*(.*)', l); out.append((int(m.group(1)) if m else i+1, m.group(2) if m else l))
    return out
def rootline(ms):
    s, span = root_lines(ms); return (min(s) if s else None), span
def secset(r):
    try: L = json.loads(r.misconceptions)
    except Exception: return set()
    return {m['category'] for m in L if not m.get('root') and m.get('category') and m['category'] != r.root_code}
ctx = lambda r: f'{r.topic_name} || {r.question_text}'

# ---------------- (A) prefix classifier
def prefix_rows(df, step_labels=True):
    X1, X2, y = [], [], []
    for r in df.itertuples():
        L = lines(r.transcription_steps); rl, _ = rootline(r.misconceptions) if r.root_code != 'NONE' else (None, None)
        for k in range(1, len(L)+1):
            X1.append(ctx(r) + ' || ' + ' \n '.join(t for _, t in L[:k])); X2.append(L[k-1][1])
            y.append(int(r.root_code != 'NONE' and rl is not None and L[k-1][0] >= rl))
    return X1, X2, np.array(y)
v1 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
v2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000)
X1, X2, y = prefix_rows(fit)
Xp = hstack([v1.fit_transform(X1), v2.fit_transform(X2)]).tocsr()
step_clf = LogisticRegression(C=4, max_iter=3000).fit(Xp, y)
# outcome-only (implicit PRM): trained on full transcripts, label = row has error
Xo = [ctx(r) + ' || ' + ' \n '.join(t for _, t in lines(r.transcription_steps)) for r in fit.itertuples()]
vo = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
out_clf = LogisticRegression(C=4, max_iter=3000).fit(vo.fit_transform(Xo), (fit.root_code != 'NONE').astype(int))

txm = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
def scan(r, tx):
    L = lines(tx)
    if not L: return L, np.zeros(0), np.zeros(0)
    a = [ctx(r) + ' || ' + ' \n '.join(t for _, t in L[:k]) for k in range(1, len(L)+1)]
    ps = step_clf.predict_proba(hstack([v1.transform(a), v2.transform([t for _, t in L])]).tocsr())[:, 1]
    po = out_clf.predict_proba(vo.transform(a))[:, 1]
    return L, ps, po
def first_cross(L, ps, tau):
    j = next((i for i, p in enumerate(ps) if p >= tau), None)
    return L[j][0] if j is not None else L[-1][0]
def max_jump(L, p):
    lo = np.log(p/(1-p)); d = np.diff(np.r_[np.log(.5/.5) if False else lo[0]-0, lo]) if False else np.diff(np.r_[0.0, lo])
    return L[int(np.argmax(d))][0]
def max_jump_step(L, ps):
    d = np.diff(np.r_[0.0, ps]); return L[int(np.argmax(d))][0]

# choose tau on fit by 5-fold? use a fixed grid reported on dev, plus tau chosen on fit itself (in-sample, labelled)
res = {'A': {}}
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy(); err = truth != 'NONE'
rows = []
for r in dev.itertuples():
    if r.root_code == 'NONE': continue
    rl, span = rootline(r.misconceptions)
    if rl is None: continue
    rec = {'tid': r.trace_id, 'q': r.question_id, 'sec': bool(secset(r))}
    for name, tx in (('gold', r.transcription_steps), ('mach', txm.get(r.trace_id, ''))):
        if name == 'gold': target = rl
        else:
            target, sc = map_span(span, tx)
            if sc < 0.6: target = None
        L, ps, po = scan(r, tx)
        if not L or target is None: rec[name] = None; continue
        rec[name] = {'target': target, 'last': L[-1][0], 'first': L[0][0], 'n': len(L),
                     **{f'step_tau{t}': first_cross(L, ps, t) for t in (.3, .4, .5, .6)},
                     'step_maxjump': max_jump_step(L, ps), 'implicit_prm': max_jump(L, po),
                     'q': {lid: float(p) for (lid, _), p in zip(L, np.clip(np.diff(np.r_[0.0, ps]), 0, None))}}
    rows.append(rec)
for name in ('gold', 'mach'):
    R = [x[name] for x in rows if x[name]]
    keys = ['last', 'first', 'step_tau0.3', 'step_tau0.4', 'step_tau0.5', 'step_tau0.6', 'step_maxjump', 'implicit_prm']
    acc = {k: round(float(np.mean([x[k] == x['target'] for x in R])), 3) for k in keys}
    acc_pm1 = {k: round(float(np.mean([abs(x[k] - x['target']) <= 1 for x in R])), 3) for k in ('last', 'step_tau0.5', 'step_maxjump', 'implicit_prm')}
    Rs = [x[name] for x in rows if x[name] and x['sec']]
    acc_sec = {k: round(float(np.mean([x[k] == x['target'] for x in Rs])), 3) for k in ('last', 'step_tau0.5', 'step_maxjump', 'implicit_prm')}
    res['A'][name] = {'n': len(R), 'exact': acc, 'within1': acc_pm1, 'rows_with_secondary': {'n': len(Rs), 'exact': acc_sec},
                      'uniform_random': round(float(np.mean([1/x['n'] for x in R])), 3)}

# ---------------- (B) code classifiers
fe = fit[fit.root_code != 'NONE']
def line_text(r, k):
    L = dict(lines(r.transcription_steps)); return L.get(k, '')
def prev_text(r, k):
    L = dict(lines(r.transcription_steps)); return L.get(k-1, '')
Xl, yl, Xf, yf = [], [], [], []
for r in fe.itertuples():
    rl, _ = rootline(r.misconceptions)
    if rl is None or r.root_code not in ci: continue
    Xl.append((ctx(r), prev_text(r, rl), line_text(r, rl))); yl.append(r.root_code)
    Xf.append((ctx(r), ' \n '.join(t for _, t in lines(r.transcription_steps)))); yf.append(r.root_code)
vq = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000).fit([a for a, *_ in Xl])
vt = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000).fit([c for *_, c in Xl] + [b for _, b in Xf])
def featL(q, p, l): return hstack([vq.transform(q), vt.transform(p) * .5, vt.transform(l)]).tocsr()
def featF(q, t): return hstack([vq.transform(q), vt.transform(t)]).tocsr()
gL = LogisticRegression(C=8, max_iter=4000).fit(featL(*zip(*Xl)), yl)
gF = LogisticRegression(C=8, max_iter=4000).fit(featF(*zip(*Xf)), yf)
def full76(clf, pr):
    P = np.full((pr.shape[0], len(classes)), 1e-6)
    for j, c in enumerate(clf.classes_): P[:, ci[c]] = pr[:, j]
    return P / P.sum(1, keepdims=True)
byid = {x['tid']: x for x in rows}
dv = dev.set_index('trace_id')
variants = {}
def score_variant(name, src):
    P = np.zeros((len(dev), len(classes)))
    for i, r in enumerate(dev.itertuples()):
        tx = r.transcription_steps if src == 'gold' else txm.get(r.trace_id, '')
        L = lines(tx)
        if name == 'full':
            P[i] = full76(gF, gF.predict_proba(featF([ctx(r)], [' \n '.join(t for _, t in L)])))[0]; continue
        if not L: P[i] = 1/len(classes); continue
        D = dict(L)
        if name == 'oracle_line':
            x = byid.get(r.trace_id); k = x[src]['target'] if x and x.get(src) else L[-1][0]; q = {k: 1.0}
        elif name == 'marg_step':
            x = byid.get(r.trace_id)
            if x and x.get(src): q = x[src]['q']
            else:
                _, ps, _ = scan(r, tx); q = {lid: float(p) for (lid, _), p in zip(L, np.clip(np.diff(np.r_[0.0, ps]), 0, None))}
            s = sum(q.values()); q = {k: v/s for k, v in q.items()} if s > 0 else {L[-1][0]: 1.0}
        ks = list(q)
        pr = full76(gL, gL.predict_proba(featL([ctx(r)]*len(ks), [D.get(k-1, '') for k in ks], [D.get(k, '') for k in ks])))
        P[i] = sum(q[k]*pr[j] for j, k in enumerate(ks))
    return P
rk = lambda M: [[classes[j] for j in np.argsort(-m)[:3]] for m in M]
groups = dev.question_id.to_numpy()
qcodes = {}
for q, c in zip(fit.question_id, fit.root_code): qcodes.setdefault(q, Counter())[c] += 1
newpair = np.array([t != 'NONE' and qcodes.get(q, Counter())[t] == 0 for q, t in zip(dev.question_id, truth)])
hassec = np.array([bool(secset(r)) and r.root_code != 'NONE' for r in dev.itertuples()])
B = blend(ids, classes); sBl = np.array(row_scores(truth, rk(B)))
res['B'] = {}
folds_d = pd.factorize(dev.question_id)[0] % 5
def fuse(P, A, w):
    nz = ci['NONE']; lp = np.log(P) + w*np.log(np.clip(A, 1e-6, None)); lp[:, nz] = np.log(P[:, nz])
    out = np.exp(lp - lp.max(1, keepdims=True)); oth = np.delete(np.arange(len(classes)), nz)
    out[:, oth] *= ((1 - P[:, nz]) / out[:, oth].sum(1))[:, None]; out[:, nz] = P[:, nz]; return out
for src in ('gold', 'mach'):
    for name in ('full', 'oracle_line', 'marg_step'):
        A = score_variant(name, src); s = np.array(row_scores(truth, rk(A)))
        top1 = np.array([classes[j] for j in A.argmax(1)])
        sec_hit = np.mean([t in secset(r) for t, r, h in zip(top1, dev.itertuples(), hassec) if h])
        # cross-fitted fusion with blend (grid w; chosen on other folds)
        grid = [0, .1, .2, .3, .5, .75, 1.0]; F = np.zeros_like(B); chosen = []
        for f in range(5):
            tr_m = folds_d != f
            best = max(grid, key=lambda w: np.mean(np.array(row_scores(truth[tr_m], rk(fuse(B[tr_m], A[tr_m], w))))))
            chosen.append(best); F[~tr_m] = fuse(B[~tr_m], A[~tr_m], best)
        sF = np.array(row_scores(truth, rk(F)))
        d, cI = boot_ci(sF, sBl, groups); dN, cN = boot_ci(np.where(newpair, sF, 0), np.where(newpair, sBl, 0), groups)
        dS, cS = boot_ci(np.where(hassec, sF, 0), np.where(hassec, sBl, 0), groups)
        res['B'][f'{src}/{name}'] = {'alone_err_rows': round(float(s[err].mean()), 3), 'alone_newpair': round(float(s[newpair].mean()), 3),
                                     'alone_sec_rows': round(float(s[hassec].mean()), 3), 'alone_top1_is_secondary_on_sec_rows': round(float(sec_hit), 3),
                                     'fused_w_chosen': chosen, 'fused_delta_all': [round(d, 4), [round(x, 4) for x in cI]],
                                     'fused_delta_newpair_pts': [round(dN, 4), [round(x, 4) for x in cN]],
                                     'fused_delta_secrows_pts': [round(dS, 4), [round(x, 4) for x in cS]]}
res['blend_dev_map'] = round(float(sBl.mean()), 4)
json.dump(res, open(OUT/'p2_prefix_proxy.json', 'w'), indent=1); print(json.dumps(res, indent=1))
