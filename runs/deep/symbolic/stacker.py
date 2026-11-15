"""(ii) Do symbolic features help on top of the blend?  Cross-fitted on dev (question-grouped 5 folds,
5 repeats).  Machine transcripts only (deployable).
 A. NONE recalibration: logistic on [logit P_blend(NONE)] (+ symbolic final-answer/invalid flags).
 B. Code-level fusion: P'(c) ∝ P_blend(c) * P_aux(c)^w over non-NONE codes, where P_aux is a TF-IDF LR
    trained on fit gold (+ Claude unlabelled) error rows, with vs without the symbolically located line.
All variants are reported with and without Sinkhorn balancing (lambda .5)."""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load, blend, boot_ci
from k12.core import row_scores, topk, normalize
from sklearn.linear_model import LogisticRegression
import value_oracle as vo
from symutil import first_of

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qg = dev.question_id.to_numpy()
P = blend(ids, classes); ni = classes.index('NONE'); ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()


def sinkhorn(P, lam=.5, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(pi_fit / Q.mean(0))
    return normalize(P * np.exp(lam * b))


def m3(Q):
    return row_scores(y, topk(Q, classes))


ref = m3(P); refS = m3(sinkhorn(P))
mdev = vo.mdev; txm = vo.txm


def feats(t):
    rec = mdev[t]; L = rec['lines']
    f = lambda v: 0.5 if v is None else v
    return [f(rec['fac']), float(rec['fac'] is None), f(rec['fac1']),
            float(first_of(L, {'arith'}) is not None), float(first_of(L, {'eq'}) is not None),
            float(first_of(L, {'cont'}) is not None), float(first_of(L, {'bind'}) is not None),
            len(L) / 10.0]


S = np.array([feats(t) for t in ids])
lg = np.log(np.clip(P[:, ni], 1e-6, 1 - 1e-6)); lg = lg - np.log1p(-np.exp(lg))
isnone = (y == 'NONE').astype(int)


def qfolds(seed):
    qs = np.array(sorted(set(qg))); rng = np.random.default_rng(seed); rng.shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}
    return np.array([g[q] for q in qg])


def renone(P, pn):
    Q = P.copy(); rest = 1 - P[:, ni]
    Q[:, [j for j in range(P.shape[1]) if j != ni]] *= ((1 - pn) / np.clip(rest, 1e-9, None))[:, None]
    Q[:, ni] = pn
    return normalize(np.clip(Q, 1e-12, None))


res = dict(ref_blend=float(ref.mean()), ref_blend_sinkhorn=float(refS.mean()))
# ---------------- A: NONE recalibration
for name, X in [('A0_recal_only', lg[:, None]), ('A1_recal+symbolic', np.c_[lg, S])]:
    outs, outsS = [], []
    for seed in range(5):
        mf = qfolds(seed); pn = np.zeros(len(y))
        for f in range(5):
            m = mf != f
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[m], isnone[m])
            pn[~m] = clf.predict_proba(X[~m])[:, 1]
        Q = renone(P, pn); outs.append(m3(Q)); outsS.append(m3(sinkhorn(Q)))
    a, b = np.mean(outs, 0), np.mean(outsS, 0)
    res[name] = dict(map3=float(a.mean()), delta_vs_blend=boot_ci(a, ref, qg),
                     map3_sinkhorn=float(b.mean()), delta_vs_blend_sinkhorn=boot_ci(b, refS, qg))
    print(name, res[name], flush=True)
# full-fit coefficients for interpretation
clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.c_[lg, S], isnone)
res['A1_coefficients'] = dict(zip(['logitPnone', 'fac', 'fac_missing', 'fac1', 'arith_flag', 'eq_flag', 'cont_flag', 'bind_flag', 'nlines/10'],
                                  [round(float(c), 3) for c in clf.coef_[0]]))

# ---------------- B: code-level fusion with a localisation-aware aux model
fitrows = vo.build_fit(); unlrows = vo.build_unl()
devrows = []
for r in dev.itertuples():
    tx = txm[r.trace_id]; rec = mdev[r.trace_id]
    devrows.append(dict(y=r.root_code, topic=r.topic_name, q=f'{r.topic_name} || {r.question_text}', tx=tx,
                        span='', rootline='', symline=vo.located(rec, tx), symtok=vo.sym_tokens(rec)))
grid = [0, .05, .1, .2, .3, .5, .75, 1.0]
nn = [j for j in range(len(classes)) if j != ni]
for reg, train in [('fit_only', fitrows), ('fit+unl', fitrows + unlrows)] if unlrows else [('fit_only', fitrows)]:
    for view in ['Q+transcript', 'Q+transcript+sym_line+symtok', 'Q+transcript+symtok', 'topic+sym_line', 'topic+transcript']:
        (Pa, cls), = vo.run_view(train, [devrows], vo.VIEWS[view], 3.0)
        A = np.full((len(ids), len(classes)), 1e-6)
        for k, c in enumerate(cls):
            A[:, ci[c]] = np.clip(Pa[:, k], 1e-6, None)
        A[:, nn] /= A[:, nn].sum(1, keepdims=True)
        cache = {}
        for w in grid:
            Q = P.copy(); Q[:, nn] = P[:, nn] * A[:, nn] ** w
            Q[:, nn] *= (P[:, nn].sum(1) / Q[:, nn].sum(1))[:, None]   # keep NONE mass
            Q = normalize(Q); cache[w] = (m3(Q), m3(sinkhorn(Q)))
        for si, tag in [(0, ''), (1, '_sinkhorn')]:
            outs, ch = [], []
            for seed in range(5):
                mf = qfolds(seed); o = np.zeros(len(y))
                for f in range(5):
                    m = mf != f
                    w = max(grid, key=lambda w: cache[w][si][m].mean()); ch.append(w)
                    o[~m] = cache[w][si][~m]
                outs.append(o)
            a = np.mean(outs, 0)
            key = f'B|{reg}|{view}{tag}'
            res[key] = dict(map3=float(a.mean()), delta=boot_ci(a, refS if si else ref, qg),
                            w_choices=sorted(set(ch)), insample={w: round(float(cache[w][si].mean()), 4) for w in grid})
            print(key, round(a.mean(), 4), res[key]['delta'], res[key]['w_choices'], flush=True)
json.dump(res, open(HERE / 'stacker.json', 'w'), indent=1)
