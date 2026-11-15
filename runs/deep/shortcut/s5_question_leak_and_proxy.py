"""(5) Can question identity be removed from the input?  And a CPU proxy test of anti-shortcut training.

Part A, leak probe: identify the question_id of each dev row from its MACHINE TRANSCRIPT ALONE (no question text),
by nearest neighbour over fit gold + Claude transcripts (char 2-5-gram TF-IDF, cosine). If this is high, masking
question_text / question dropout cannot hide the question: the student's working names it.

Part B, proxy: multinomial logistic regression on transcript TF-IDF (train: fit gold transcripts + Claude rows with
Claude transcripts; dev: machine transcripts), scored per slice. Variants:
  T    transcript only
  TQ   transcript + question one-hot (shortcut handed to the model)
  TQd  TQ with question dropout (each training row duplicated, the copy with the question one-hot zeroed; test without q)
  PoE  transcript model trained with the log question-prior as a fixed additive offset (product of experts,
       Clark et al. 2019 / He et al. 2019); the offset is dropped at test
  RW   T with leave-one-out new-pair rows up-weighted x4 (group reweighting)
This is a linear proxy, not the LLM; only the DIRECTION of each technique on new-pair vs seen-pair rows is informative.
"""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from k12.baseline import HierarchicalPrior
from k12.core import topk, row_scores, normalize
from sc_common import load, slices, OUT, ROOT, r, boot_ci, dev_preds
from collections import Counter

tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
S, qc = slices(dev, fit)
y = dev.root_code.to_numpy()
unl = pd.read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv', dtype=str, keep_default_na=False).set_index('trace_id')
tx_dev = json.loads((ROOT / 'runs/pod/tx/tx_dev.json').read_text())
# Claude rows: first code + transcription (unlabelled photos only); cap NONE to a third like the distill sets
rng = np.random.default_rng(2026)
cl = []
for line in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    x = json.loads(line); a = x.get('answer') or {}
    if a.get('first_code') in ci and str(a.get('transcription', '')).strip() and x['trace_id'] in unl.index:
        cl.append(dict(trace_id=x['trace_id'], question_id=unl.at[x['trace_id'], 'question_id'], topic_name=unl.at[x['trace_id'], 'topic_name'],
                       root_code=a['first_code'], text=a['transcription']))
cl = pd.DataFrame(cl)
err = cl[cl.root_code != 'NONE']; non = cl[cl.root_code == 'NONE']
cl = pd.concat([err, non.sample(n=min(len(non), len(err) // 2), random_state=2026)]).reset_index(drop=True)
fitd = fit.assign(text=fit.transcription_steps)[['trace_id', 'question_id', 'topic_name', 'root_code', 'text']]
pool = pd.concat([fitd, cl[['trace_id', 'question_id', 'topic_name', 'root_code', 'text']]], ignore_index=True)
strip = lambda t: ' '.join(str(t).replace('|', ' ').split())
import re
noL = lambda t: re.sub(r'(?m)^\s*L\d+\s*\|\s?', '', str(t))
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, preprocessor=lambda t: noL(t).lower())
Xp = vec.fit_transform(pool.text)
Xd = vec.transform([tx_dev[t] for t in dev.trace_id])
res = {'pool_rows': len(pool), 'claude_rows': len(cl), 'fit_rows': len(fitd)}

# ---- Part A: question identification from the transcript alone
sim = (Xd @ Xp.T).toarray()
nn = sim.argmax(1)
pq = pool.question_id.to_numpy()[nn]
inpool = dev.question_id.isin(set(pool.question_id)).to_numpy()
# top-5 distinct questions by max similarity
top5 = []
for i in range(len(dev)):
    order = np.argsort(-sim[i])
    seen = []
    for j in order:
        q = pq_j = pool.question_id.iat[j]
        if q not in seen:
            seen.append(q)
        if len(seen) == 5:
            break
    top5.append(dev.question_id.iat[i] in seen)
res['A_question_id_from_transcript'] = dict(dev_rows_q_in_pool=int(inpool.sum()), nn_top1_acc=r((pq == dev.question_id.to_numpy())[inpool].mean()),
                                            nn_top5_acc=r(np.mean(np.array(top5)[inpool])),
                                            chance=r(1 / pool.question_id.nunique()),
                                            by_slice={s: r((pq == dev.question_id.to_numpy())[(S == s) & inpool].mean()) for s in ['none', 'seen_pair', 'new_pair']})
# Same probe restricted to the 1,199 fit gold transcripts only (what a fit-only model sees)
fm = np.arange(len(fitd))
pqf = fitd.question_id.to_numpy()[sim[:, fm].argmax(1)]
inf = dev.question_id.isin(set(fitd.question_id)).to_numpy()
res['A_question_id_from_transcript']['fit_only_pool_nn_top1_acc'] = r((pqf == dev.question_id.to_numpy())[inf].mean())

# ---- Part B: proxy models
ytr = pool.root_code.map(ci).to_numpy()
Q = sorted(set(pool.question_id) | set(dev.question_id)); qi = {q: i for i, q in enumerate(Q)}
def qoh(qs):
    return sp.csr_matrix((np.ones(len(qs)), (np.arange(len(qs)), [qi[q] for q in qs])), shape=(len(qs), len(Q)))
Qtr, Qdev = qoh(pool.question_id), qoh(dev.question_id)
Qdev0 = sp.csr_matrix((len(dev), len(Q)))


def softreg(X, yy, w=None, offset=None, l2=1.0, iters=300):
    n, d = X.shape
    w = np.ones(n) if w is None else w
    Y = np.zeros((n, K)); Y[np.arange(n), yy] = 1
    off = np.zeros((n, K)) if offset is None else offset
    Xb = sp.hstack([X, np.ones((n, 1))]).tocsr()
    def f(theta):
        Wm = theta.reshape(d + 1, K)
        Z = Xb @ Wm + off
        Z -= Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        P = np.exp(Z - lse[:, None])
        loss = (w * (lse - (Z * Y).sum(1))).sum() / w.sum() + .5 * l2 * (Wm[:-1] ** 2).sum() / n
        G = Xb.T @ ((P - Y) * w[:, None]) / w.sum()
        G[:-1] += l2 * Wm[:-1] / n
        return loss, G.ravel()
    th = minimize(f, np.zeros((d + 1) * K), jac=True, method='L-BFGS-B', options=dict(maxiter=iters)).x
    Wm = th.reshape(d + 1, K)
    return lambda Xt: normalize(np.exp((lambda Z: Z - Z.max(1, keepdims=True))(sp.hstack([Xt, np.ones((Xt.shape[0], 1))]).tocsr() @ Wm)))


# LOO new-pair flag in the pool (truth absent from the other rows of the question)
cnt = {q: Counter(g.root_code) for q, g in pool.groupby('question_id')}
newp = np.array([(c != 'NONE') and cnt[q][c] <= 1 for q, c in zip(pool.question_id, pool.root_code)])
res['B_pool_loo_new_pair_share_error_rows'] = r(newp[pool.root_code.to_numpy() != 'NONE'].mean())
prior_pool = HierarchicalPrior(classes).fit(pool)
# LOO prior offset for PoE (own row removed so the offset is not the label itself)
def loo_prior(df):
    yy = df.root_code.map(ci).to_numpy()
    g = np.bincount(yy, minlength=K) + .5; g /= g.sum()
    tc = {t: np.bincount(yy[i], minlength=K).astype(float) for t, i in df.groupby('topic_name').indices.items()}
    qq = {q: np.bincount(yy[i], minlength=K).astype(float) for q, i in df.groupby('question_id').indices.items()}
    out = np.zeros((len(df), K))
    for k, (q, t) in enumerate(zip(df.question_id, df.topic_name)):
        a = tc[t].copy(); a[yy[k]] -= 1; b = qq[q].copy(); b[yy[k]] -= 1
        pt = (a + 8 * g) / (a.sum() + 8); out[k] = (b + 2 * pt) / (b.sum() + 2)
    return out
off = np.log(np.clip(loo_prior(pool), 1e-6, 1))

variants = {}
variants['T'] = softreg(Xp, ytr)(Xd)
variants['TQ'] = softreg(sp.hstack([Xp, Qtr]).tocsr(), ytr)(sp.hstack([Xd, Qdev]).tocsr())
f_tq = softreg(sp.vstack([sp.hstack([Xp, Qtr]), sp.hstack([Xp, sp.csr_matrix(Qtr.shape)])]).tocsr(), np.concatenate([ytr, ytr]))
variants['TQd_test_with_q'] = f_tq(sp.hstack([Xd, Qdev]).tocsr())
variants['TQd_test_no_q'] = f_tq(sp.hstack([Xd, Qdev0]).tocsr())
variants['PoE'] = softreg(Xp, ytr, offset=off)(Xd)
variants['RW_x4'] = softreg(Xp, ytr, w=np.where(newp, 4., 1.))(Xd)
P = dev_preds(dev, classes)
variants['_qprior_pool'] = prior_pool.predict_proba(dev)
res['B_proxy_map3'] = {}
qid = dev.question_id.to_numpy()
for k, p in variants.items():
    sc = row_scores(y, topk(p, classes))
    res['B_proxy_map3'][k] = {s: r(sc[S == s].mean()) for s in ['none', 'seen_pair', 'new_pair', 'unseen_q']} | {'all': r(sc.mean())}
# paired deltas vs T on new-pair + unseen-q rows and on seen-pair rows
base = row_scores(y, topk(variants['T'], classes))
nu = np.isin(S, ['new_pair', 'unseen_q'])
res['B_delta_vs_T'] = {}
for k in ['TQ', 'TQd_test_no_q', 'PoE', 'RW_x4']:
    sc = row_scores(y, topk(variants[k], classes))
    res['B_delta_vs_T'][k] = dict(new_or_unseen=dict(zip(['delta', 'ci95'], boot_ci(sc[nu], base[nu], qid[nu]))),
                                  seen_pair=dict(zip(['delta', 'ci95'], boot_ci(sc[S == 'seen_pair'], base[S == 'seen_pair'], qid[S == 'seen_pair']))),
                                  all=dict(zip(['delta', 'ci95'], boot_ci(sc, base, qid))))
# Fusion with the blend: does any proxy add to the blend? cross-fitted geometric weight (5 question folds)
qs = dev.question_id.unique(); rr = np.random.default_rng(2026); rr.shuffle(qs); qf = {q: i % 5 for i, q in enumerate(qs)}
fold = dev.question_id.map(qf).to_numpy()
grid = [0, .1, .2, .3, .5]
bl = row_scores(y, topk(P['blend'], classes))
res['B_fusion_with_blend'] = {}
for k in ['T', 'TQ', 'TQd_test_no_q', 'PoE', 'RW_x4']:
    sc = {a: row_scores(y, topk(normalize(np.exp(np.log(P['blend']) + a * np.log(np.clip(variants[k], 1e-9, 1)))), classes)) for a in grid}
    out = np.zeros(len(dev)); ch = []
    for f in range(5):
        a = max(grid, key=lambda a: sc[a][fold != f].mean()); ch.append(a); out[fold == f] = sc[a][fold == f]
    res['B_fusion_with_blend'][k] = dict(crossfit_map3=r(out.mean()), chosen=ch, delta=dict(zip(['delta', 'ci95'], boot_ci(out, bl, qid))),
                                         new_or_unseen_delta=r((out - bl)[nu].mean()))
json.dump(res, open(OUT / 's5_question_leak_and_proxy.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
