"""Supplement to c_test.py: the cross-fitted fusion weight is 0 for every recipe (proxy too weak), so fused numbers equal blend+Sinkhorn
by construction.  Here: fixed fusion weights (same weight for base and variant) to isolate the synthetic-row effect inside fusion,
plus per-row view of the dev new-pair/unseen rows whose exact (question, code) pair has a synthetic row."""
import sys, json, warnings
warnings.filterwarnings('ignore')
src = open('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W5/c_test.py').read()
src = src.split('# ---------- (a) realism ----------')[0]
exec(src)
from scipy.sparse import vstack
fit_uncov = ~fit.question_id.isin(pq.index).to_numpy(); g3 = np.where(fit_uncov, 3., 1.)
wA = scheme('all_labelshift_pairbal_gen')
def fitpred3(w_syn, ysyn_used):
    m = wA > 0; ms = w_syn > 0
    X = vstack([Xfit, Xpool[m], Xsyn[ms]]); yy = np.concatenate([yfit, ypool[m], ysyn_used[ms]])
    sw = np.concatenate([g3, wA[m], w_syn[ms]])
    lr = LogisticRegression(C=4.0, max_iter=400, tol=1e-3); lr.fit(X, yy, sample_weight=sw)
    P = np.full((Xdev.shape[0], K), 1e-6); P[:, lr.classes_] += lr.predict_proba(Xdev); return normalize(P)
rng = np.random.default_rng(11); placebo = ysyn.copy(); rng.shuffle(placebo)
Ps = {'base': fitpred3(np.zeros(len(syn)), ysyn), 'syn_w0.5': fitpred3(np.full(len(syn), .5), ysyn),
      'syn_w1.0': fitpred3(np.full(len(syn), 1.), ysyn), 'placebo_w0.5': fitpred3(np.full(len(syn), .5), placebo)}
out = dict(fixed_w={}, exact_pair_newpair_rows=[])
for w in [.1, .2, .3]:
    S = {k: row_scores(y, topk(sinkhorn(normalize(B * P ** w), pi_fit, .5), classes)) for k, P in Ps.items()}
    for k in Ps:
        if k == 'base': continue
        out['fixed_w'][f'w{w}/{k}'] = dict(
            fusedS=round(float(S[k].mean()), 4), base_fusedS=round(float(S['base'].mean()), 4),
            minus_base=boot_ci(S[k], S['base'], qid), minus_blendS=boot_ci(S[k], refS, qid),
            minus_base_inj_err=boot_ci(S[k][injdev], S['base'][injdev], qid[injdev]),
            minus_base_none=boot_ci(S[k][y == 'NONE'], S['base'][y == 'NONE'], qid[y == 'NONE']))
    out['fixed_w'][f'w{w}/base_minus_blendS'] = boot_ci(S['base'], refS, qid)
m = npu & pair_hit
for i in np.where(m)[0]:
    out['exact_pair_newpair_rows'].append(dict(code=y[i], slice=sl[i], blendS_pts=float(refS[i]),
        rank_true={k: int((P[i] > P[i, ci[y[i]]]).sum()) + 1 for k, P in Ps.items()},
        n_syn_rows_same_pair=int(((syn.question_id == dev.question_id[i]) & (syn.root_code == y[i])).sum())))
json.dump(out, open(OUT2 / 'c_test2.json', 'w'), indent=1, default=str)
for k, v in out['fixed_w'].items(): print(k, v)
for r in out['exact_pair_newpair_rows']: print(r)
