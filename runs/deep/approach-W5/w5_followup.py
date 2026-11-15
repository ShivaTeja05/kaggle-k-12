"""W5 follow-up (CPU): (1) post-hoc prior correction of a student trained on a capped mix vs train-time label-shift weights;
(2) coverage: does upweighting gold rows on questions the Claude pool never covers help uncovered-question error rows?
Reuses the setup of w5_proxy.py (exec of its data/feature section)."""
import sys
src = open('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/approach-W5/w5_proxy.py').read().split('res = dict(diagnostics')[0]
exec(src)
from scipy.sparse import vstack
fit_uncov = ~fit.question_id.isin(pq.index).to_numpy()
def fitpred2(w, gw):
    m = w > 0
    X = vstack([Xfit, Xpool[m]]); yy = np.concatenate([yfit, ypool[m]]); sw = np.concatenate([gw, w[m]])
    lr = LogisticRegression(C=4.0, max_iter=400, tol=1e-3); lr.fit(X, yy, sample_weight=sw)
    P = np.full((Xdev.shape[0], K), 1e-6); P[:, lr.classes_] += lr.predict_proba(Xdev)
    pi_tr = np.bincount(yy, weights=sw, minlength=K) + .5; pi_tr /= pi_tr.sum()
    return normalize(P), pi_tr
g1 = np.ones(len(fit)); g3 = np.where(fit_uncov, 3., 1.)
runs = {'gold_only': ('gold_only', g1), 'v5_as_built': ('v5_as_built', g1), 'v5_goldcov3': ('v5_as_built', g3),
        'pairbal_gen': ('all_labelshift_pairbal_gen', g1), 'pairbal_gen_goldcov3': ('all_labelshift_pairbal_gen', g3)}
errU = (y != 'NONE') & ~covered; errC = (y != 'NONE') & covered
out = dict(fit_rows_on_pool_uncovered_q=int(fit_uncov.sum()), dev_err_uncovered=int(errU.sum()), dev_err_covered=int(errC.sum()), runs={})
PTS = {}
for name, (sch, gw) in runs.items():
    w = scheme(sch); P, pi_tr = fitpred2(w, gw)
    for tag, PP in [('raw', P), ('posthoc_prior', normalize(P * pi_fit / pi_tr))]:   # Saerens/Menon prior swap: p_tr -> p_gold
        a, pa = report(PP); F, ch = crossfit_fuse(PP); FS = sinkhorn(F, pi_fit, .5); fs, pfs = report(FS)
        PTS[(name, tag)] = (pa, pfs)
        out['runs'][f'{name}/{tag}'] = dict(alone=a, fusedS=fs, w=ch, fusedS_minus_blendS=boot_ci(pfs, refS, qid))
    print(name, json.dumps(out['runs'][f'{name}/raw']['alone']), flush=True)
b = 'v5_as_built'
for (name, tag), (pa, pfs) in PTS.items():
    r = out['runs'][f'{name}/{tag}']
    r['alone_minus_v5raw_err_uncovered'] = boot_ci(pa[errU], PTS[(b, 'raw')][0][errU], qid[errU])
    r['alone_minus_v5raw_err_covered'] = boot_ci(pa[errC], PTS[(b, 'raw')][0][errC], qid[errC])
    r['alone_minus_v5raw_none'] = boot_ci(pa[y == 'NONE'], PTS[(b, 'raw')][0][y == 'NONE'], qid[y == 'NONE'])
    r['fusedS_minus_v5raw_fusedS'] = boot_ci(pfs, PTS[(b, 'raw')][1], qid)
json.dump(out, open(OUT / 'w5_followup.json', 'w'), indent=1, default=str)
for k, r in out['runs'].items():
    print(k, 'alone', r['alone']['map3'], 'errU', r['alone']['map3_err_uncovered'], 'errC', r['alone']['map3_err_covered'], 'none', r['alone']['map3_none'],
          '| fusedS', r['fusedS']['map3'], r['w'], 'vs blendS', r['fusedS_minus_blendS'], '| errU-v5', r['alone_minus_v5raw_err_uncovered'], 'fS-v5', r['fusedS_minus_v5raw_fusedS'])
