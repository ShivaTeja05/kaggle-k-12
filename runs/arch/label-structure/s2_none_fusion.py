from common import *
from checker import check
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import json, sys
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ni = classes.index('NONE'); none = (y == 'NONE')
ref_pts = row_scores(y, topk(P, classes))
print('ref', ref_pts.mean())
# Oracle NONE handling (upper bound for any NONE detector)
Q = P.copy(); Q[none] = 0; Q[none, ni] = 1; Q[~none, ni] = 0; Q = normalize(Q + 1e-12)
print('oracle-NONE map3', map3(y, Q, classes))
# oracle using final_answer_correct gold as a feature (upper bound of perfect checker)
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
feats = {}
for k in (1, 2, 3):
    feats[f'chk{k}'] = np.array([check(tx[t], a, k)[0] for t, a in zip(ids, dev.correct_answer)])
feats['fac_gold'] = dev.final_answer_correct.astype(str).eq('True').to_numpy().astype(float)
lp = np.log(np.clip(P[:, ni], 1e-6, 1-1e-6)); lpo = np.log(np.clip(1-P[:, ni], 1e-6, 1))
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])

def fuse(cols, C=1.0):
    X = np.column_stack([lp - lpo] + [feats[c] for c in cols])
    q = np.empty(len(y))
    for f in range(5):
        a = mf != f
        m = LogisticRegression(C=C).fit(X[a], none[a]); q[~a] = m.predict_proba(X[~a])[:, 1]
    R = P.copy(); scale = (1-q)/np.clip(1-P[:, ni], 1e-9, None)
    R = R * scale[:, None]; R[:, ni] = q
    R = normalize(np.clip(R, 1e-12, None))
    pts = row_scores(y, topk(R, classes))
    d, ci = boot_ci(pts, ref_pts, dev.question_id.to_numpy())
    return dict(auc=float(roc_auc_score(none, q)), map3=float(pts.mean()), delta=d, ci95=ci)
out = {'ref_map3': float(ref_pts.mean()), 'oracle_none_map3': float(map3(y, Q, classes)),
       'blend_none_auc': float(roc_auc_score(none, P[:, ni]))}
for cols in [[], ['chk1'], ['chk2'], ['chk1', 'chk2', 'chk3'], ['fac_gold']]:
    r = fuse(cols); out['fuse_' + '+'.join(cols or ['blend_only'])] = r; print(cols, r)
json.dump(out, open('none_fusion.json', 'w'), indent=1)
