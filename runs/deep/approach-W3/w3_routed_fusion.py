"""W3 test 3: label-free routing. On rows where text members and the image member disagree about NONE
(the signature of a 'repaired' transcript), move blend weight toward the VL image member. Cross-fitted
(5 question-grouped folds x 5 repeats); compare to the fixed .35/.30/.35 blend, before and after Sinkhorn(0.5)."""
import json, sys
import numpy as np, pandas as pd
from common import load, ROOT, boot_ci, blend
from k12.core import aligned_bundle, topk, row_scores, normalize
tr, tax, classes, folds, dev, fit = load(); ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
qid = dev.question_id.to_numpy(); ci = {c: i for i, c in enumerate(classes)}; NI = ci['NONE']
L = {k: np.log(aligned_bundle(ROOT/p, ids, classes)) for k, p in
     [('32', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8', 'runs/pod/tx/text_tx_dev_e3.npz'), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz')]}
def mix(wvl):
    wvl = np.asarray(wvl, float)[:, None]; s = (1 - wvl) / .65
    lp = s * (.35 * L['32'] + .30 * L['8']) + wvl * L['vl']
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))
ref_P = mix(np.full(len(y), .35)); assert np.allclose(ref_P, blend(ids, classes), atol=1e-6)
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def sinkhorn(P, target=pi_fit, lam=.5, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(target / Q.mean(0))
    return normalize(P * np.exp(lam * b))
Pt = normalize(np.exp((.35 * L['32'] + .30 * L['8']) / .65)); Pv = np.exp(L['vl'])
dn = Pt[:, NI] - Pv[:, NI]                                   # text says NONE more than image does
js = .5 * (Pt * (np.log(Pt + 1e-12) - np.log(.5 * (Pt + Pv) + 1e-12))).sum(1) + .5 * (Pv * (np.log(Pv + 1e-12) - np.log(.5 * (Pt + Pv) + 1e-12))).sum(1)
t1t, t1v = Pt.argmax(1), Pv.argmax(1)
flags = {'textNONE_imgNOT': (t1t == NI) & (t1v != NI)}
for t in (.2, .4, .6): flags[f'dNONE>{t}'] = dn > t
for q in (.67, .8, .9): flags[f'JS>q{q}'] = js > np.quantile(js, q)
flags['top1_disagree'] = t1t != t1v
betas = [0, .1, .2, .3, .45, .65]
cand = {(f, b): mix(np.where(m, .35 + b, .35)) for f, m in flags.items() for b in betas}
ref = row_scores(y, topk(ref_P, classes)); refS = row_scores(y, topk(sinkhorn(ref_P), classes))
pts_all = {k: row_scores(y, topk(P, classes)) for k, P in cand.items()}
insample = sorted(((float(v.mean()), k) for k, v in pts_all.items()), reverse=True)[:5]
print('blend', ref.mean().round(4), 'blend+sinkhorn', refS.mean().round(4))
print('flag sizes / truth-NONE share / M6 share:')
lab = pd.read_csv(ROOT/'runs/deep/casebook/labels.csv', keep_default_na=False).set_index('trace_id')
mech = np.array([lab.mech.get(t, 'OK') for t in ids])
for f, m in flags.items(): print(f'  {f:16s} n={int(m.sum()):3d} NONE={np.mean(y[m]=="NONE"):.2f} M6={int((mech[m]=="M6").sum())}/21')
print('in-sample best (optimistic):', insample)
reps = []
for rep in range(5):
    qs = np.unique(qid).copy(); np.random.default_rng(100 + rep).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
    P = np.empty_like(ref_P); ch = []
    for f in range(5):
        tr_ = mf != f
        best = max(cand, key=lambda k: (pts_all[k][tr_].mean(), -k[1]))
        ch.append(best); P[~tr_] = cand[best][~tr_]
    reps.append((P, ch))
pts = np.mean([row_scores(y, topk(P, classes)) for P, _ in reps], 0)
ptsS = np.mean([row_scores(y, topk(sinkhorn(P), classes)) for P, _ in reps], 0)
d1 = boot_ci(pts, ref, qid); d2 = boot_ci(ptsS, refS, qid)
m6 = mech == 'M6'
out = dict(blend=float(ref.mean()), blend_sinkhorn=float(refS.mean()), routed_crossfit=float(pts.mean()), delta=d1,
           routed_sinkhorn=float(ptsS.mean()), delta_sinkhorn=d2,
           m6_rows=dict(ref=float(ref[m6].mean()), routed=float(pts[m6].mean())),
           none_rows=dict(ref=float(ref[y == 'NONE'].mean()), routed=float(pts[y == 'NONE'].mean())),
           choices=[[list(map(str, c)) for c in ch] for _, ch in reps], insample_top5=[[s, list(map(str, k))] for s, k in insample])
print(json.dumps({k: v for k, v in out.items() if k != 'choices'}, indent=1))
print('choices rep0:', out['choices'][0])
json.dump(out, open(ROOT/'runs/deep/approach-W3/w3_routed_fusion.json', 'w'), indent=1)
