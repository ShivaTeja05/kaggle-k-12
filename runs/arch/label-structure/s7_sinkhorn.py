"""Transductive marginal matching (label-free on the evaluated set) toward the capped train label distribution."""
from common import *
import json
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ref = row_scores(y, topk(P, classes))
ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def sinkhorn(P, target, lam, iters=30):
    b = np.zeros(P.shape[1]); Q = P
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(target / Q.mean(0))
    return normalize(P * np.exp(lam * b))
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
grid = [0, .1, .25, .5, .75, 1.]
cache = {l: sinkhorn(P, pi_fit, l) for l in grid}
ins = {l: float(row_scores(y, topk(cache[l], classes)).mean()) for l in grid}
R = np.empty_like(P); ch = []
for f in range(5):
    m = mf != f
    sc = [row_scores(y[m], topk(cache[l][m], classes)).mean() for l in grid]
    l = grid[int(np.argmax(sc))]; ch.append(l); R[~m] = cache[l][~m]
pts = row_scores(y, topk(R, classes)); d, c95 = boot_ci(pts, ref, dev.question_id.to_numpy())
err = y != 'NONE'
out = dict(in_sample=ins, crossfit=float(pts.mean()), delta=d, ci95=c95, choices=ch,
           fixed_lam_0p5=dict(map3=ins[.5], delta_ci=boot_ci(row_scores(y, topk(cache[.5], classes)), ref, dev.question_id.to_numpy())),
           err_rows_ref=float(ref[err].mean()), err_rows_lam0p5=float(row_scores(y[err], topk(cache[.5][err], classes)).mean()),
           none_rows_ref=float(ref[~err].mean()), none_rows_lam0p5=float(row_scores(y[~err], topk(cache[.5][~err], classes)).mean()))
# per-model: does balancing help each component?
for n, p in [('32b', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8b', 'runs/pod/tx/text_tx_dev_e3.npz'), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz')]:
    Pm = aligned_bundle(ROOT/p, ids, classes)
    out[f'component_{n}'] = {l: float(map3(y, sinkhorn(Pm, pi_fit, l), classes)) for l in (0, .5, 1.)}
print(json.dumps(out, indent=1))
json.dump(out, open('sinkhorn_balance.json', 'w'), indent=1)
