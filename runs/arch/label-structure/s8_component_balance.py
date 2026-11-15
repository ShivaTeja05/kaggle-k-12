"""Balance each blend component to the capped train label marginals before the geometric blend."""
from common import *
import json, itertools
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ref = row_scores(y, topk(P, classes)); qg = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def bias(P, target, iters=50):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(target / Q.mean(0))
    return b
comps = [('32b', 'runs/pod/text32/validation_tx_dev_epoch_1.npz', .35), ('8b', 'runs/pod/tx/text_tx_dev_e3.npz', .30), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz', .35)]
Ps = [aligned_bundle(ROOT/p, ids, classes) for _, p, _ in comps]; W = [w for *_, w in comps]
Bs = [bias(Pm, pi_fit) for Pm in Ps]
def make(lams, final=0.):
    lp = sum(w * (np.log(Pm) + l * b) for Pm, b, w, l in zip(Ps, Bs, W, lams))
    Q = normalize(np.exp(lp - lp.max(1, keepdims=True)))
    if final: Q = normalize(Q * np.exp(final * bias(Q, pi_fit)))
    return Q
out = {}
for lams in [(0, 0, 0), (1, 1, 1), (.5, .5, .5), (1, 0, 0), (1, .5, .5), (1, 1, 0)]:
    Q = make(lams); pts = row_scores(y, topk(Q, classes))
    out[f'fixed_lams_{lams}'] = dict(map3=float(pts.mean()), delta_ci=boot_ci(pts, ref, qg))
# cross-fit over shared/per-component grid + final
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
grid = [(l, f) for l in itertools.product((0, .5, 1.), repeat=3) for f in (0, .5)]
cache = {c: make(*c) for c in grid}
for seed in (7, 11, 23):
    qs2 = dev.question_id.unique().copy(); np.random.default_rng(seed).shuffle(qs2)
    g2 = {q: i % 5 for i, q in enumerate(qs2)}; mf2 = np.array([g2[q] for q in dev.question_id])
    R = np.empty_like(P); ch = []
    for f in range(5):
        m = mf2 != f
        sc = [row_scores(y[m], topk(cache[c][m], classes)).mean() for c in grid]
        c = grid[int(np.argmax(sc))]; ch.append(str(c)); R[~m] = cache[c][~m]
    pts = row_scores(y, topk(R, classes))
    out[f'crossfit_seed{seed}'] = dict(map3=float(pts.mean()), delta_ci=boot_ci(pts, ref, qg), choices=ch)
best = max(grid, key=lambda c: row_scores(y, topk(cache[c], classes)).mean())
out['in_sample_best(optimistic)'] = [str(best), float(row_scores(y, topk(cache[best], classes)).mean())]
# blend-weight re-tune after balancing (1,1,1) to see if weights shift
print(json.dumps(out, indent=1))
json.dump(out, open('component_balance.json', 'w'), indent=1)
