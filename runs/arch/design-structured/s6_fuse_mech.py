"""Ceiling of the proposed pipeline's last stage: fuse the mechanism->code mapper (Claude dev reasoning
stripped, i.e. 'student as good as teacher at writing the mechanism') into the blend, error-conditional,
NONE mass from the blend, weight cross-fitted (5 question folds x 10 repeats), question bootstrap vs 0.718.
Then Sinkhorn-free. Diagnostic upper bound only."""
import json, re, numpy as np
exec(open('runs/arch/design-structured/s5_claude_mech.py').read().split("\nout = {}")[0])
from k12.core import normalize
Xd = [strip((cl.get(t) or {}).get('reasoning', '')) for t in dev.trace_id]
Pm = run(Xtr, Ytr, W)
Xd = [strip(x) for x in [root_txt(m) for m in dev.misconceptions]]
Pa = run(Xtr, Ytr, W)
L = lambda p: aligned_bundle(p, dev.trace_id.tolist(), classes)
blend = normalize(np.exp(.35*np.log(L('runs/pod/text32/validation_tx_dev_epoch_1.npz'))+.30*np.log(L('runs/pod/tx/text_tx_dev_e3.npz'))+.35*np.log(L('runs/pod/dev-distill/validation_epoch_2.npz'))))
ci = {c: i for i, c in enumerate(classes)}; ei = [ci[c] for c in err_codes]
ref = row_scores(y, topk(blend, classes)); qs = dev.question_id.values; uq = np.unique(qs)
be = blend[:, ei] / blend[:, ei].sum(1, keepdims=True)
def fuse(P, w):
    e = normalize(np.exp((1-w)*np.log(be) + w*np.log(P))); f = blend.copy()
    f[:, ei] = (1 - blend[:, [ci['NONE']]]) * e; return f
grid = np.round(np.arange(0, 1.01, .1), 2)
def crossfit(P):
    cache = {w: row_scores(y, topk(fuse(P, w), classes)) for w in grid}; res = []
    for rep in range(10):
        perm = np.random.default_rng(rep).permutation(uq); fq = {q: i % 5 for i, q in enumerate(perm)}
        f = np.array([fq[q] for q in qs]); sc = np.zeros(len(y))
        for k in range(5):
            wb = max(grid, key=lambda w: (cache[w][f != k].mean(), -w)); sc[f == k] = cache[wb][f == k]
        res.append(sc)
    return np.mean(res, 0), {float(w): round(float(cache[w].mean()), 4) for w in grid}
def boot(d):
    g = {q: np.flatnonzero(qs == q) for q in uq}; r = np.random.default_rng(1); m = []
    for _ in range(4000):
        idx = np.concatenate([g[q] for q in r.choice(uq, len(uq))]); m.append(d[idx].mean())
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
out = {}
for name, P in [('claude_mechanism', Pm), ('annotator_mechanism', Pa)]:
    sc, ins = crossfit(P); d = sc - ref
    out[name] = {'crossfit': round(float(sc.mean()), 4), 'delta': round(float(d.mean()), 4), 'ci': boot(d), 'in_sample': ins,
                 'delta_by_bin': {b: round(float(d[bin_ == b].sum() / len(y)), 4) for b in set(bin_)}}
print(json.dumps(out, indent=1)); json.dump(out, open('runs/arch/design-structured/s6_fuse_mech.json', 'w'), indent=1)
