"""CPU design check: how strong must a top-K candidate verifier be (relative to Claude) to reach 0.80,
and does Sinkhorn balancing + 235B few-shot stack. Claude is used ONLY as an oracle stand-in on dev."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_claude import claude_scores
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
P = blend(ids, classes); ref = row_scores(y, topk(P, classes))
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
F = aligned_bundle(ROOT/'runs/pod/probe235/fewshot235/validation.npz', ids, classes)
ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def sinkhorn(P, lam, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(pi_fit / Q.mean(0))
    return normalize(P * np.exp(lam * b))
def sc(M): return row_scores(y, topk(M, classes))
qs = np.unique(qid).copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
def crossfit(cands):  # cands: dict param->matrix; choose on 4/5 question folds
    keys = list(cands); pts = {k: sc(cands[k]) for k in keys}; out = np.empty(len(y)); ch = []
    for f in range(5):
        m = mf != f; k = max(keys, key=lambda k: pts[k][m].mean()); ch.append(k); out[~m] = pts[k][~m]
    d, c = boot_ci(out, ref, qid); return dict(map3=float(out.mean()), delta=d, ci95=c, choices=[str(x) for x in ch])
res = {'ref': float(ref.mean())}
# 1) Sinkhorn + 235B stacking
def geo(ms, ws):
    lp = sum(w*np.log(m) for m, w in zip(ms, ws)); return normalize(np.exp(lp - lp.max(1, keepdims=True)))
cands = {}
for wf in [0, .1, .2, .3]:
    B = geo([P, F], [1, wf])
    for lam in [0, .25, .5, .75]: cands[(wf, lam)] = sinkhorn(B, lam)
res['sinkhorn_plus_235'] = crossfit(cands)
# 2) Verifier-strength curve: verifier knows Claude's ranking on a fraction r of rows (random, 20 seeds), uninformative otherwise.
def verifier_mix(r, K, w, seed, lam=0.0, gate=None):
    rng = np.random.default_rng(seed); Pb = sinkhorn(P, lam) if lam else P
    top = np.argsort(-Pb, 1)[:, :K]; V = np.full_like(Pb, 1e-3)
    know = rng.random(len(y)) < r
    for i in range(len(y)):
        cand = top[i]
        V[i, cand] = C[i, cand] if know[i] else 1.0 / K
    V = normalize(V)
    M = geo([Pb, V], [1, w])
    if gate is not None: M = np.where(gate[:, None], M, Pb)
    return M
curve = {}
for K in [5, 10]:
    for r in [0.25, 0.5, 0.75, 1.0]:
        vals = []
        for s in range(10):
            cands = {w: verifier_mix(r, K, w, s) for w in [0, .5, 1, 2, 4]}
            vals.append(crossfit(cands)['map3'])
        curve[f'K{K}_r{r}'] = float(np.mean(vals))
res['verifier_curve_mean_crossfit'] = curve
# 3) Cascade: verifier (r=1, K=10) only on the lowest-confidence fraction of rows
conf = P.max(1); casc = {}
for frac in [.2, .4, .6, 1.0]:
    gate = conf <= np.quantile(conf, frac)
    casc[frac] = crossfit({w: verifier_mix(1.0, 10, w, 0, gate=gate) for w in [0, .5, 1, 2, 4]})['map3']
res['cascade_r1_K10'] = casc
# 4) Ceiling of the Claude oracle restricted to blend top-10 alone (no blend product)
top10 = np.argsort(-P, 1)[:, :10]; Cr = np.full_like(C, 1e-6)
for i in range(len(y)): Cr[i, top10[i]] = C[i, top10[i]] + 1e-4*P[i, top10[i]]
res['claude_within_blend_top10'] = float(sc(normalize(Cr)).mean())
res['claude_alone'] = float(sc(C).mean())
print(json.dumps(res, indent=1)); json.dump(res, open('verifier_sim.json', 'w'), indent=1)
