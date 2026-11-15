"""CPU probe (HiE, Jain et al. NeurIPS 2023; ErrorRadar family confusion): factorise p(code)=P(section)^a * p(code|section),
with P(section) from a per-model log-pool of section marginals (own weights) and p(code|section) from the blend.
Params (a, section weights) cross-fitted over 5 question folds. Base = blend, and blend after Sinkhorn(0.5)."""
import sys, json, itertools
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
tr, tax, classes, folds, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
secs = sorted(set(sec[c] for c in classes)); si = np.array([secs.index(sec[c]) for c in classes]); S = len(secs)
M = np.zeros((K, S)); M[np.arange(K), si] = 1
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
paths = [ROOT/'runs/pod/text32/validation_tx_dev_epoch_1.npz', ROOT/'runs/pod/tx/text_tx_dev_e3.npz', ROOT/'runs/pod/dev-distill/validation_epoch_2.npz']
Ps = [aligned_bundle(p, ids, classes) for p in paths]
B = blend(ids, classes)
pi = np.bincount(fit.root_code.map(ci), minlength=K) + .5; pi /= pi.sum()
def sink(P, lam=.5):
    b = np.zeros(K)
    for _ in range(30): Q = normalize(P*np.exp(b)); b += np.log(pi/Q.mean(0))
    return normalize(P*np.exp(lam*b))
def fact(base, a, ws):
    Fs = sum(w*np.log(P @ M + 1e-9) for w, P in zip(ws, Ps)); Fs = normalize(np.exp(Fs - Fs.max(1, keepdims=True)))
    Fb = base @ M; cond = base / (Fb[:, si] + 1e-12)
    lp = np.log(cond + 1e-12) + (1 - a)*np.log(Fb[:, si] + 1e-12) + a*np.log(Fs[:, si] + 1e-12)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))
grid = [(a, ws) for a in (0, .25, .5, .75, 1.) for ws in itertools.product((.2, .35, .5), repeat=3)]
qu = np.unique(qid); rng = np.random.default_rng(7); rng.shuffle(qu); g = {q: i % 5 for i, q in enumerate(qu)}
mf = np.array([g[q] for q in qid])
out = {'per_model_section_acc_err': {}}
err = y != 'NONE'; ys = np.array([secs.index(sec[c]) for c in y])
for n, P in zip(('32b', '8b', 'vl', 'blend'), Ps + [B]):
    out['per_model_section_acc_err'][n] = float(((P @ M).argmax(1)[err] == ys[err]).mean())
for name, base in (('blend', B), ('blend_sinkhorn0.5', sink(B))):
    ref = row_scores(y, topk(base, classes)); pts = np.zeros(len(ids)); ch = []
    for f in range(5):
        trn = mf != f
        scores = []
        for a, ws in grid:
            Pf = fact(base, a, ws); scores.append(row_scores(y[trn], topk(Pf[trn], classes)).mean())
        a, ws = grid[int(np.argmax(scores))]; ch.append([a, list(ws)])
        te = mf == f; pts[te] = row_scores(y[te], topk(fact(base, a, ws)[te], classes))
    out[name] = {'ref': float(ref.mean()), 'crossfit': float(pts.mean()), 'delta_ci': boot_ci(pts, ref, qid), 'choices': ch}
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT/'runs/research/family-probe/hie_probe.json', 'w'), indent=1)
