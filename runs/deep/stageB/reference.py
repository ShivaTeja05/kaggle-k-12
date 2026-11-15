"""Step 0 (DEEP_PLAN): the fixed yardstick for every Stage B comparison. Cross-fitted blend+Sinkhorn on dev (λ chosen per
question-grouped fold, as runs/arch/label-structure/s7_sinkhorn.py), slice scores, and the label-free test Simpson index of
the blend's top-1 (pairs of rows on the same question). Writes runs/deep/stageB/reference.json."""
import hashlib, json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure')]
from common import load, blend
from k12.core import aligned_bundle, normalize, read_csv, row_scores, topk

tr, tax, classes, folds, dev, fit = load()
ids, y = dev.trace_id.tolist(), dev.root_code.to_numpy()
P = blend(ids, classes)
ci = {c: i for i, c in enumerate(classes)}
marg = lambda codes: (lambda b: b / b.sum())(np.bincount(codes.map(ci), minlength=len(classes)) + .5)


def sinkhorn(P, target, lam, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))


pi_fit = marg(fit.root_code)
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
grid = [0, .1, .25, .5, .75, 1.]
cache = {l: sinkhorn(P, pi_fit, l) for l in grid}
R, choices = np.empty_like(P), []
for f in range(5):
    m = mf != f
    l = grid[int(np.argmax([row_scores(y[m], topk(cache[l][m], classes)).mean() for l in grid]))]
    choices.append(l); R[~m] = cache[l][~m]
pts = row_scores(y, topk(R, classes))
qc = fit.groupby('question_id').root_code.apply(set).to_dict()
sl = np.array(['none' if t == 'NONE' else 'unseen_q' if q not in qc else 'seen_pair' if t in qc[q] else 'new_pair'
               for q, t in zip(dev.question_id, y)])
slices = {k: dict(n=int(m.sum()), map3=round(float(pts[m].mean()), 4)) for k, m in
          [('none', sl == 'none'), ('seen_pair', sl == 'seen_pair'), ('new_pair', sl == 'new_pair'),
           ('unseen_q', sl == 'unseen_q'), ('new_or_unseen_error', np.isin(sl, ['new_pair', 'unseen_q'])),
           ('error', sl != 'none')]}

test = read_csv(ROOT / 'data/raw/test.csv'); tids = test.trace_id.tolist()
lp = sum(w * np.log(aligned_bundle(ROOT / p, tids, classes)) for p, w in
         [('runs/pod/text32-full/test.npz', .35), ('runs/pod/text-full/test.npz', .30), ('runs/pod/full-distill/test.npz', .35)])
PT = normalize(np.exp(lp - lp.max(1, keepdims=True)))
pairs = [(i, j) for _, grp in test.groupby('question_id') for i, j in combinations(grp.index, 2)]
simpson = lambda Q: float(np.mean([a == b for a, b in ((t[i], t[j]) for t in [Q.argmax(1)] for i, j in pairs)]))
fallback = ROOT / 'artifacts/submission-3way-sinkhorn/submission.csv'
out = dict(dev_rows=len(y), crossfit_map3=round(float(pts.mean()), 4), crossfit_lambda_choices=choices,
           insample_lam_0p5=round(float(row_scores(y, topk(cache[.5], classes)).mean()), 4),
           raw_blend=round(float(row_scores(y, topk(P, classes)).mean()), 4), slices=slices,
           test_simpson_top1=dict(blend=round(simpson(PT), 4), blend_sinkhorn_0p5=round(simpson(sinkhorn(PT, marg(tr.root_code), .5, 50)), 4),
                                  pairs=len(pairs)),
           fallback=dict(path=str(fallback.relative_to(ROOT)), sha256=hashlib.sha256(fallback.read_bytes()).hexdigest(),
                         public_map3=0.71716, kaggle_ref=58258119),
           gate=dict(expected=0.7337, tol=.001))
out['gate']['pass'] = abs(out['crossfit_map3'] - .7337) <= .001
(ROOT / 'runs/deep/stageB/reference.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
