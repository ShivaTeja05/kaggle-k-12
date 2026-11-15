"""Cross-fitting helpers: repeated question-grouped 5-fold within the 301 dev rows."""
import numpy as np
from common import *

def splits(n_rep=10, k=5, seed=7):
    q = dev.question_id.to_numpy(); uq = np.unique(q)
    out = []
    for r in range(n_rep):
        rng = np.random.default_rng(seed + r); perm = rng.permutation(uq)
        g = {x: i % k for i, x in enumerate(perm)}
        f = np.array([g[x] for x in q]); out.append(f)
    return out

def crossfit_grid(P, n_rep=10, k=5):
    """P: (n_params, n_rows) per-row points for every candidate setting. For each repeat/fold pick the best
    param on the other folds, score held rows. Returns per-row points averaged over repeats and the chosen params."""
    res = np.zeros(P.shape[1]); chosen = []
    for f in splits(n_rep, k):
        row = np.zeros(P.shape[1])
        for j in range(k):
            tr = f != j
            b = int(np.argmax(P[:, tr].mean(1))); chosen.append(b)
            row[~tr] = P[b, ~tr]
        res += row / n_rep
    return res, chosen

def crossfit_model(fit_apply, n_rep=5, k=5):
    """fit_apply(train_mask, held_mask) -> per-row points for held rows."""
    res = np.zeros(len(dev))
    for f in splits(n_rep, k):
        row = np.zeros(len(dev))
        for j in range(k):
            tr = f != j
            row[~tr] = fit_apply(tr, ~tr)
        res += row / n_rep
    return res
