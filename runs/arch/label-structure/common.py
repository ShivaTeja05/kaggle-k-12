import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
from k12.core import read_csv, label_space, normalize, topk, row_scores, map3, make_folds, aligned_bundle

def load():
    tr = read_csv(ROOT/'data/raw/train.csv'); tax = read_csv(ROOT/'data/raw/taxonomy_v1.csv')
    classes = label_space(tax); folds = make_folds(tr)
    dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    return tr, tax, classes, folds, dev, fit

def blend(ids, classes, tx8='text_tx_dev_e3'):
    paths = [(ROOT/'runs/pod/text32/validation_tx_dev_epoch_1.npz', .35),
             (ROOT/f'runs/pod/tx/{tx8}.npz', .30),
             (ROOT/'runs/pod/dev-distill/validation_epoch_2.npz', .35)]
    lp = sum(w*np.log(aligned_bundle(p, ids, classes)) for p, w in paths)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))

def boot_ci(new_pts, ref_pts, groups=None, B=4000, seed=2026):
    d = np.asarray(new_pts) - np.asarray(ref_pts)
    rng = np.random.default_rng(seed)
    if groups is None:
        s = rng.integers(0, len(d), (B, len(d))); b = d[s].mean(1)
    else:
        g = pd.DataFrame({'g': groups, 'd': d}).groupby('g').d.agg(['sum', 'size']).to_numpy()
        s = rng.integers(0, len(g), (B, len(g))); b = g[s, 0].sum(1)/g[s, 1].sum(1)
    return float(d.mean()), [float(x) for x in np.quantile(b, [.025, .975])]
