"""Shared helpers for the adversarial test of the design-pragmatist proposal (CPU only, read-only on inputs)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from k12.core import read_csv, label_space, normalize, topk, row_scores, map3, make_folds, aligned_bundle

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
OUT = ROOT/'runs/arch/test-pragmatist'


def load():
    tr = read_csv(ROOT/'data/raw/train.csv'); tax = read_csv(ROOT/'data/raw/taxonomy_v1.csv')
    classes = label_space(tax); folds = make_folds(tr)
    dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    return tr, tax, classes, folds, dev, fit


def geo(parts):
    lp = sum(w*np.log(np.clip(P, 1e-12, None)) for P, w in parts)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


def blend(ids, classes, tx8='text_tx_dev_e3'):
    paths = [(ROOT/'runs/pod/text32/validation_tx_dev_epoch_1.npz', .35),
             (ROOT/f'runs/pod/tx/{tx8}.npz', .30),
             (ROOT/'runs/pod/dev-distill/validation_epoch_2.npz', .35)]
    return geo([(aligned_bundle(p, ids, classes), w) for p, w in paths])


def boot_ci(new_pts, ref_pts, groups=None, B=4000, seed=2026):
    """Paired bootstrap of the mean delta, resampling question groups when given."""
    d = np.asarray(new_pts, float) - np.asarray(ref_pts, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        s = rng.integers(0, len(d), (B, len(d))); b = d[s].mean(1)
    else:
        g = pd.DataFrame({'g': groups, 'd': d}).groupby('g').d.agg(['sum', 'size']).to_numpy()
        s = rng.integers(0, len(g), (B, len(g))); b = g[s, 0].sum(1)/g[s, 1].sum(1)
    return [float(d.mean()), [float(x) for x in np.quantile(b, [.025, .975])]]


def sinkhorn(P, target, lam=.5, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P*np.exp(b)); b += np.log(target/Q.mean(0))
    return normalize(P*np.exp(lam*b))


def qfolds(qid, k=5, seed=7):
    qs = np.unique(qid); np.random.default_rng(seed).shuffle(qs)
    g = {q: i % k for i, q in enumerate(qs)}
    return np.array([g[q] for q in qid])


def crossfit(score_by_cfg, mf, k=5):
    """score_by_cfg: {cfg: per-row points}. Pick cfg on the other folds, score on the held-out fold."""
    keys = list(score_by_cfg)
    n = len(next(iter(score_by_cfg.values())))
    out = np.empty(n); ch = []
    for f in range(k):
        m = mf != f
        best = max(keys, key=lambda c: score_by_cfg[c][m].mean()); ch.append(best)
        out[~m] = score_by_cfg[best][~m]
    return out, ch


CODE = re.compile(r'\b[A-Za-z]{2,3}-\d{2}\b|\bNONE\b')


def claude_text(a):
    return CODE.sub(' ', (a.get('root_line') or '') + ' || ' + (a.get('reasoning') or ''))


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def dump(obj, name):
    json.dump(obj, open(OUT/name, 'w'), indent=1, default=float)
    print(json.dumps(obj, indent=1, default=float))
