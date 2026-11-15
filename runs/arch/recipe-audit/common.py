"""Shared loaders for the recipe audit (CPU only, reads existing files, writes only under runs/arch/recipe-audit)."""
import json
from pathlib import Path

import sys
sys.path[:0] = ["/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/src", "/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/scripts"]
import numpy as np

from k12.core import aligned_bundle, label_space, make_folds, map3, normalize, read_csv, row_scores, topk

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
OUT = ROOT / 'runs/arch/recipe-audit'

train = read_csv(ROOT / 'data/raw/train.csv')
taxonomy = read_csv(ROOT / 'data/raw/taxonomy_v1.csv')
classes = label_space(taxonomy)
ci = {c: i for i, c in enumerate(classes)}
folds = make_folds(train)
dev = train.loc[folds == 0].reset_index(drop=True)
fit = train.loc[folds != 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
truth = dev.root_code.to_numpy()
fam = lambda c: c.split('-')[0].upper()

BUNDLES = {
    'q32_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
    'q8_tx_e3': 'runs/pod/tx/text_tx_dev_e3.npz',
    'q8_tx_e2': 'runs/pod/tx/text_tx_dev_e2.npz',
    'vl_distill': 'runs/pod/dev-distill/validation_epoch_2.npz',
    's100_tx': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
    's100_gold': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz',
    's0_tx': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
    's0_gold': 'runs/pod/scaling/scale_0/validation_gold_epoch_1.npz',
    's25_tx': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
    'q32_gold': 'runs/pod/text32/validation_gold_epoch_1.npz',
}


def load(name):
    return aligned_bundle(ROOT / BUNDLES[name], ids, classes)


def geo(parts):
    lp = sum(w * np.log(np.maximum(p, 1e-12)) for p, w in parts)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


def points(p, y=None):
    return row_scores(truth if y is None else y, topk(p, classes))


def claude_rank(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        a = r['answer']
        out[r['trace_id']] = [a.get('first_code'), a.get('second_code'), a.get('third_code')]
    return out


def question_bootstrap(delta, frame=dev, n=4000, seed=2026):
    """Paired bootstrap over questions (rows of one question move together)."""
    q = frame.question_id.to_numpy()
    keys, inv = np.unique(q, return_inverse=True)
    s = np.bincount(inv, weights=delta); c = np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(keys), (n, len(keys)))
    boot = s[idx].sum(1) / c[idx].sum(1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]
