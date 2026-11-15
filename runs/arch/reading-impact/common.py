import json
import numpy as np
from k12.core import read_csv, label_space, make_folds, aligned_bundle, row_scores, topk, normalize

ROOT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train = read_csv(ROOT + 'data/raw/train.csv')
classes = label_space(read_csv(ROOT + 'data/raw/taxonomy_v1.csv'))
fold = make_folds(train)
dev = train[fold == 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
y = dev.root_code.tolist()

B = {
 '32_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
 '32_gold': 'runs/pod/text32/validation_gold_epoch_1.npz',
 '8_tx_e2': 'runs/pod/tx/text_tx_dev_e2.npz',
 '8_tx_e3': 'runs/pod/tx/text_tx_dev_e3.npz',
 'vl': 'runs/pod/dev-distill/validation_epoch_2.npz',
}
for s in (0, 25, 100):
    B[f's{s}_tx'] = f'runs/pod/scaling/scale_{s}/validation_tx_dev_epoch_1.npz'
    B[f's{s}_gold'] = f'runs/pod/scaling/scale_{s}/validation_gold_epoch_1.npz'
P = {k: aligned_bundle(ROOT + v, ids, classes) for k, v in B.items()}

def rs(p):
    return row_scores(y, topk(p, classes))

def geo(parts):
    lp = sum(w * np.log(np.clip(P[k] if isinstance(k, str) else k, 1e-12, None)) for k, w in parts)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))

def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
