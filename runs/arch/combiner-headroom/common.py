import numpy as np, pandas as pd
from k12.core import read_csv, label_space, make_folds, aligned_bundle, normalize, topk, row_scores, map3

R = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train = read_csv(R+'data/raw/train.csv')
tax = read_csv(R+'data/raw/taxonomy_v1.csv')
classes = label_space(tax)
folds = make_folds(train)
fit = train.loc[folds != 0].reset_index(drop=True)
dev = train.loc[folds == 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
ci = {c: i for i, c in enumerate(classes)}
y = dev.root_code.map(ci).to_numpy()
K = len(classes)

BUNDLES = {
 't32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
 't8e3': 'runs/pod/tx/text_tx_dev_e3.npz',
 't8e2': 'runs/pod/tx/text_tx_dev_e2.npz',
 'vl': 'runs/pod/dev-distill/validation_epoch_2.npz',
 's0': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
 's25': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
 's100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
 'f235': 'runs/pod/probe235/fewshot235/validation.npz',
 'z235': 'runs/pod/probe235/zeroshot235/validation.npz',
 'f32': 'runs/pod/probe235/fewshot32/validation.npz',
 'vlv3': 'runs/pod/dev-v3/validation_epoch_3.npz',
 'vlaux': 'runs/pod/dev-aux/validation_epoch_3.npz',
 'vlcls': 'runs/pod/dev-cls/validation_epoch_3.npz',
 'vlread': 'runs/pod/dev-read/validation_epoch_3.npz',
 'vision': 'runs/vision-development/k12/validation_epoch_3.npz',
}

def load(name):
    return aligned_bundle(R+BUNDLES[name], ids, classes)

def pts(p, yy=None):
    yy = y if yy is None else yy
    order = np.argsort(-p, axis=1, kind='stable')[:, :3]
    hit = order == yy[:, None]
    w = np.array([1, .5, 1/3])
    return (hit * w).sum(1)

def geo(ps, ws):
    lp = sum(w*np.log(np.maximum(p, 1e-12)) for p, w in zip(ps, ws))
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))

def boot_ci(new_pts, ref_pts, groups=None, n=4000, seed=2026):
    groups = dev.question_id.to_numpy() if groups is None else groups
    d = pd.DataFrame({'g': groups, 'd': new_pts - ref_pts}).groupby('g').d.agg(['sum', 'size']).to_numpy()
    s = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    b = d[s, 0].sum(1) / d[s, 1].sum(1)
    return float((new_pts-ref_pts).mean()), [float(x) for x in np.quantile(b, [.025, .975])], float((b > 0).mean())

def family(c):
    return c.split('-', 1)[0].upper()
FAM = np.array([family(c) for c in classes])
