"""Dev-only check: does the 235B zero-shot probe add to the offline 3-way ensemble? (fold 0, 301 rows)"""
import itertools, sys
from pathlib import Path
import numpy as np
from k12.core import aligned_bundle, label_space, make_folds, map3, normalize, read_csv

train, classes = read_csv('data/raw/train.csv'), label_space(read_csv('data/raw/taxonomy_v1.csv'))
dev = train.loc[make_folds(train) == 0].reset_index(drop=True)
ids, y = dev.trace_id.tolist(), dev.root_code
B = {'t32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 't8': 'runs/pod/tx/text_tx_dev_e3.npz',
     'img': 'runs/pod/dev-distill/validation_epoch_2.npz'}
probe = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/pod/probe235/validation.npz')
P = {k: aligned_bundle(v, ids, classes) for k, v in B.items()}
if probe.exists():
    P['probe'] = aligned_bundle(probe, ids, classes)
for k, p in P.items():
    print(f'{k:6s} {map3(y, p, classes):.4f}')

def geo(ws):
    lp = sum(w * np.log(np.clip(P[k], 1e-6, 1)) for k, w in ws.items())
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))

def arith(ws):
    return normalize(sum(w * P[k] for k, w in ws.items()))

base = {'t32': 1, 't8': 1, 'img': 1}
print('equal geo 3-way', round(map3(y, geo(base), classes), 4), ' arith', round(map3(y, arith(base), classes), 4))
grid = [0, .5, 1, 1.5, 2]
best3 = max((map3(y, geo(dict(zip(base, w))), classes), w) for w in itertools.product(grid, repeat=3) if sum(w))
print('best-grid geo 3-way (in-sample, optimistic)', round(best3[0], 4), best3[1])
if 'probe' in P:
    for w in (.1, .25, .5, 1, 2):
        print(f'equal 3-way + probe w={w}: geo {map3(y, geo({**base, "probe": w}), classes):.4f}'
              f'  arith {map3(y, arith({**base, "probe": w}), classes):.4f}')
    best4 = max((map3(y, geo(dict(zip(list(base) + ['probe'], w))), classes), w)
                for w in itertools.product(grid, repeat=4) if sum(w[:3]))
    print('best-grid geo 4-way (in-sample, optimistic)', round(best4[0], 4), best4[1])
    # paired bootstrap of equal-weight 3-way vs 3-way + probe(w=1)
    from k12.core import row_scores
    a = np.asarray(row_scores(y, [[classes[i] for i in r] for r in np.argsort(-geo(base), 1)[:, :3]]))
    b = np.asarray(row_scores(y, [[classes[i] for i in r] for r in np.argsort(-geo({**base, 'probe': 1}), 1)[:, :3]]))
    rng = np.random.default_rng(0); d = b - a
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f'probe w=1 gain {d.mean():+.4f}, 95% CI [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]')
