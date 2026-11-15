from common import *
import json, glob
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
ci = {c: i for i, c in enumerate(classes)}
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
def bal(P, lam):
    b = np.zeros(P.shape[1])
    for _ in range(50): b += np.log(pi / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))
out = {}
paths = sorted(glob.glob(str(ROOT/'runs/pod/scaling/scale_*/validation_tx_dev_epoch_1.npz'))) + [str(ROOT/'runs/pod/probe235/fewshot235/validation.npz'), str(ROOT/'runs/vision-development/k12/validation_epoch_3.npz')] + sorted(glob.glob(str(ROOT/'runs/pod/dev-*/validation_epoch_*.npz')))
for p in paths:
    try:
        P = aligned_bundle(p, ids, classes)
    except Exception as e:
        continue
    out[p.replace(str(ROOT) + '/', '')] = {str(l): round(map3(y, bal(P, l), classes), 4) for l in (0, .5, 1.)}
print(json.dumps(out, indent=1)); json.dump(out, open('balance_generality.json', 'w'), indent=1)
