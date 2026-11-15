import glob, numpy as np
from common import *
devs = set(ids); fits = set(fit.trace_id)
for f in sorted(glob.glob(R+'runs/**/*.npz', recursive=True) + glob.glob(R+'artifacts/**/*.npz', recursive=True)):
    try:
        with np.load(f) as b: i = set(b['ids'].tolist()); p = b['p']
    except Exception: continue
    nf = len(i & fits)
    if nf > 0: print(f.replace(R, ''), 'rows', len(i), 'fit-rows', nf, 'dev-rows', len(i & devs))
f235 = load('f235'); print('f235 nonzero per row (>1e-4):', np.mean((f235 > 1e-4).sum(1)), 'max mean', f235.max(1).mean())
for n in ['t32', 't8e3', 'vl', 's100']:
    p = load(n); print(n, 'mean max prob', p.max(1).mean(), 'mean #>1e-3', (p > 1e-3).sum(1).mean())
