"""Finite-difference check of w2_fit.f (ce and poe) on a 200-row subset with random parameters."""
import sys, numpy as np, pickle
src = open('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W2/w2_fit.py').read().split('t0 = time.time()')[0]
for mode in ['ce', 'poe']:
    sys.argv = ['x', mode, '3', '0.5', '1e-3']
    src2 = src.replace("tr, dv = C['tr'], C['dv']", "tr, dv = C['tr'], C['dv']\nidx=np.random.default_rng(1).choice(8505,300,replace=False)\ntr={k:(v[idx] if hasattr(v,'shape') else v) for k,v in tr.items()}")
    ns = {}; exec(src2, ns)
    rng = np.random.default_rng(0); th = rng.normal(0, .3, ns['P_'] + ns['F'] * ns['NC'])
    l, gr = ns['f'](th); errs = []
    for j in list(range(ns['P_'])) + list(rng.choice(len(th), 30)):
        e = np.zeros_like(th); e[j] = 1e-5
        num = (ns['f'](th + e)[0] - ns['f'](th - e)[0]) / 2e-5; errs.append(abs(num - gr[j]) / max(1e-6, abs(num) + abs(gr[j])))
    print(mode, 'loss', l, 'max rel err', max(errs), 'median', np.median(errs), 'theta errs', np.round(errs[:6],8))
