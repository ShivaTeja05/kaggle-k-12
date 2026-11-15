"""Headroom from a wider model pool: cross-fitted Caruana greedy selection over every dev bundle, add-one gains,
and oracle bounds. Reference = .35/.30/.35 blend (0.7182)."""
import json, itertools
from common import *
from cf import *
names = [n for n in BUNDLES if n != 't8e2']
M = {n: load(n) for n in names}; L = {n: np.log(np.maximum(M[n], 1e-12)) for n in names}
base = ['t32', 't8e3', 'vl']
ref = geo([M[n] for n in base], [.35, .30, .35]); refp = pts(ref)
rep = {}
def show(name, r, extra=None):
    d, c, pg = boot_ci(r, refp); rep[name] = {'map3': float(r.mean()), 'delta': d, 'ci95': c, 'p_gt0': pg, **(extra or {})}
    print(f'{name:50s} {r.mean():.4f} d={d:+.4f} CI[{c[0]:+.4f},{c[1]:+.4f}] P>0={pg:.2f}', flush=True)
# add-one to fixed reference with weight w (crossfit w in {0,.1,...,.5})
for n in names:
    if n in base: continue
    ws = [0, .1, .2, .3, .4, .5]
    P = np.array([pts(np.log(ref)*(1-w) + w*L[n]) for w in ws]); r, ch = crossfit_grid(P)
    show(f'ref+{n} (w crossfit)', r, {'insample_best': float(P.mean(1).max()), 'w_insample': ws[int(P.mean(1).argmax())]})
# Caruana greedy (with replacement), steps chosen inside each training fold
def caruana(tr, steps=12, pool=names):
    sel = []; acc = 0
    for _ in range(steps):
        best = None
        for n in pool:
            s = acc + L[n]; v = pts(s[tr], y[tr]).mean()
            if best is None or v > best[0]: best = (v, n)
        sel.append(best[1]); acc = acc + L[best[1]]
    return sel
def fa(tr, he):
    sel = caruana(tr); s = sum(L[n] for n in sel); return pts(s[he], y[he])
show('caruana_all15_crossfit', crossfit_model(fa, n_rep=3))
def fa3(tr, he):
    sel = caruana(tr, pool=base); s = sum(L[n] for n in sel); return pts(s[he], y[he])
show('caruana_base3_crossfit', crossfit_model(fa3, n_rep=3))
full = np.ones(len(dev), bool); sel = caruana(full); print('insample caruana selection', sel)
show('caruana_all15_insample(optimistic)', pts(sum(L[n] for n in sel)), {'sel': sel})
# fixed simple blends
show('equal_geo_t32_t8e3_vl_s100', pts(geo([M[n] for n in ['t32', 't8e3', 'vl', 's100']], [.25]*4)))
show('equal_geo_t32_s100_vl', pts(geo([M[n] for n in ['t32', 's100', 'vl']], [1/3]*3)))
show('equal_geo_t32_t8e3_vl_f235', pts(geo([M[n] for n in ['t32', 't8e3', 'vl', 'f235']], [.25]*4)))
show('equal_geo_t32_t8e3_vl_s100_f235', pts(geo([M[n] for n in ['t32', 't8e3', 'vl', 's100', 'f235']], [.2]*5)))
# oracle bounds
best_any = np.max([pts(M[n]) for n in names], 0); show('ORACLE best single model per row (all15)', best_any)
best_b = np.max([pts(M[n]) for n in base], 0); show('ORACLE best of base3 per row', best_b)
o = np.argsort(-ref, 1); rk = np.argmax(o == y[:, None], 1)
for k in [3, 5, 8]:
    show(f'ORACLE truth moved to #1 if in ref top{k}', np.where(rk < k, 1.0, 0.0))
json.dump(rep, open(R+'runs/arch/combiner-headroom/s4_pool.json', 'w'), indent=1)
