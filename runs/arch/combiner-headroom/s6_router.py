"""Per-row confidence routing / confidence-weighted blends / rank (Borda) blends; cross-fitted. Plus loss breakdown."""
import json
from common import *
from cf import *
base = ['t32', 't8e3', 'vl']
M = {n: load(n) for n in base + ['f235', 's100']}; L = {n: np.log(np.maximum(M[n], 1e-12)) for n in M}
W = {'t32': .35, 't8e3': .30, 'vl': .35}
ref = geo([M[n] for n in base], [.35, .30, .35]); refp = pts(ref)
rep = {}
def show(name, r, extra=None):
    d, c, pg = boot_ci(r, refp); rep[name] = {'map3': float(r.mean()), 'delta': d, 'ci95': c, 'p_gt0': pg, **(extra or {})}
    print(f'{name:50s} {r.mean():.4f} d={d:+.4f} CI[{c[0]:+.4f},{c[1]:+.4f}] P>0={pg:.2f}', flush=True)
def confw(names, g, w=None):
    w = w or {n: 1/len(names) for n in names}
    conf = {n: M[n].max(1) ** g for n in names}
    s = sum(w[n]*conf[n][:, None]*L[n] for n in names) / sum(w[n]*conf[n] for n in names)[:, None]
    return s
gs = [0, .5, 1, 2, 4, 8]
P = np.array([pts(confw(base, g, W)) for g in gs]); r, ch = crossfit_grid(P)
show('confidence-weighted geo (max-prob^g)', r, {'insample': [float(x) for x in P.mean(1)]})
route = np.argmax(np.stack([M[n].max(1) for n in base]), 0)
show('route: most-confident model only', pts(np.stack([M[n] for n in base])[route, np.arange(len(dev))]))
def borda(names, w):
    return sum(w[n] / (np.argsort(np.argsort(-M[n], 1), 1) + 1 + 2) for n in names)
show('weighted reciprocal-rank blend (fixed .35/.30/.35)', pts(borda(base, W)))
# loss breakdown of reference
none = np.array(classes)[y] == 'NONE'; o = np.argsort(-ref, 1)
fam_ok = FAM[o[:, 0]] == FAM[y]
print('ref NONE rows map3 %.4f (n=%d) | error rows map3 %.4f (n=%d)' % (refp[none].mean(), none.sum(), refp[~none].mean(), (~none).sum()))
print('error rows predicted NONE first:', int((o[~none, 0] == ci['NONE']).sum()), '| NONE rows with error first:', int((o[none, 0] != ci['NONE']).sum()))
print('error rows: top1 family correct %.3f, top1 exact %.3f' % (fam_ok[~none].mean(), (o[~none, 0] == y[~none]).mean()))
lost = 1 - refp
print('points lost: NONE rows %.1f, error rows wrong family %.1f, error rows right family wrong code %.1f, error rows predicted NONE %.1f' % (
    lost[none].sum(), lost[~none & ~fam_ok & (o[:, 0] != ci['NONE'])].sum(), lost[~none & fam_ok].sum(), lost[~none & (o[:, 0] == ci['NONE'])].sum()))
rep['loss_breakdown'] = {'none_rows_map3': float(refp[none].mean()), 'error_rows_map3': float(refp[~none].mean())}
json.dump(rep, open(R+'runs/arch/combiner-headroom/s6_router.json', 'w'), indent=1)
