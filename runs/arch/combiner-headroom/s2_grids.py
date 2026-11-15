"""Grid-type post-processors, each cross-fitted within dev; reference = fixed .35/.30/.35 blend (0.7182)."""
import itertools, json
from common import *
from priors import *
from cf import *

M = {n: load(n) for n in ['t32', 't8e3', 'vl']}
L = {n: np.log(np.maximum(p, 1e-12)) for n, p in M.items()}
ref = geo(list(M.values()), [.35, .30, .35]); refp = pts(ref)
lref = np.log(np.maximum(ref, 1e-12))
gp = gold_prior(); cp = claude_prior(a=5)
lgp, lcp = np.log(gp), np.log(cp)
report = {'reference_map3': float(refp.mean())}

def show(name, rowpts, extra=None):
    d, ci_, pg = boot_ci(rowpts, refp)
    report[name] = {'map3': float(rowpts.mean()), 'delta': d, 'ci95': ci_, 'p_gt0': pg, **(extra or {})}
    print(f'{name:45s} {rowpts.mean():.4f}  d={d:+.4f} CI[{ci_[0]:+.4f},{ci_[1]:+.4f}] P>0={pg:.2f}', flush=True)

# A. model weights (= per-model temperatures in a geometric blend), crossfit
grid = [w for w in itertools.product(np.arange(0, 1.01, .05), repeat=3) if abs(sum(w) - 1) < 1e-6]
PA = np.array([pts(sum(wi*L[n] for wi, n in zip(w, M))) for w in grid])
r, ch = crossfit_grid(PA); show('A_weights_crossfit', r)
show('A_weights_insample_best(optimistic)', PA[PA.mean(1).argmax()], {'w': list(map(float, grid[PA.mean(1).argmax()]))})
# arithmetic vs geometric
show('A_arith_.35/.30/.35(fixed)', pts(.35*M['t32'] + .30*M['t8e3'] + .35*M['vl']))

# B. log-linear question prior (gold fit rows) + Claude-unlabelled prior
aa = [0, .05, .1, .15, .2, .3, .4, .5, .7, 1.0]
PB = np.array([pts(lref + a*lgp) for a in aa]); r, ch = crossfit_grid(PB); show('B_gold_prior_loglinear', r, {'chosen': sorted(set(aa[c] for c in ch))})
for a, row in zip(aa, PB): print('   a=%.2f %.4f' % (a, row.mean()))
PBc = np.array([pts(lref + a*lcp) for a in aa]); r, ch = crossfit_grid(PBc); show('B_claude_prior_loglinear', r, {'chosen': sorted(set(aa[c] for c in ch))})
for a, row in zip(aa, PBc): print('   a=%.2f %.4f' % (a, row.mean()))
pairs = [(a, b) for a in aa for b in aa]
PB2 = np.array([pts(lref + a*lgp + b*lcp) for a, b in pairs]); r, ch = crossfit_grid(PB2); show('B_gold+claude_prior', r, {'chosen': sorted(set(pairs[c] for c in ch))})
# prior strengths of the hierarchical prior
hp = [(qa, tb) for qa in [.5, 1, 2, 4, 8] for tb in [2, 8, 32]]
PB3 = []
keys = []
for qa, tb in hp:
    l = np.log(gold_prior(qa, tb))
    for a in aa[1:]:
        PB3.append(pts(lref + a*l)); keys.append((qa, tb, a))
PB3 = np.array([refp] + PB3); keys = [(0, 0, 0)] + keys
r, ch = crossfit_grid(PB3); show('B_gold_prior_tuned_strengths', r, {'chosen': sorted(set(keys[c] for c in ch))})
# arithmetic mix (holdout_calibration.calibrated style)
mix = [0, .05, .1, .15, .2, .3]
PBa = np.array([pts((1-w)*ref + w*gp) for w in mix]); r, ch = crossfit_grid(PBa); show('B_gold_prior_arith_mix', r)

# C. NONE gating: shift log-odds of NONE
dd = np.arange(-3, 3.01, .25)
def shift(lp, d, cls='NONE'):
    lp = lp.copy(); lp[:, ci[cls]] += d; return lp
PC = np.array([pts(shift(lref, d)) for d in dd]); r, ch = crossfit_grid(PC); show('C_NONE_shift', r, {'chosen': sorted(set(float(dd[c]) for c in ch))})
# NONE: put NONE at fixed position if prob above threshold? covered by shift. Also family-level bias
fams = sorted(set(FAM))
def fam_bias_cv():
    # coordinate ascent per family bias on train folds, grid [-1..1]
    steps = [-1, -.5, -.25, 0, .25, .5, 1]
    def fa(tr, he):
        b = np.zeros(K)
        for _ in range(2):
            for f in fams:
                m = FAM == f; best = None
                for s in steps:
                    bb = b.copy(); bb[m] = s
                    v = pts(lref[tr] + bb, y[tr]).mean()
                    if best is None or v > best[0] + 1e-9: best = (v, s)
                b[m] = best[1]
        return pts(lref[he] + b, y[he])
    return crossfit_model(fa, n_rep=3)
show('C_family_bias_coordasc', fam_bias_cv())

# D. diversity rerank on positions 2-3: if pos-2 shares family with pos-1 and p2/p1 < t, demote it below the next other-family code
def diversify(p, t, scope='fam'):
    o = np.argsort(-p, 1)[:, :10]; out = np.zeros_like(p)
    for i in range(len(p)):
        top = list(o[i]); first = top[0]; sel = [first]
        rest = top[1:]
        while len(sel) < 3 and rest:
            c = rest[0]
            if FAM[c] in {FAM[s] for s in sel} and p[i, c] / p[i, first] < t:
                alt = [d for d in rest if FAM[d] not in {FAM[s] for s in sel}]
                if alt: c = alt[0]
            sel.append(c); rest.remove(c)
        for r_, c in enumerate(sel): out[i, c] = 3 - r_
        out[i] += 1e-9 * p[i]
    return out
tt = [0, .05, .1, .2, .3, .5, .7, 1.01]
PD = np.array([pts(diversify(ref, t)) for t in tt]); r, ch = crossfit_grid(PD); show('D_family_diversity_rerank', r, {'chosen': sorted(set(tt[c] for c in ch))})
for t, row in zip(tt, PD): print('   t=%.2f %.4f' % (t, row.mean()))
# NONE-in-top3 rule: if NONE not in top3 and p(NONE) > t*p(top1), insert it at position 3
def none_slot(p, t, pos=2):
    o = np.argsort(-p, 1); out = np.zeros_like(p); n = ci['NONE']
    for i in range(len(p)):
        sel = list(o[i, :3])
        if n not in sel and p[i, n] >= t * p[i, sel[0]]: sel = sel[:pos] + [n] + sel[pos:2]
        for r_, c in enumerate(sel[:3]): out[i, c] = 3 - r_
        out[i] += 1e-9 * p[i]
    return out
tt2 = [0, .001, .01, .03, .1, 2]
PE = np.array([pts(none_slot(ref, t)) for t in tt2]); r, ch = crossfit_grid(PE); show('D_NONE_slot3_rule', r, {'chosen': sorted(set(tt2[c] for c in ch))})
for t, row in zip(tt2, PE): print('   t=%.3f %.4f' % (t, row.mean()))

# E. Joint best: weights x gold prior x NONE shift (coarse) crossfit
grid2 = [w for w in itertools.product(np.arange(0, 1.01, .1), repeat=3) if abs(sum(w) - 1) < 1e-6]
combos = [(w, a, d) for w in grid2 for a in [0, .1, .2, .3, .5] for d in [-1, -.5, 0, .5, 1]]
PJ = np.array([pts(sum(wi*L[n] for wi, n in zip(w, M)) + a*lgp + shift(np.zeros_like(lref), d)) for w, a, d in combos])
r, ch = crossfit_grid(PJ); show('E_joint_weights+prior+NONE', r)
show('E_joint_insample(optimistic)', PJ[PJ.mean(1).argmax()], {'setting': str(combos[PJ.mean(1).argmax()])})
json.dump(report, open(R+'runs/arch/combiner-headroom/s2_grids.json', 'w'), indent=1, default=str)
