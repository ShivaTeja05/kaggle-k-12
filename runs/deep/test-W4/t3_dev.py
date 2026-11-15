"""W4 test T3 (CPU, dev 301 rows, adversarial). How much of the dev score is actually lost to 'a visible secondary error
beats the root', and is that loss caused by training on the late-biased Claude pool?
(a) Ceiling: oracle hard-negative rejection = delete the row's annotated non-root codes from the blend+Sinkhorn ranking.
    This is the most approach A(1) + B's margin loss could ever deliver on dev (O = oracle).
(b) Mechanism: pairwise root-vs-later-secondary preference for each member, and for the controlled 8B scaling runs with
    0% / 25% / 100% Claude pool labels (same recipe). If pool exposure caused W4, scale_100 should prefer the later
    secondary more often than scale_0.
(c) Deployable proxy: secondary-propensity demotion p(c) * exp(-beta * s_c), s_c = log((#non-root annotations of c + 1) /
    (#root annotations of c + 1)) from FIT rows only; beta cross-fitted (5 question-grouped folds x 5 repeats),
    before or after Sinkhorn. Paired question-grouped bootstrap vs blend+Sinkhorn (0.737).
Writes runs/deep/test-W4/t3_dev.json."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT / 'runs/deep/test-W4'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(ROOT / 'runs/deep/learnability'), str(ROOT / 'runs/deep/symbolic')]
from lcommon import get_all, boot_ci, sinkhorn
from k12.core import row_scores, topk, normalize, aligned_bundle
from symutil import root_lines

D = get_all(); dev, fit, y, classes, P, PS, preds = D['dev'], D['fit'], D['y'], D['classes'], D['P'], D['PS'], D['preds']
ci = {c: i for i, c in enumerate(classes)}; g = dev.question_id.to_numpy(); n = len(y)
base = row_scores(y, topk(PS, classes)); print('ref blend+Sinkhorn', base.mean())
rd = lambda x: round(float(x), 4)
ci_ = lambda a, b, m=None: (lambda r: [rd(r[0]), [rd(v) for v in r[1]]])(boot_ci(a[m] if m is not None else a, b[m] if m is not None else b, g[m] if m is not None else g))
pair = Counter(zip(fit.question_id, fit.root_code))
new = np.array([c != 'NONE' and pair[(q, c)] == 0 for q, c in zip(dev.question_id, dev.root_code)])

def stepnums(s):
    nums = [int(x) for x in re.findall(r'L(\d+)', str(s))]
    return list(range(nums[0], nums[1] + 1)) if len(nums) == 2 else nums
sec, sec_after = [], []
for r in dev.itertuples():
    ms = json.loads(r.misconceptions) if str(r.misconceptions).strip() else []
    rs, _ = root_lines(r.misconceptions); r0 = min(rs) if rs else None
    s = [m['category'] for m in ms if not m.get('root') and m.get('category') in ci and m['category'] != r.root_code]
    sa = [m['category'] for m in ms if not m.get('root') and m.get('category') in ci and m['category'] != r.root_code
          and r0 is not None and stepnums(m.get('step')) and min(stepnums(m['step'])) > r0]
    sec.append(list(dict.fromkeys(s))); sec_after.append(list(dict.fromkeys(sa)))
has_sec = np.array([bool(s) for s in sec]) & (y != 'NONE'); has_after = np.array([bool(s) for s in sec_after]) & (y != 'NONE')
res = {'ref_blend_sinkhorn': rd(base.mean()), 'n_sec_rows': int(has_sec.sum()), 'n_sec_after_rows': int(has_after.sum()),
       'n_new_pair_rows': int(new.sum())}

# ---------------- (a) oracle hard-negative ceiling
def drop(Pm, lists):
    Q = Pm.copy()
    for i, s in enumerate(lists):
        for c in s: Q[i, ci[c]] = 0
    return Q
for name, lists in [('drop_all_secondaries', sec), ('drop_after_root_secondaries', sec_after)]:
    s = row_scores(y, topk(drop(PS, lists), classes))
    res[f'a_oracle_{name}'] = {'map': rd(s.mean()), 'delta_all': ci_(s, base), 'delta_on_sec_rows_per_row': ci_(s, base, has_sec),
                               'rows_changed_score': int((s != base).sum())}
cl = preds['claude']; clp = row_scores(y, cl)
cl_drop = np.array([[c for c in list(r) + ['NONE'] * 3 if c not in s][:3] for r, s in zip(cl, sec)])
res['a_claude_same_oracle'] = {'claude_map': rd(clp.mean()), 'claude_drop_all_secondaries': rd(row_scores(y, cl_drop).mean())}
# lost-point anatomy on the secondary rows
t1 = topk(PS, classes)
top1_sec = np.array([t1[i, 0] in sec[i] for i in range(n)]) & has_sec
res['a_anatomy_sec_rows'] = {'blend_sk_map_on_sec_rows': rd(base[has_sec].mean()), 'claude_on_sec_rows': rd(clp[has_sec].mean()),
                             'rows_top1_is_secondary': int(top1_sec.sum()),
                             'points_lost_on_those_rows': rd((1 - base[top1_sec]).sum()),
                             'points_lost_on_other_sec_rows': rd((1 - base[has_sec & ~top1_sec]).sum()),
                             'claude_top1_is_secondary': int(sum(cl[i, 0] in sec[i] for i in np.where(has_sec)[0]))}

# ---------------- (b) mechanism: root vs later secondary preference
mats = {k: aligned_bundle(ROOT / p, dev.trace_id.tolist(), classes) for k, p in {
    'm32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 'm8': 'runs/pod/tx/text_tx_dev_e3.npz',
    'vl_distill': 'runs/pod/dev-distill/validation_epoch_2.npz',
    'scale_0': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
    'scale_25': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
    'scale_100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'}.items()}
mats['blend'] = P; mats['blend_sk'] = PS
idx = np.where(has_after)[0]
def pref(M):
    return np.array([float(max(M[i, ci[c]] for c in sec_after[i]) > M[i, ci[y[i]]]) for i in idx])
def logratio(M):
    return np.array([np.log(max(M[i, ci[c]] for c in sec_after[i])) - np.log(M[i, ci[y[i]]]) for i in idx])
prefs = {k: pref(M) for k, M in mats.items()}
res['b_pref_later_secondary_over_root'] = {k: rd(v.mean()) for k, v in prefs.items()}
res['b_mean_logratio_sec_minus_root'] = {k: rd(logratio(M).mean()) for k, M in mats.items()}
rank = lambda row, c: (list(row).index(c) if c in row else 3)
cp = np.array([float(min(rank(cl[i], c) for c in sec_after[i]) < rank(cl[i], y[i])) for i in idx])
res['b_pref_later_secondary_over_root']['claude(top3 ranks)'] = rd(cp.mean())
gi = g[idx]
for a, b in [('scale_100', 'scale_0'), ('scale_25', 'scale_0'), ('vl_distill', 'm32'), ('vl_distill', 'm8')]:
    res[f'b_diff_{a}_minus_{b}'] = [rd(x) if not isinstance(x, list) else [rd(v) for v in x] for x in boot_ci(prefs[a], prefs[b], gi)]
    res[f'b_logratio_diff_{a}_minus_{b}'] = [rd(x) if not isinstance(x, list) else [rd(v) for v in x] for x in boot_ci(logratio(mats[a]), logratio(mats[b]), gi)]
res['b_member_maps'] = {k: rd(row_scores(y, topk(M, classes)).mean()) for k, M in mats.items()}
print(json.dumps({k: v for k, v in res.items() if k.startswith(('a_', 'b_'))}, indent=1))

# ---------------- (c) deployable: secondary-propensity demotion, cross-fitted
rc, sc = Counter(), Counter()
for m in fit.misconceptions:
    for a in (json.loads(m) if str(m).strip() else []):
        c = a.get('category')
        if c in ci: (rc if a.get('root') else sc)[c] += 1
s_c = np.array([0. if c == 'NONE' else np.log((sc[c] + 1) / (rc[c] + 1)) for c in classes])
res['c_top_secondary_propensity'] = sorted([(c, rd(s_c[ci[c]]), sc[c], rc[c]) for c in classes], key=lambda x: -x[1])[:8]
pi = D['pi_fit']
def adj(beta, mode):
    if mode == 'post':
        return normalize(PS * np.exp(-beta * s_c))
    return sinkhorn(normalize(P * np.exp(-beta * s_c)), pi, .5)
grid = [0, .05, .1, .2, .3, .5, .75, 1., 1.5]
uq = np.unique(g); rng = np.random.default_rng(7)
for mode in ('post', 'pre'):
    cache = {b: row_scores(y, topk(adj(b, mode), classes)) for b in grid}
    cfs, picks = [], []
    for rep in range(5):
        fold = dict(zip(rng.permutation(uq), np.arange(len(uq)) % 5)); f = np.array([fold[q] for q in g]); cf = np.zeros(n)
        for k in range(5):
            trm, te = f != k, f == k
            best = max(grid, key=lambda b: (cache[b][trm].mean(), -b)); picks.append(best); cf[te] = cache[best][te]
        cfs.append(cf)
    cf = np.mean(cfs, 0)
    res[f'c_demote_{mode}'] = {'crossfit_map': rd(cf.mean()), 'delta_all': ci_(cf, base), 'delta_sec_rows_per_row': ci_(cf, base, has_sec),
                               'delta_new_pair_per_row': ci_(cf, base, new), 'beta_picks': Counter(map(str, picks)),
                               'insample_curve': {str(b): rd(cache[b].mean()) for b in grid}}
    print(mode, json.dumps(res[f'c_demote_{mode}']))
json.dump(res, open(OUT / 't3_dev.json', 'w'), indent=1, default=str)
