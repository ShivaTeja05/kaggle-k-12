"""T2: is the gold 'specific beats generic' convention usable as a decision rule on top of blend+Sinkhorn?
Rule: if top-1 is a generic code and a specific (topic) code sits in the top-K with p >= t * p_top1, promote it.
t and K chosen by 5-fold question-grouped cross-fitting; paired question bootstrap vs blend+Sinkhorn.
Also: convention statistics on train rows annotated with both a generic and a specific misconception."""
import json, sys
from collections import Counter
import numpy as np
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, boot_ci, ROOT, GENERIC, fam
from k12.core import row_scores, topk, read_csv
D = get_all(); dev, fit, y, classes, PS, tr = D['dev'], D['fit'], D['y'], D['classes'], D['PS'], D['tr']
gen = np.array([c != 'NONE' and fam(c) in GENERIC for c in classes]); spec = np.array([c != 'NONE' and fam(c) not in GENERIC for c in classes])
def apply(P, t, K):
    out = []
    for p in P:
        o = list(np.argsort(-p)[:3]); top = o[0]
        if gen[top]:
            cand = [j for j in np.argsort(-p)[1:K] if spec[j] and p[j] >= t * p[top]]
            if cand: j = cand[0]; o = [j] + [x for x in o if x != j]; o = o[:3]
        out.append([classes[i] for i in o])
    return np.array(out)
grid = [(t, K) for t in [.2, .3, .4, .5, .6, .7, .8, 1.01] for K in [2, 3, 5]]
g = dev.question_id.to_numpy(); uq = np.unique(g); rng = np.random.default_rng(0)
base = row_scores(y, topk(PS, classes)); runs = []
for rep in range(5):
    fold = dict(zip(rng.permutation(uq), np.arange(len(uq)) % 5)); f = np.array([fold[q] for q in g])
    cf = np.zeros(len(y)); picks = []
    for k in range(5):
        tr_m, te = f != k, f == k
        best = max(grid, key=lambda tk: row_scores(y[tr_m], apply(PS[tr_m], *tk)).mean()); picks.append(best)
        cf[te] = row_scores(y[te], apply(PS[te], *best))
    runs.append((cf, picks))
cf = np.mean([r[0] for r in runs], 0)
pair = Counter(zip(fit.question_id, fit.root_code))
new = np.array([c != 'NONE' and pair[(q, c)] == 0 for q, c in zip(dev.question_id, dev.root_code)])
res = {'base_blend_sk': float(base.mean()), 'crossfit_rule': float(cf.mean()), 'delta_ci': boot_ci(cf, base, g),
       'delta_new_pair_rows': boot_ci(cf[new], base[new], g[new]), 'n_new': int(new.sum()),
       'picks': [str(p) for r in runs for p in r[1]]}
for tk in [(.5, 3), (.3, 3), (.7, 2)]:
    s = row_scores(y, apply(PS, *tk)); res[f'insample_{tk}'] = [float(s.mean()), boot_ci(s, base, g)]
# where the generic/specific error actually is on dev
top1 = topk(PS, classes)[:, 0]
gt_gen = np.array([c != 'NONE' and fam(c) in GENERIC for c in y]); pr_gen = np.array([c != 'NONE' and fam(c) in GENERIC for c in top1])
gt_sp = (y != 'NONE') & ~gt_gen; pr_sp = (top1 != 'NONE') & ~pr_gen
res['dev_top1_generic_gold_specific'] = int((pr_gen & gt_sp).sum()); res['dev_top1_specific_gold_generic'] = int((pr_sp & gt_gen).sum())
res['dev_gold_specific_in_top3_when_top1_generic'] = int(sum(1 for i in np.where(pr_gen & gt_sp)[0] if y[i] in topk(PS, classes)[i]))
# convention on train: rows annotated with both a generic and a specific misconception
both = Counter()
for m in tr.misconceptions:
    try: ms = json.loads(m)
    except Exception: continue
    cats = [(x.get('category'), bool(x.get('root'))) for x in ms if x.get('category')]
    if any(fam(c) in GENERIC for c, _ in cats) and any(fam(c) not in GENERIC for c, _ in cats):
        r = [c for c, isr in cats if isr]
        both['root_specific' if r and fam(r[0]) not in GENERIC else 'root_generic' if r else 'no_root'] += 1
res['train_rows_with_both_generic_and_specific'] = dict(both)
print(json.dumps(res, indent=1)[:3000])
json.dump(res, open(ROOT / 'runs/deep/approach-W1/t2_specific_over_generic.json', 'w'), indent=1)
