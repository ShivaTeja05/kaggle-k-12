"""T5: how many rows would yield a 'shortcut' preference pair (wrong top-1 that is one of the question's fit codes, truth in top-10)?
Dev (out-of-sample blend) as a proxy for what out-of-fold student samples on fit rows would give; also what current v5 candidate recipe covers."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, ROOT, fam
D = get_all(); dev, fit, y, classes, tax, PS = D['dev'], D['fit'], D['y'], D['classes'], D['tax'], D['PS']
qc = defaultdict(set)
for q, c in zip(fit.question_id, fit.root_code): qc[q].add(c)
sec = dict(zip(tax.id, tax.section)); o = np.argsort(-PS, 1)
top1 = np.array([classes[r[0]] for r in o]); top10 = [set(classes[j] for j in r[:10]) for r in o]
wrong_q = np.array([top1[i] != y[i] and top1[i] in qc[dev.question_id[i]] for i in range(len(y))])
res = {'rows': len(y), 'wrong_top1_is_question_fit_code': int(wrong_q.sum()),
       'and_truth_in_top10': int(sum(wrong_q[i] and y[i] in top10[i] for i in range(len(y)))),
       'of_which_truth_NONE': int(sum(wrong_q[i] and y[i] == 'NONE' for i in range(len(y)))),
       'of_which_top1_NONE': int(sum(wrong_q[i] and top1[i] == 'NONE' for i in range(len(y))))}
# current v5 recipe coverage of the actual wrong top-1 (question fit codes up to 4 by frequency, or same section as truth)
rows = [i for i in range(len(y)) if y[i] != 'NONE' and top1[i] not in ('NONE', y[i])]
fq = defaultdict(Counter)
for q, c in zip(fit.question_id, fit.root_code): fq[q][c] += 1
v5 = [top1[i] in [c for c, _ in fq[dev.question_id[i]].most_common(5) if c != y[i]][:4] or sec.get(top1[i]) == sec.get(y[i]) for i in rows]
res['v5_recipe_covers_wrong_top1'] = round(float(np.mean(v5)), 3); res['n_wrong_nonNONE'] = len(rows)
print(res); json.dump(res, open(ROOT / 'runs/deep/approach-W1/t5_preference_pairs.json', 'w'), indent=1)
