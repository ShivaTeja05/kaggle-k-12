"""T4: which hard-negative source would put the dev blend's actual wrong top-1 next to the truth in training candidate sets?
For dev error rows with a wrong non-NONE top-1 (blend+Sinkhorn), check whether that wrong code is among the truth's top-k
neighbours under: Claude-pool co-ranking (first vs second/third code), taxonomy text similarity, same section, same-question fit codes."""
import json, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, ROOT
from k12.core import topk
D = get_all(); dev, fit, y, classes, tax, PS = D['dev'], D['fit'], D['y'], D['classes'], D['tax'], D['PS']
co = defaultdict(Counter)
for l in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    a = json.loads(l).get('answer') or {}; c = [a.get(k) for k in ('first_code', 'second_code', 'third_code')]
    c = [x for x in c if x and x != 'NONE']
    for i in range(len(c)):
        for j in range(len(c)):
            if i != j: co[c[i]][c[j]] += 1
for m in fit.misconceptions:  # gold co-annotation (root vs non-root on the same row)
    try: ms = json.loads(m)
    except Exception: continue
    cats = [x.get('category') for x in ms if x.get('category')]
    for a in cats:
        for b in cats:
            if a != b: co[a][b] += 3
sim = pd.read_csv(ROOT / 'runs/deep/learnability/code_similarity_tfidf.csv', index_col=0)
sec = dict(zip(tax.id, tax.section)); qcodes = defaultdict(Counter)
for q, c in zip(fit.question_id, fit.root_code): qcodes[q][c] += 1
top1 = topk(PS, classes)[:, 0]; rows = [i for i in range(len(y)) if y[i] != 'NONE' and top1[i] not in ('NONE', y[i])]
def nb_co(c, k): return [x for x, _ in co[c].most_common(k)]
def nb_sim(c, k): return list(sim.loc[c].drop(c).sort_values(ascending=False).index[:k]) if c in sim.index else []
res = {'n_rows': len(rows)}
for k in [3, 5, 8]:
    hit = Counter()
    for i in rows:
        t, p, q = y[i], top1[i], dev.question_id[i]
        hit['co_rank'] += p in nb_co(t, k); hit['tfidf'] += p in nb_sim(t, k)
        hit['union'] += p in set(nb_co(t, k)) | set(nb_sim(t, k))
        hit['union+qcodes'] += p in set(nb_co(t, k)) | set(nb_sim(t, k)) | set(qcodes[q])
    res[f'k{k}'] = {s: round(v / len(rows), 3) for s, v in hit.items()}
res['same_section'] = round(np.mean([sec.get(y[i]) == sec.get(top1[i]) for i in rows]), 3)
res['in_question_fit_codes'] = round(np.mean([top1[i] in qcodes[dev.question_id[i]] for i in rows]), 3)
res['top_confusions'] = Counter(f'{y[i]}->{top1[i]}' for i in rows).most_common(12)
print(json.dumps(res, indent=1)); json.dump(res, open(ROOT / 'runs/deep/approach-W1/t4_hard_negative_sources.json', 'w'), indent=1)
