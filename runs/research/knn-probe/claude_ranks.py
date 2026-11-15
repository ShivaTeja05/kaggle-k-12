"""How often is gold at Claude's rank 1/2/3 on dev (for calibrated rank-soft distillation targets, CanDist-style)."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
tr, tax, classes, folds, dev, fit = load()
gold = dict(zip(dev.trace_id, dev.root_code)); fam = lambda c: c.split('-')[0]
h = np.zeros(4); hf = 0; n = 0; ne = 0
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); a = r['answer']; g = gold.get(r['trace_id'])
    if g is None: continue
    c = [a.get('first_code'), a.get('second_code'), a.get('third_code')]; n += 1
    k = c.index(g) if g in c else 3; h[k] += 1
    if k == 3 and g != 'NONE': ne += 1; hf += any(x and fam(x) == fam(g) for x in c)
out = {'n': n, 'p_rank1': h[0]/n, 'p_rank2': h[1]/n, 'p_rank3': h[2]/n, 'p_miss': h[3]/n,
       'misses_error_rows': ne, 'misses_where_a_claude_code_shares_family': int(hf)}
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT/'runs/research/knn-probe/claude_ranks.json', 'w'), indent=1)
