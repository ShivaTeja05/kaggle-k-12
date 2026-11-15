"""Claude's answers restricted to the blend's top-k candidates, by mechanism (oracle diagnostic)."""
import sys, numpy as np, pandas as pd
from common import load, blend, ROOT
from k12.core import topk, row_scores
sys.path.insert(0, str(ROOT/'scripts')); from evaluate_claude import claude_scores
tr, tax, classes, folds, dev, fit = load(); ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
d = pd.read_csv(ROOT/'runs/deep/casebook/casebook_labelled.csv', keep_default_na=False)
P = blend(ids, classes); C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
out = {}
for k in [5, 10, 15, 20, 76]:
    top = np.argsort(-P, 1)[:, :k]; m = np.full_like(C, 1e-9); np.put_along_axis(m, top, 1, 1); Cr = C*m
    s = row_scores(y, topk(Cr/Cr.sum(1, keepdims=True), classes)); d[f'k{k}'] = s
    out[k] = round(float(s.mean()), 4)
print('overall', out)
print(d.groupby('mech')[[f'k{k}' for k in [5, 10, 15, 20, 76]]].mean().round(3).to_string())
print('truth recall@k on misses by mech'); print(d[d.mech != 'OK'].groupby('mech').rank_truth.apply(lambda r: [round((r <= k).mean(), 2) for k in (3, 10, 20)]).to_string())
