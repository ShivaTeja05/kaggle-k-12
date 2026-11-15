"""Paired, question-grouped bootstrap contrasts between open-model probes on dev segments (CPU)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'scripts', 'runs/arch/label-structure']
from common import load, blend, boot_ci
from k12.core import aligned_bundle, topk, row_scores
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy(); q = dev.question_id.to_numpy()
qc = fit.groupby('question_id').root_code.apply(set).to_dict()
bins = np.where(truth == 'NONE', 'NONE', np.where(~pd.Series(q).isin(qc).to_numpy(), 'q_unseen',
        np.where([t in qc.get(a, set()) for t, a in zip(truth, q)], 'pair_seen', 'pair_unseen')))
S = {'blend': row_scores(truth, topk(blend(ids, classes), classes))}
for d in ['zeroshot32', 'fewshot32', 'zeroshot235', 'fewshot235']:
    S[d] = row_scores(truth, topk(aligned_bundle(f'runs/pod/probe235/{d}/validation.npz', ids, classes), classes))
# thinking length vs correctness
tok = {}
for d in ['zeroshot32', 'zeroshot235', 'fewshot235']:
    r = {json.loads(l)['trace_id']: json.loads(l).get('tokens') for l in open(f'runs/pod/probe235/{d}/responses.jsonl')}
    tok[d] = np.array([r.get(i, np.nan) or np.nan for i in ids], float)
out = {}
def c(a, b, seg):
    m = np.isin(bins, seg); return boot_ci(S[a][m], S[b][m], q[m])
for seg in [['pair_unseen'], ['pair_seen'], ['NONE'], ['pair_unseen', 'q_unseen']]:
    k = '+'.join(seg)
    out[k] = {'zs235-zs32': c('zeroshot235', 'zeroshot32', seg), 'fs235-fs32': c('fewshot235', 'fewshot32', seg),
              'fs235-zs235 (same-question examples)': c('fewshot235', 'zeroshot235', seg),
              'fs32-zs32': c('fewshot32', 'zeroshot32', seg), 'zs235-blend': c('zeroshot235', 'blend', seg)}
for d, t in tok.items():
    m = ~np.isnan(t); s = S[d]
    med = np.nanmedian(t)
    out[f'tokens_{d}'] = {'median_tokens': float(med),
        'map_long_half': round(float(s[m & (t > med)].mean()), 3), 'map_short_half': round(float(s[m & (t <= med)].mean()), 3),
        'spearman_tokens_vs_score_error_rows': round(float(pd.Series(t[m & (bins != 'NONE')]).rank().corr(pd.Series(s[m & (bins != 'NONE')]).rank())), 3)}
Path('runs/deep/open-models/contrasts.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
