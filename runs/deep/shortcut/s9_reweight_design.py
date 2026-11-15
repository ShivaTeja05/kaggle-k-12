"""(9) CPU design check for anti-shortcut row weights in the candidate-student data (data/v5/cand_train.jsonl).
Weights: debiased-focal style w = (1 - p_q(truth))^gamma (Karimi Mahabadi et al. 2020), where p_q is the leave-own-row-out
question prior from fit gold + Claude first codes (alpha=2 smoothing to the global code mix), and inverse pair frequency
w = 1/sqrt(n(q,truth)). Reported: weighted 'unseen' likelihood ratio (dev error rows: 0.49), weighted new-pair share of error
rows (dev vs fit+Claude: 0.37), and Kish effective sample size. NONE rows keep weight 1 (their prior is genuine signal)."""
import json
from collections import Counter
import numpy as np
from sc_common import load, claude_unl, OUT, ROOT, r
tr, tax, classes, dev, fit = load()
cl, unl = claude_unl(); cl_ok = cl[~cl.trace_id.isin(set(dev.trace_id))]
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
qcount = {}
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount.setdefault(q, Counter())[c] += 1
g = Counter(fit.root_code) + Counter(cl_ok.root_code); tot = sum(g.values())
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))
L = 'ABCDEFGHIJ'
rows = []
for line in open(ROOT / 'data/v5/cand_train.jsonl'):
    x = json.loads(line); truth = x['codes'][L.index(x['target'][-1])]
    cnt = Counter(qcount.get(qmap[x['trace_id']], Counter())); cnt[own.get(x['trace_id'])] -= 1
    n = sum(v for v in cnt.values() if v > 0)
    p = (max(cnt.get(truth, 0), 0) + 2 * g[truth] / tot) / (n + 2)
    rows.append(dict(src=x['source'], none=truth == 'NONE', w0=x['weight'], p=p, pair=max(cnt.get(truth, 0), 0) + 1,
                     seen_t=cnt.get(truth, 0) > 0, seen_d=[cnt.get(c, 0) > 0 for c in x['codes'] if c not in (truth, 'NONE')]))
def summary(wfun):
    e = [x for x in rows if not x['none']]
    w = np.array([x['w0'] * wfun(x) for x in e])
    st = np.array([x['seen_t'] for x in e]); sd = np.array([np.mean(x['seen_d']) for x in e])
    pt = np.average(st, weights=w); pd_ = np.average(sd, weights=w)
    wall = np.array([x['w0'] * (1. if x['none'] else wfun(x)) for x in rows])
    return dict(LR_unseen=r((1 - pt) / (1 - pd_)), new_pair_share_error_rows=r(1 - pt),
                error_weight_share=r(w.sum() / wall.sum()), kish_ess_error_rows=int(w.sum() ** 2 / (w ** 2).sum()), error_rows=len(e))
res = {'none (current)': summary(lambda x: 1.)}
for gam in [1, 2, 4]:
    res[f'focal_gamma{gam}'] = summary(lambda x, gam=gam: (1 - x['p']) ** gam)
res['inv_sqrt_pair_count'] = summary(lambda x: 1 / np.sqrt(x['pair']))
res['inv_pair_count'] = summary(lambda x: 1 / x['pair'])
res['new_pair_x5'] = summary(lambda x: 5. if not x['seen_t'] else 1.)
res['dev_reference'] = dict(LR_unseen=0.49, new_pair_share_error_rows_vs_fit_plus_claude=0.37)
json.dump(res, open(OUT / 's9_reweight_design.json', 'w'), indent=1); print(json.dumps(res, indent=1))
