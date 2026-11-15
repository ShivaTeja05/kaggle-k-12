"""W2 probe, part 3: error-conditional bias product (PoE / learned-mixin style) in the prior-only proxy.
Training logits get + lam * centred log p(code | question, error) on non-NONE candidates (NONE offset 0); the offset is
dropped at dev time. Fixes the -inf of probe 2 with add-one global smoothing. Also a 'PoE + group x3' combination."""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/approach-W2')
import w2_probe as P
from sc_common import r
from common import boot_ci
T, D, S, dev, qcount, own, g, tot = P.T, P.D, P.S, P.dev, P.qcount, P.own, P.g, P.tot
for x, line in zip(T, open(P.ROOT / 'data/v5/cand_train.jsonl')):
    x['tid'] = json.loads(line)['trace_id']
ge = tot - g['NONE']
def err_lp(codes, cnt):
    ne = sum(v for k, v in cnt.items() if v > 0 and k != 'NONE')
    lp = np.array([0. if c == 'NONE' else np.log((max(cnt.get(c, 0), 0) + 2 * (g[c] + 1) / (ge + 76)) / (ne + 2)) for c in codes])
    m = np.array([c != 'NONE' for c in codes]); lp[m] -= lp[m].mean(); return lp
for x in T:
    c = Counter(qcount[x['q']]); c[own.get(x['tid'])] -= 1; x['cnt'] = c; x['lp'] = err_lp(x['codes'], c)
    x['seen_t'] = x['truth'] != 'NONE' and c.get(x['truth'], 0) > 0
w0 = np.array([x['w0'] for x in T]); e = np.array([x['truth'] != 'NONE' for x in T])
def rescale(w):
    w = np.array(w, float); tgt = w0[e].sum() / w0.sum(); a = tgt * w[~e].sum() / ((1 - tgt) * w[e].sum()); w[e] *= a; return w
base = P.dev_scores(P.fit_cl(T, w0), D); res = {}
def report(name, w, off):
    th = P.fit_cl(T, w, offset=off); sc = P.dev_scores(th, D)
    res[name] = dict(theta=dict(zip(P.FEATS, np.round(th, 3).tolist())), shortcut_at_n1=r(th[0] + th[1] * np.log(2)), dev_map=P.summ(sc),
                     d_all=boot_ci(sc, base, dev.question_id.values),
                     d_new_unseen=boot_ci(np.where(np.isin(S, ['new_pair', 'unseen_q']), sc, base), base, dev.question_id.values),
                     d_seen_pair=boot_ci(np.where(S == 'seen_pair', sc, base), base, dev.question_id.values))
    print(name, res[name]['shortcut_at_n1'], res[name]['theta'], res[name]['dev_map'], res[name]['d_all'], flush=True)
for lam in [.3, .6, 1.]:
    report(f'poe_errcond_lambda{lam}', w0, lam)
report('poe_errcond_lambda0.6_plus_group_x3', rescale([x['w0'] * (3 if (x['truth'] != 'NONE' and not x['seen_t']) else 1) for x in T]), .6)
json.dump(P.tojs(res), open(P.OUT / 'w2_probe3.json', 'w'), indent=1)
