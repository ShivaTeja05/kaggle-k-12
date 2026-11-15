"""W2 probe, part 2: variants that target the binary 'seen-for-question' cue directly (the probe-1 finding), each with
the error-row weight share rescaled to the current 0.664 so the NONE/error balance does not move.
 - group importance weights: error rows whose truth is unseen for their question (LOO) get weight K, then rescale.
 - prior-matched hard negatives: swap unseen distractors for unseen-by-candidate-set but seen-for-question codes so the
   train P(seen | distractor) reaches a target; no weights, no ESS loss.
 - error-conditional bias product (PoE): offset = lam * centred log p(code | question, error) on non-NONE letters only.
Same prior-only conditional-logit learner and dev evaluation as w2_probe.py."""
import json, sys
import numpy as np
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/approach-W2')
import w2_probe as P  # re-runs probe 1 (cheap) and exposes T, D, S, helpers
from sc_common import r
from common import boot_ci
T, D, S, dev, qcount, own, g, tot, L = P.T, P.D, P.S, P.dev, P.qcount, P.own, P.g, P.tot, P.L
rng = np.random.default_rng(1)
for x in T:
    cnt = dict(qcount[x['q']]); cnt[own.get(x['trace_id'] if 'trace_id' in x else None)] = cnt.get(None, 0)
for x, line in zip(T, open(P.ROOT / 'data/v5/cand_train.jsonl')):
    j = json.loads(line); x['tid'] = j['trace_id']
def loo(x):
    from collections import Counter
    c = Counter(qcount[x['q']]); c[own.get(x['tid'])] -= 1; return c
for x in T:
    c = loo(x); x['cnt'] = c; x['seen_t'] = x['truth'] != 'NONE' and c.get(x['truth'], 0) > 0
def err_lp(codes, cnt):
    ne = sum(v for k, v in cnt.items() if v > 0 and k != 'NONE'); ge = tot - g['NONE']
    lp = np.array([0. if c == 'NONE' else np.log((max(cnt.get(c, 0), 0) + 2 * (g[c] + 1) / (ge + 76)) / (ne + 2)) for c in codes])
    m = np.array([c != 'NONE' for c in codes]); lp[m] -= lp[m].mean(); return lp
for x in T: x['lp'] = err_lp(x['codes'], x['cnt'])
for x in D: x['lp'] = err_lp(x['codes'], qcount[x['q']])
def rescale(w):
    w = np.array(w, float); e = np.array([x['truth'] != 'NONE' for x in T]); w0 = np.array([x['w0'] for x in T])
    tgt = w0[e].sum() / w0.sum(); a = tgt * w[~e].sum() / ((1 - tgt) * w[e].sum()); w[e] *= a; return w
def stats(rows, w):
    e = [i for i, x in enumerate(rows) if x['truth'] != 'NONE']; w = np.asarray(w)[e]
    st = np.array([rows[i]['X'][rows[i]['codes'].index(rows[i]['truth']), 0] for i in e])
    sd = np.array([np.mean([rows[i]['X'][k, 0] for k, c in enumerate(rows[i]['codes']) if c not in (rows[i]['truth'], 'NONE')]) for i in e])
    pt, pd_ = np.average(st, weights=w), np.average(sd, weights=w)
    return dict(p_seen_truth=r(pt), p_seen_distr=r(pd_), LR_unseen=r((1 - pt) / (1 - pd_)), ess_err=int(w.sum() ** 2 / (w ** 2).sum()))
res = {}
base_th = P.fit_cl(T, [x['w0'] for x in T]); base = P.dev_scores(base_th, D)
def report(name, rows, w, off=0.):
    th = P.fit_cl(rows, w, offset=off); sc = P.dev_scores(th, D)
    res[name] = dict(theta=dict(zip(P.FEATS, np.round(th, 3).tolist())), shortcut_at_n1=r(th[0] + th[1] * np.log(2)),
                     dev_map=P.summ(sc), train_cue=stats(rows, w),
                     d_all=boot_ci(sc, base, dev.question_id.values),
                     d_new_unseen=boot_ci(np.where(np.isin(S, ['new_pair', 'unseen_q']), sc, base), base, dev.question_id.values),
                     d_seen_pair=boot_ci(np.where(S == 'seen_pair', sc, base), base, dev.question_id.values))
report('current', T, [x['w0'] for x in T])
for K in [2, 3, 5, 8]:
    report(f'group_newpair_x{K}_rescaled', T, rescale([x['w0'] * (K if (x['truth'] != 'NONE' and not x['seen_t']) else 1) for x in T]))
report('focal_g4_rescaled', T, rescale([x['w0'] * (1 if x['truth'] == 'NONE' else (1 - x['p']) ** 4) for x in T]))
# prior-matched hard negatives
ex_rate = np.mean([x['X'][k, 2] for x in T for k, c in enumerate(x['codes']) if c not in (x['truth'], 'NONE') and x['X'][k, 0] > 0])
avail = []
def swapped(target):
    out = []
    for x in T:
        y = dict(x); X = x['X'].copy(); codes = list(x['codes'])
        if x['truth'] != 'NONE':
            pool = [c for c, v in x['cnt'].items() if v > 0 and c not in codes and c != 'NONE']
            rng.shuffle(pool); avail.append(len(pool))
            ds = [k for k, c in enumerate(codes) if c not in (x['truth'], 'NONE')]
            need = int(round(target * len(ds))) - int(sum(X[k, 0] for k in ds))
            for k in [k for k in ds if X[k, 0] == 0][:max(need, 0)]:
                if not pool: break
                c = pool.pop(); codes[k] = c; n = x['cnt'][c]
                X[k] = [1., np.log1p(n), float(rng.random() < ex_rate), 0., np.log(g[c] / tot + 1e-6)]
        y['X'] = X; y['codes'] = codes; y['lp'] = err_lp(codes, x['cnt']); out.append(y)
    return out
for tgt in [.8, .9]:
    avail.clear(); rows = swapped(tgt)
    report(f'prior_matched_negatives_t{tgt}', rows, [x['w0'] for x in rows])
    res[f'prior_matched_negatives_t{tgt}']['median_seen_codes_available_to_swap'] = float(np.median(avail))
for lam in [.3, .6, 1.]:
    report(f'poe_errcond_lambda{lam}', T, [x['w0'] for x in T], off=lam)
res['dev_optimal_reference'] = dict(theta=P.res['dev_optimal']['theta'],
    shortcut_at_n1=r(P.res['dev_optimal']['theta']['seen'] + P.res['dev_optimal']['theta']['log_pair'] * np.log(2)),
    dev_map=P.res['dev_optimal']['dev_map'], dev_cue=stats([x | {'X': x['X']} for x in D if x['truth'] in x['codes']], np.ones(sum(x['truth'] in x['codes'] for x in D))))
json.dump(P.tojs(res), open(P.OUT / 'w2_probe2.json', 'w'), indent=1)
for k, v in res.items():
    print(k, v.get('shortcut_at_n1'), v.get('theta'), v['dev_map'], v.get('train_cue', v.get('dev_cue')), v.get('d_all'), v.get('d_new_unseen'), v.get('d_seen_pair'))
