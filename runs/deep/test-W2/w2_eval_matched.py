"""Adversarial follow-up: paired K vs K=1 comparisons at MATCHED l2 (no hyper-parameter selection in either arm),
standalone and fused with the blend + Sinkhorn(.5) at fixed a-priori weights w. Slice means with question-grouped
bootstrap CIs. Also dev 'same top-1 on same-question different-truth pairs' rate (the W2 observable).
Writes runs/deep/test-W2/w2_eval_matched.json."""
import glob, json, pickle, itertools
import numpy as np
from k12.core import normalize, topk, row_scores
from common import blend, boot_ci

OUT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W2'
C = pickle.load(open(f'{OUT}/cache.pkl', 'rb')); cl = C['classes']; ci = {c: i for i, c in enumerate(cl)}
y, S, qid, ids = C['dev_y'], C['slices'], C['dev_q'], C['dev_ids']; n = len(y)
P0 = blend(ids, cl)
pi = np.bincount([ci[c] for c in C['fit_codes']], minlength=len(cl)) + .5; pi /= pi.sum()


def sinkhorn(P, target, lam, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))


pts = lambda P: row_scores(y, topk(P, cl))
SL = {'all': np.ones(n, bool), 'new_or_unseen': np.isin(S, ['new_pair', 'unseen_q']), 'seen_pair': S == 'seen_pair', 'none': S == 'none'}
# same-question pairs of dev rows with different truths
pairs = [(i, j) for i, j in itertools.combinations(range(n), 2) if qid[i] == qid[j] and y[i] != y[j]]


def pb(tag):
    Pd = np.load(f'{OUT}/fits/{tag}.npz')['Pd']; P = np.full((n, len(cl)), 1e-4)
    P[np.arange(n)[:, None], C['dv']['cidx']] = Pd; return normalize(P)


def same_top1(P):
    t = P.argmax(1); return float(np.mean([t[i] == t[j] for i, j in pairs]))


def cmp(a, b):
    return {s: dict(slice_mean=round(float((a - b)[m].mean()), 4),
                    ci95_slice=[round(x * n / m.sum(), 4) for x in boot_ci(np.where(m, a, b), b, qid)[1]]) for s, m in SL.items()}


res = dict(dev_diff_truth_pairs=len(pairs), same_top1_blend_raw=same_top1(P0), same_top1_ref=same_top1(sinkhorn(P0, pi, .5)))
for l2 in ['0.0001', '0.0003', '0.001', '0.003']:
    base = pb(f'ce_K1_lam0_l2{l2}'); bs = pts(base)
    out = {'K1_same_top1': round(same_top1(base), 4)}
    for arm in ['ce_K2_lam0', 'ce_K3_lam0', 'ce_K5_lam0', 'poe_K1_lam0.3', 'poe_K1_lam0.5']:
        P = pb(f'{arm}_l2{l2}'); o = dict(standalone=cmp(pts(P), bs), same_top1=round(same_top1(P), 4))
        for w in [.1, .2, .3, .5]:
            o[f'fused_w{w}'] = cmp(pts(sinkhorn(normalize(P0 * P ** w), pi, .5)), pts(sinkhorn(normalize(P0 * base ** w), pi, .5)))
        out[arm] = o
    res[f'l2={l2}'] = out
json.dump(res, open(f'{OUT}/w2_eval_matched.json', 'w'), indent=1)
print('pairs', len(pairs), 'same_top1 raw blend', round(res['same_top1_blend_raw'], 3), 'ref', round(res['same_top1_ref'], 3))
for l2 in ['0.0001', '0.0003', '0.001', '0.003']:
    o = res[f'l2={l2}']; print(f'== l2={l2}  K1 same_top1 {o["K1_same_top1"]}')
    for arm in ['ce_K2_lam0', 'ce_K3_lam0', 'ce_K5_lam0', 'poe_K1_lam0.3', 'poe_K1_lam0.5']:
        a = o[arm]
        print(f'  {arm:14s} std', {s: (v['slice_mean'], v['ci95_slice']) for s, v in a['standalone'].items()}, 'same_top1', a['same_top1'])
        for w in [.1, .3]:
            print(f'  {"":14s} fused w{w}', {s: (v['slice_mean'], v['ci95_slice']) for s, v in a[f'fused_w{w}'].items()})
