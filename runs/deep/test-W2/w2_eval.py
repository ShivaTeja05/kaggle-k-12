"""Evaluate the W2 proxy sweep on dev: standalone and fused with the blend (+ Sinkhorn lambda .5, the 0.737 reference),
hyper-parameters (l2, fusion weight) cross-fitted over the same 5 question-grouped folds as scripts/fuse_candidate.py,
question-grouped paired bootstrap CIs (common.boot_ci). Writes runs/deep/test-W2/w2_eval.json."""
import glob, json, pickle
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
REF = pts(sinkhorn(P0, pi, .5)); RAW = pts(P0)
qs = np.array(list(dict.fromkeys(qid))); np.random.default_rng(7).shuffle(qs)
gq = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([gq[q] for q in qid])
WS = [0, .1, .2, .3, .5, .75, 1.]
NU = np.isin(S, ['new_pair', 'unseen_q'])
SL = {'all': np.ones(n, bool), 'new_or_unseen': NU, 'seen_pair': S == 'seen_pair', 'none': S == 'none'}

fits = {}
for f in sorted(glob.glob(f'{OUT}/fits/*.npz')):
    info = json.load(open(f[:-4] + '.json'))
    Pd = np.load(f)['Pd']; PB = np.full((n, len(cl)), 1e-4); PB[np.arange(n)[:, None], C['dv']['cidx']] = Pd
    fits[info['tag']] = dict(PB=normalize(PB), info=info)
fam = {}
for tag in fits:
    mode, K, lam, l2 = tag.split('_'); fam.setdefault(f'{mode}_{K}_{lam}', []).append(tag)


def crossfit(cands):  # cands: {key: per-row points}; choose key on 4 folds, score the 5th
    out = np.zeros(n); ch = []
    for f in range(5):
        m = mf != f; k = max(cands, key=lambda k: cands[k][m].mean()); ch.append(k); out[~m] = cands[k][~m]
    return out, ch


def slice_tab(p):
    return {s: round(float(p[m].mean()), 4) for s, m in SL.items()} | {'unseen_q': round(float(p[S == 'unseen_q'].mean()), 4),
                                                                       'new_pair': round(float(p[S == 'new_pair'].mean()), 4)}


def deltas(a, b):
    out = {}
    for s, m in SL.items():
        d, c = boot_ci(np.where(m, a, b), b, qid)  # contribution of slice s to the overall delta (rows outside = 0)
        out[s] = dict(contrib=round(d, 4), ci95=[round(x, 4) for x in c], slice_mean=round(float((a - b)[m].mean()), 4))
    return out


res = dict(n=n, ref_blend_sinkhorn=slice_tab(REF), raw_blend=slice_tab(RAW), slice_sizes={s: int(m.sum()) for s, m in SL.items()})
R = {}
for fk, tags in sorted(fam.items()):
    sa = {t: pts(fits[t]['PB']) for t in tags}
    std, ch_s = crossfit(sa)
    fz = {(t, w): pts(sinkhorn(normalize(P0 * fits[t]['PB'] ** w), pi, .5)) for t in tags for w in WS}
    fused, ch_f = crossfit(fz)
    fzr = {(t, w): pts(normalize(P0 * fits[t]['PB'] ** w)) for t in tags for w in WS}  # no Sinkhorn (s5 comparability)
    fused_raw, _ = crossfit(fzr)
    rec3 = lambda P: float(np.mean([t in r_ for t, r_ in zip(y[NU], topk(P[NU], cl))]))
    R[fk] = dict(std=std, fused=fused, fused_raw=fused_raw, fz=fz)
    res[fk] = dict(per_l2={t: dict(standalone=slice_tab(sa[t]), shortcut_at_n1=round(fits[t]['info']['shortcut_at_n1'], 3),
                                   theta=fits[t]['info']['theta'], ess_err=round(fits[t]['info']['ess_err']),
                                   fused_insample_by_w={str(w): round(float(fz[(t, w)].mean()), 4) for w in WS}) for t in tags},
                   standalone_crossfit=slice_tab(std), standalone_l2_choices=ch_s,
                   fused_crossfit=slice_tab(fused), fused_choices=[list(c) for c in ch_f],
                   fused_vs_ref=deltas(fused, REF), fused_raw_vs_rawblend=dict(map=round(float(fused_raw.mean()), 4),
                                                                              d=boot_ci(fused_raw, RAW, qid)))
base = 'ce_K1_lam0'
for fk in R:
    if fk == base: continue
    res[fk]['standalone_vs_K1'] = deltas(R[fk]['std'], R[base]['std'])
    res[fk]['fused_vs_K1'] = deltas(R[fk]['fused'], R[base]['fused'])
    # matched comparison: identical (l2, w) setting for both arms, at each of K1's settings; mean over the 5 fold choices
    res[fk]['fused_vs_K1_matched_setting'] = {}
    for t1, w in sorted(set(map(tuple, res[base]['fused_choices']))):
        l2 = t1.split('_l2')[1]; tk = [x for x in fam[fk] if x.endswith('_l2' + l2)][0]
        a, b = R[fk]['fz'][(tk, w)], R[base]['fz'][(t1, w)]
        res[fk]['fused_vs_K1_matched_setting'][f'l2={l2},w={w}'] = {s: dict(zip(['contrib', 'ci95'], boot_ci(np.where(m, a, b), b, qid))) for s, m in SL.items()}
# kill rules
kill = {}
for K in ['2', '3', '5']:
    fk = f'ce_K{K}_lam0'
    if fk not in res: continue
    sv, fv = res[fk]['standalone_vs_K1'], res[fk]['fused_vs_K1']
    kill[f'A1_K{K}'] = dict(standalone_new_unseen_ci_at_or_below_0=sv['new_or_unseen']['ci95'][1] <= 0,
                           fused_seen_loss_exceeds_new_gain=-fv['seen_pair']['contrib'] > fv['new_or_unseen']['contrib'],
                           fused_overall_delta_vs_K1=fv['all']['contrib'], fused_overall_vs_ref=res[fk]['fused_vs_ref']['all'])
for lam in ['0.3', '0.5']:
    fk = f'poe_K1_lam{lam}'
    if fk not in res: continue
    kill[f'A3_lam{lam}'] = dict(none_drop_standalone=round(res[base]['standalone_crossfit']['none'] - res[fk]['standalone_crossfit']['none'], 4),
                               none_drop_gt_0p01=res[base]['standalone_crossfit']['none'] - res[fk]['standalone_crossfit']['none'] > .01,
                               fused_new_unseen_ci_at_or_below_0=res[fk]['fused_vs_ref']['new_or_unseen']['ci95'][1] <= 0,
                               fused_new_unseen_vs_K1=res[fk]['fused_vs_K1']['new_or_unseen'],
                               fused_raw_vs_rawblend=res[fk]['fused_raw_vs_rawblend'], old_s5_poe_fused=0.0055)
res['kill_rules'] = kill
json.dump(res, open(f'{OUT}/w2_eval.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('REF', res['ref_blend_sinkhorn'], 'RAW', res['raw_blend'])
for fk in sorted(R):
    print(fk, 'std', res[fk]['standalone_crossfit'], '\n   fused', res[fk]['fused_crossfit'], res[fk]['fused_choices'])
    print('   shortcut', {t.split('_l2')[1]: v['shortcut_at_n1'] for t, v in res[fk]['per_l2'].items()})
    if 'fused_vs_K1' in res[fk]:
        print('   std vs K1', {s: (v['contrib'], v['ci95']) for s, v in res[fk]['standalone_vs_K1'].items()})
        print('   fused vs K1', {s: (v['contrib'], v['ci95']) for s, v in res[fk]['fused_vs_K1'].items()})
    print('   fused vs REF', {s: (v['contrib'], v['ci95']) for s, v in res[fk]['fused_vs_ref'].items()})
print(json.dumps(kill, indent=1, default=str))
