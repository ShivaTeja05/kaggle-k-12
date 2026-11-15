"""Label-prior shift: train is capped (~57/code, NONE 506). Do the models' implied priors match it?"""
from common import *
import json
tr, tax, classes, folds, dev, fit = load()
te = read_csv(ROOT/'data/raw/test.csv')
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ref = row_scores(y, topk(P, classes))
ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit = pi_fit / pi_fit.sum()
pi_dev = np.bincount(dev.root_code.map(ci), minlength=len(classes)) / len(dev)
tids = te.trace_id.tolist()
lp = (.35*np.log(aligned_bundle(ROOT/'runs/pod/text32-full/test.npz', tids, classes)) + .30*np.log(aligned_bundle(ROOT/'runs/pod/text-full/test.npz', tids, classes))
      + .35*np.log(aligned_bundle(ROOT/'runs/pod/full-distill/test.npz', tids, classes)))
PT = normalize(np.exp(lp - lp.max(1, keepdims=True)))
out = {}
ni = ci['NONE']
out['NONE_share_train'] = float(pi_fit[ni]); out['dev_blend_mean_p_NONE'] = float(P[:, ni].mean())
out['dev_blend_top1_NONE_share'] = float((P.argmax(1) == ni).mean()); out['dev_true_NONE_share'] = float(pi_dev[ni])
out['test_blend_mean_p_NONE'] = float(PT[:, ni].mean()); out['test_blend_top1_NONE_share'] = float((PT.argmax(1) == ni).mean())
# per-code predicted top1 counts on test vs expected from train shares
exp_cnt = pi_fit * len(te); top1_cnt = np.bincount(PT.argmax(1), minlength=len(classes))
order = np.argsort(-(top1_cnt - exp_cnt))
out['test_over_predicted_top1'] = [(classes[i], int(top1_cnt[i]), round(float(exp_cnt[i]), 1)) for i in order[:10]]
out['test_under_predicted_top1'] = [(classes[i], int(top1_cnt[i]), round(float(exp_cnt[i]), 1)) for i in order[::-1][:15]]
out['test_codes_never_top1'] = int((top1_cnt == 0).sum()); out['test_codes_never_in_top3'] = int(len(set(range(len(classes))) - set(np.argsort(-PT, 1)[:, :3].ravel())))
# prior correction using test-mean (label-free) as model prior
pi_model_test = PT.mean(0)
pi_model_dev = P.mean(0)
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
res = {}
for name, pm in [('test_mean', pi_model_test), ('dev_mean', pi_model_dev)]:
    grid = [0, .1, .2, .3, .5, .75, 1.]
    cache = {l: normalize(P * (pi_fit / pm) ** l) for l in grid}
    insample = {l: float(row_scores(y, topk(cache[l], classes)).mean()) for l in grid}
    R = np.empty_like(P); ch = []
    for f in range(5):
        m = mf != f
        sc = [row_scores(y[m], topk(cache[l][m], classes)).mean() for l in grid]
        l = grid[int(np.argmax(sc))]; ch.append(l); R[~m] = cache[l][~m]
    pts = row_scores(y, topk(R, classes)); d, c95 = boot_ci(pts, ref, dev.question_id.to_numpy())
    res[name] = dict(in_sample=insample, crossfit=float(pts.mean()), delta=d, ci95=c95, choices=ch)
# Saerens EM on test to estimate test prior from the blend itself (label-free)
pi = pi_model_test.copy()
for _ in range(50):
    W = PT * (pi / pi_model_test); W = W / W.sum(1, keepdims=True); pi = W.mean(0)
out['EM_test_prior_NONE'] = float(pi[ni])
out['prior_correction'] = res
print(json.dumps(out, indent=1))
json.dump(out, open('prior_shift.json', 'w'), indent=1)
