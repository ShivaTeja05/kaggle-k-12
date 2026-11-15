"""T2. Is the Sinkhorn gain an artefact of the stratified dev fold?
make_folds stratifies root_code, so dev's label counts are (almost) exactly proportional to fit's: the target
marginal is known almost perfectly on dev. On test, target = full-train frequencies, and test's true marginal is
unknown. We perturb the target the way an iid train/test split would (N_eff = 1/(1/1500+1/3498) ~ 1050), harsher
(N_eff=301), and with a structurally different target (Claude's unlabelled-photo code frequencies; uniform codes).
Then we check label-free whether the test blend shows the same per-class bias pattern that Sinkhorn corrects on dev."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes); mf = qfolds(qid)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
cnt_fit = np.bincount(fit.root_code.map(ci), minlength=K).astype(float)
pi = (cnt_fit + .5)/(cnt_fit + .5).sum()
cnt_dev = np.bincount(dev.root_code.map(ci), minlength=K).astype(float)
res = {'dev_vs_fit_marginal_L1': float(np.abs(cnt_dev/cnt_dev.sum() - cnt_fit/cnt_fit.sum()).sum()),
       'iid_expected_L1_n301_vs_1199': None}
rng = np.random.default_rng(0)
p0 = cnt_fit/cnt_fit.sum()
sim = [np.abs(rng.multinomial(301, p0)/301 - rng.multinomial(1199, p0)/1199).sum() for _ in range(200)]
res['iid_expected_L1_n301_vs_1199'] = float(np.mean(sim))

H = np.load(OUT/'t1_heads.npz', allow_pickle=True)
assert list(H['ids']) == ids
Fh = geo([(B, .5), (H['ids_stripped'], .5)])
members = {'blend': B, 'blend_x_claude_head_w0.5': Fh}


def gains(target, lam=.5):
    return {n: row_scores(y, topk(sinkhorn(P, target, lam), classes)) - row_scores(y, topk(P, classes)) for n, P in members.items()}


res['exact_target'] = {n: float(g.mean()) for n, g in gains(pi).items()}
for neff in (1050, 301):
    acc = {n: [] for n in members}
    for s in range(60):
        t = np.random.default_rng(100+s).multinomial(neff, p0).astype(float) + .5; t /= t.sum()
        for n, g in gains(t).items(): acc[n].append(g.mean())
    res[f'iid_noise_neff{neff}'] = {n: dict(mean=float(np.mean(v)), p05=float(np.quantile(v, .05)), p95=float(np.quantile(v, .95)),
                                         frac_positive=float(np.mean(np.array(v) > 0))) for n, v in acc.items()}
cl = np.zeros(K)
for r in read_jsonl(ROOT/'runs/claude/unl_all_v4.jsonl'):
    c = (r.get('answer') or {}).get('first_code')
    if c in ci: cl[ci[c]] += 1
res['target_claude_unl_freq'] = {n: float(g.mean()) for n, g in gains((cl+.5)/(cl+.5).sum()).items()}
u = np.where(cnt_fit > 0, 1., .01); u[ci['NONE']] = 0; u = u/u.sum()*(1-p0[ci['NONE']]); u[ci['NONE']] = p0[ci['NONE']]
res['target_uniform_codes'] = {n: float(g.mean()) for n, g in gains(u).items()}

# Label-free transfer check: Sinkhorn bias vector on dev (fit-only blend) vs on test (full-train blend).
def sink_bias(P, target, iters=30):
    b = np.zeros(K)
    for _ in range(iters):
        Q = normalize(P*np.exp(b)); b += np.log(target/Q.mean(0))
    return b
te = read_csv(ROOT/'data/raw/test.csv'); tid = te.trace_id.tolist()
T = geo([(aligned_bundle(ROOT/'runs/pod/text32-full/test.npz', tid, classes), .35),
         (aligned_bundle(ROOT/'runs/pod/text-full/test.npz', tid, classes), .30),
         (aligned_bundle(ROOT/'runs/pod/full-distill/test.npz', tid, classes), .35)])
cnt_all = np.bincount(tr.root_code.map(ci), minlength=K).astype(float); pi_all = (cnt_all+.5)/(cnt_all+.5).sum()
bd, bt = sink_bias(B, pi), sink_bias(T, pi_all)
sup = cnt_all >= 10
res['bias_corr_dev_vs_test_classes_ge10'] = float(np.corrcoef(bd[sup], bt[sup])[0, 1])
res['bias_corr_dev_vs_test_spearman_ge10'] = float(pd.Series(bd[sup]).corr(pd.Series(bt[sup]), method='spearman'))
res['n_classes_ge10'] = int(sup.sum())
t1 = np.bincount(T.argmax(1), minlength=K); d1 = np.bincount(B.argmax(1), minlength=K)
res['top1_share_vs_target_L1'] = dict(dev=float(np.abs(d1/d1.sum() - pi).sum()), test=float(np.abs(t1/t1.sum() - pi_all).sum()))
Ts = sinkhorn(T, pi_all, .5)
res['test_top1_changed_by_sinkhorn'] = float((Ts.argmax(1) != T.argmax(1)).mean())
res['dev_top1_changed_by_sinkhorn'] = float((sinkhorn(B, pi, .5).argmax(1) != B.argmax(1)).mean())
dump(res, 't2_sinkhorn.json')
