"""Part 1: per-code learning curves and the (question, code) example-count -> accuracy relation."""
from lcommon import *

D = get_all(); y = D['y']; dev, fit, unl, classes = D['dev'], D['fit'], D['unl'], D['classes']
preds = D['preds']; q = dev.question_id.to_numpy(); err = y != 'NONE'
S = {k: score(y, v) for k, v in preds.items()}
H = {k: np.array([t in p for t, p in zip(y, v)]) for k, v in preds.items()}  # hit@3

n_fit_code = fit.root_code.value_counts()
n_cl_code = unl.c1.value_counts()
n_cl_code_any = Counter(c for cs in unl.c3 for c in cs)
pair_fit = Counter(zip(fit.question_id, fit.root_code))
pair_cl = Counter(zip(unl.question_id, unl.c1))
q_fit_rows = fit.question_id.value_counts(); q_unl_rows = unl.question_id.value_counts()
# Claude pool actually used by scale_100 / VL distill (data/v3/claude_unl_v4.jsonl soft labels)
used = {}
for line in open(ROOT / 'data/v3/claude_unl_v4.jsonl'):
    r = json.loads(line); used[r['trace_id']] = max(r['soft'], key=r['soft'].get)
uq = dict(zip(unl.trace_id, unl.question_id))
pair_used = Counter((uq.get(t), c) for t, c in used.items())

# ---- per-code table
rows = []
for c in classes:
    m = y == c
    rec = dict(code=c, family=fam(c), n_fit_gold=int(n_fit_code.get(c, 0)), n_train_gold=int((D['tr'].root_code == c).sum()),
               n_claude_top1=int(n_cl_code.get(c, 0)), n_claude_any3=int(n_cl_code_any.get(c, 0)),
               n_dev=int(m.sum()), n_test_q_with_code_in_train=None)
    for k in ['blend', 'blend_sk', 'claude', 'm32', 'm8', 'mvl', 'scale_0', 'scale_100']:
        rec[f'rec3_{k}'] = float(H[k][m].mean()) if m.any() else np.nan
        rec[f'map_{k}'] = float(S[k][m].mean()) if m.any() else np.nan
    # how often is c predicted top-1 (precision)
    for k in ['blend_sk', 'claude']:
        pm = preds[k][:, 0] == c
        rec[f'prec1_{k}'] = float((y[pm] == c).mean()) if pm.any() else np.nan
        rec[f'n_pred1_{k}'] = int(pm.sum())
    # gap mass: points lost by blend_sk vs claude on this class (summed, per dev row)
    rec['gap_points'] = float((S['claude'][m] - S['blend_sk'][m]).sum())
    rows.append(rec)
pc = pd.DataFrame(rows).drop(columns='n_test_q_with_code_in_train')
pc.to_csv(OUT / 'per_code.csv', index=False)

# spearman across codes with >=2 dev rows (rank correlation of recall vs log count)
def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])
sub = pc[(pc.n_dev >= 2) & (pc.code != 'NONE')]
code_curve = {'codes_with_dev>=2': int(len(sub))}
for k in ['blend_sk', 'claude', 'scale_0', 'scale_100']:
    code_curve[k] = {'spearman_rec3_vs_n_fit_gold': round(spearman(sub[f'rec3_{k}'], sub.n_fit_gold), 3),
                     'spearman_rec3_vs_n_claude_top1': round(spearman(sub[f'rec3_{k}'], sub.n_claude_top1), 3)}
# dev-row-weighted binned curve by n_fit_gold of the gold code (error rows)
bins = [(0, 4), (5, 9), (10, 19), (20, 39), (40, 999)]
nf = np.array([n_fit_code.get(c, 0) for c in y]); ncl = np.array([n_cl_code.get(c, 0) for c in y])
code_bins = {}
for lo, hi in bins:
    m = err & (nf >= lo) & (nf <= hi)
    code_bins[f'{lo}-{hi}'] = {'rows': int(m.sum()), **{k: round(float(S[k][m].mean()), 3) for k in ['blend_sk', 'claude', 'scale_0', 'scale_100']}}
cl_bins = {}
for lo, hi in [(0, 19), (20, 49), (50, 99), (100, 299), (300, 99999)]:
    m = err & (ncl >= lo) & (ncl <= hi)
    cl_bins[f'{lo}-{hi}'] = {'rows': int(m.sum()), **{k: round(float(S[k][m].mean()), 3) for k in ['blend_sk', 'claude', 'scale_0', 'scale_100']}}

# ---- row-level (question, code) pair counts
npf = np.array([pair_fit.get((a, c), 0) for a, c in zip(q, y)])
npc = np.array([pair_cl.get((a, c), 0) for a, c in zip(q, y)])
npu = np.array([pair_used.get((a, c), 0) for a, c in zip(q, y)])
qseen = np.array([a in q_fit_rows.index for a in q])
models = ['blend', 'blend_sk', 'claude', 'm32', 'm8', 'mvl', 'scale_0', 'scale_25', 'scale_100']
def tab(mask_fn, labels):
    out = {}
    for lab, m in zip(labels, mask_fn):
        m = m & err
        out[lab] = {'rows': int(m.sum()), **{k: round(float(S[k][m].mean()), 3) for k in models}}
    return out
pair_fit_tab = tab([~qseen, qseen & (npf == 0), npf == 1, npf == 2, npf >= 3], ['q_unseen', 'q_seen_pair0', 'pair1', 'pair2', 'pair3+'])
pair_cl_tab = tab([npc == 0, (npc >= 1) & (npc <= 2), (npc >= 3) & (npc <= 9), npc >= 10], ['cl0', 'cl1-2', 'cl3-9', 'cl10+'])
# cross tab: pair_fit==0 split by Claude pair count (does the Claude pool cover the new-code rows?)
hard = err & qseen & (npf == 0)
cross = tab([hard & (npc == 0), hard & (npc >= 1) & (npc <= 4), hard & (npc >= 5)], ['hard_cl0', 'hard_cl1-4', 'hard_cl5+'])
cross_used = tab([hard & (npu == 0), hard & (npu >= 1)], ['hard_used0', 'hard_used1+'])

# what did adding Claude labels buy, per pair regime (scale_100 - scale_0), with grouped CIs
gain = {}
for lab, m in [('pair_seen(npf>=1)', err & (npf >= 1)), ('hard(npf=0,q seen)', hard), ('hard & used_pair>=1', hard & (npu >= 1)),
               ('hard & used_pair=0', hard & (npu == 0)), ('q_unseen', err & ~qseen), ('NONE', ~err)]:
    d, ci = boot_ci(S['scale_100'][m], S['scale_0'][m], q[m])
    gain[lab] = {'rows': int(m.sum()), 'delta': round(d, 3), 'ci95': [round(x, 3) for x in ci]}

# ---- logistic fit: hit@3 ~ log1p(pair_fit) + log1p(code_fit) + log1p(pair_claude) + log1p(code_claude) + generic
def logit_fit(X, t, l2=1e-2):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p) + 1e-9
        g = X.T @ (t - p) - l2 * np.r_[0, w[1:]]
        Hm = X.T @ (X * W[:, None]) + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w = w + np.linalg.solve(Hm, g)
    return w
gen = np.array([fam(c) in GENERIC for c in y]).astype(float)
X = np.c_[np.log1p(npf), np.log1p(nf), np.log1p(npc), np.log1p(ncl), gen][err]
names = ['log1p_pair_fit', 'log1p_code_fit', 'log1p_pair_claude', 'log1p_code_claude', 'generic_code']
Xs = (X - X.mean(0)) / X.std(0)
logit = {}
rng = np.random.default_rng(0); qe = q[err]; uqs = np.unique(qe)
for k in ['blend_sk', 'claude', 'scale_100']:
    t = H[k][err].astype(float); w = logit_fit(Xs, t)
    bs = []
    for _ in range(500):
        pick = rng.choice(uqs, len(uqs)); idx = np.concatenate([np.flatnonzero(qe == u) for u in pick])
        bs.append(logit_fit(Xs[idx], t[idx]))
    bs = np.array(bs)
    logit[k] = {n: {'coef_per_sd': round(float(w[i + 1]), 3), 'ci95': [round(float(x), 3) for x in np.quantile(bs[:, i + 1], [.025, .975])]} for i, n in enumerate(names)}
# simple marginal curve: P(hit@3 | pair_fit = k) for blend_sk on error rows, and implied "examples to reach Claude"
res = dict(per_code_summary=code_curve, err_rows_by_code_fit_count=code_bins, err_rows_by_code_claude_count=cl_bins,
           err_rows_by_pair_fit_count=pair_fit_tab, err_rows_by_pair_claude_count=pair_cl_tab,
           hard_rows_by_pair_claude_count=cross, hard_rows_by_pair_in_used_training_pool=cross_used,
           scale100_minus_scale0_by_regime=gain, logistic_hit3_standardised=logit,
           claude_pool_label_mix={'rows': int(len(unl)), 'NONE_top1_share': round(float((unl.c1 == 'NONE').mean()), 3),
                                  'error_top1_rows': int((unl.c1 != 'NONE').sum()), 'REL-01_top1': int((unl.c1 == 'REL-01').sum()),
                                  'codes_with_<20_top1_examples': int(sum(n_cl_code.get(c, 0) < 20 for c in classes)),
                                  'codes_with_<5_top1_examples': int(sum(n_cl_code.get(c, 0) < 5 for c in classes))},
           gold_fit_label_mix={'codes_with_<10_fit_rows': int(sum(n_fit_code.get(c, 0) < 10 for c in classes)),
                               'median_fit_rows_per_code': float(np.median([n_fit_code.get(c, 0) for c in classes]))})
json.dump(res, open(OUT / 'p1_learning_curves.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
cols = ['code', 'n_fit_gold', 'n_claude_top1', 'n_dev', 'rec3_blend_sk', 'rec3_claude', 'prec1_blend_sk', 'prec1_claude', 'gap_points']
print(pc[pc.n_dev > 0].sort_values('gap_points', ascending=False)[cols].head(20).to_string())
print(pc[pc.n_dev > 0].sort_values('gap_points')[cols].head(8).to_string())
