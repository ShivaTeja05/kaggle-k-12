"""Part 3: per-question structure and the share of TEST rows in the 'new code for a seen question' regime.

Three estimators, each validated on dev (fit -> dev) before being applied to test (full train -> test):
  A. subsampling curve P_new(m): within-train, hold one row out of a question, keep m other rows as reference.
  B. per-question Good-Turing: P_new(q) ~ (#codes seen once in q's reference rows) / (#reference rows), shrunk.
  C. Claude-pool proxy: share of Claude-labelled unlabelled photos of q whose top-1 is outside q's reference codes,
     calibrated on dev by the ratio observed/proxy.
'New' is defined on ERROR rows (gold != NONE) as in PLAN.md; unseen questions are reported separately.
"""
from lcommon import *

D = get_all(); tr, fit, dev, test, unl, folds = D['tr'], D['fit'], D['dev'], D['test'], D['unl'], D['folds']
rng = np.random.default_rng(3)
res = {}
# ---------- structure
g = tr.groupby('question_id')
res['train'] = {'questions': int(tr.question_id.nunique()), 'rows_per_q_quantiles': np.quantile(g.size(), [.1, .25, .5, .75, .9]).tolist(),
                'distinct_root_codes_per_q_dist': g.root_code.nunique().value_counts().sort_index().to_dict(),
                'distinct_error_codes_per_q_mean': round(float(tr[tr.root_code != 'NONE'].groupby('question_id').root_code.nunique().mean()), 3),
                'singleton_share_of_error_pairs': None}
pc = Counter(zip(tr.question_id, tr.root_code))
err_pairs = {k: v for k, v in pc.items() if k[1] != 'NONE'}
res['train']['singleton_share_of_error_pairs'] = round(sum(v == 1 for v in err_pairs.values()) / len(err_pairs), 3)
res['train']['error_rows_whose_pair_is_singleton'] = round(sum(v for v in err_pairs.values() if v == 1) / sum(err_pairs.values()), 3)
ntr = tr.question_id.value_counts(); nte = test.question_id.value_counts(); nun = unl.question_id.value_counts()
res['test'] = {'rows': int(len(test)), 'questions': int(test.question_id.nunique()),
               'test_rows_per_q_quantiles': np.quantile(nte, [.1, .25, .5, .75, .9]).tolist(),
               'share_test_rows_q_unseen_in_train': round(float((~test.question_id.isin(ntr.index)).mean()), 4),
               'share_test_rows_by_train_rows_for_q': {f'{a}-{b}': round(float(test.question_id.map(ntr).fillna(0).between(a, b).mean()), 3) for a, b in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99)]},
               'corr_test_rows_vs_train_rows_per_q': round(float(np.corrcoef(nte, nte.index.map(ntr).fillna(0))[0, 1]), 3),
               'test_q_with_claude_unl_photos_share_rows': round(float(test.question_id.isin(nun.index).mean()), 3),
               'claude_unl_photos_per_test_q_median': float(test.question_id.map(nun).fillna(0).median())}

# ---------- A. subsampling curve on train error rows (questions only within one 'reference' pool)
def curve(df, M=12, reps=30):
    """P(code of held-out error row not among codes of m other rows of same question) for m=1..M."""
    out = {}
    byq = {qq: gg.root_code.tolist() for qq, gg in df.groupby('question_id')}
    for m in range(1, M + 1):
        hits = []; w = []
        for qq, codes in byq.items():
            if len(codes) < m + 1:
                continue
            for _ in range(reps):
                perm = rng.permutation(len(codes)); held = codes[perm[0]]
                if held == 'NONE':
                    continue
                ref = {codes[j] for j in perm[1:m + 1]}
                hits.append(held not in ref)
        out[m] = (float(np.mean(hits)) if hits else np.nan, len(hits) // reps)
    return out
cur = curve(tr)
res['A_curve_P_new_given_m_ref_rows(train, all rows as reference)'] = {m: {'p_new': round(v[0], 3), 'eligible_q': v[1]} for m, v in cur.items()}
pnew = {m: v[0] for m, v in cur.items()}
def pnew_m(m):
    m = int(m)
    return 1.0 if m == 0 else pnew[min(m, max(k for k in pnew if not np.isnan(pnew[k])))]
# validation on dev: reference = fit rows of the question
de = dev[dev.root_code != 'NONE']; nfit = fit.question_id.value_counts()
fq = {qq: set(gg.root_code) for qq, gg in fit.groupby('question_id')}
obs_new = np.array([r.root_code not in fq.get(r.question_id, set()) for r in de.itertuples()])
obs_unseen_q = np.array([r.question_id not in fq for r in de.itertuples()])
predA = np.array([pnew_m(nfit.get(qq, 0)) for qq in de.question_id])
res['dev_observed'] = {'err_rows': int(len(de)), 'new_code_incl_unseen_q': round(float(obs_new.mean()), 3),
                       'new_code_seen_q_only(share of all err rows)': round(float((obs_new & ~obs_unseen_q).mean()), 3), 'unseen_q': round(float(obs_unseen_q.mean()), 3)}
res['A_dev_predicted_new_incl_unseen'] = round(float(predA.mean()), 3)
# 5-fold cross check of A (all folds, reference = other folds)
cf = []
for f in range(5):
    ref = tr[folds != f]; v = tr[(folds == f) & (tr.root_code != 'NONE')]
    rq = {qq: set(gg.root_code) for qq, gg in ref.groupby('question_id')}; rn = ref.question_id.value_counts()
    o = np.mean([r.root_code not in rq.get(r.question_id, set()) for r in v.itertuples()])
    p = np.mean([pnew_m(rn.get(qq, 0)) for qq in v.question_id])
    cf.append((round(float(o), 3), round(float(p), 3)))
res['A_5fold_observed_vs_predicted'] = cf
# Apply to test: reference = all train rows of q. Weight by test rows; error rows assumed to follow same question mix.
predA_test = np.array([pnew_m(ntr.get(qq, 0)) for qq in test.question_id])
res['A_test_predicted_new_incl_unseen(err rows)'] = round(float(predA_test.mean()), 3)

# ---------- B. per-question Good-Turing with shrinkage toward curve A
def gt(ref_codes_by_q, nref, qs, alpha=2.0):
    out = []
    for qq in qs:
        codes = ref_codes_by_q.get(qq)
        if not codes:
            out.append(1.0); continue
        e = [c for c in codes if c != 'NONE']; n = len(codes)
        n1 = sum(v == 1 for c, v in Counter(codes).items())
        prior = pnew_m(n)
        out.append((n1 + alpha * prior) / (n + alpha))
    return np.array(out)
fcodes = {qq: gg.root_code.tolist() for qq, gg in fit.groupby('question_id')}
tcodes = {qq: gg.root_code.tolist() for qq, gg in tr.groupby('question_id')}
res['B_dev_predicted'] = round(float(gt(fcodes, nfit, de.question_id).mean()), 3)
res['B_test_predicted'] = round(float(gt(tcodes, ntr, test.question_id).mean()), 3)

# ---------- C. Claude-pool proxy
ue = unl[unl.c1 != 'NONE']
def proxy(refsets, qs):
    out = []
    for qq in qs:
        u = ue[ue.question_id == qq].c1 if qq in uq_idx else None
        if u is None or len(u) == 0 or qq not in refsets:
            out.append(np.nan); continue
        out.append(float((~u.isin(refsets[qq])).mean()))
    return np.array(out)
uq_idx = set(ue.question_id)
pr_dev = proxy(fq, de.question_id); pr_test = proxy({qq: set(c) for qq, c in tcodes.items()}, test.question_id)
m = ~np.isnan(pr_dev) & ~obs_unseen_q
calib = float(obs_new[m].mean() / pr_dev[m].mean())
res['C_claude_proxy'] = {'dev_rows_with_proxy': int(m.sum()), 'dev_proxy_raw_mean': round(float(pr_dev[m].mean()), 3), 'dev_observed_on_same_rows': round(float(obs_new[m].mean()), 3),
                         'calibration_ratio': round(calib, 3),
                         'row_level_corr_proxy_vs_observed': round(float(np.corrcoef(pr_dev[m], obs_new[m])[0, 1]), 3),
                         'test_rows_with_proxy': int((~np.isnan(pr_test)).sum()), 'test_proxy_raw_mean': round(float(np.nanmean(pr_test)), 3),
                         'test_calibrated_seen_q': round(float(np.nanmean(pr_test) * calib), 3)}
# Also: Claude pool proxy restricted to codes, comparing fit-reference vs full-train reference on the SAME questions (shrink in proxy)
both_q = [qq for qq in set(test.question_id) if qq in fq and qq in uq_idx]
a = np.mean([float((~ue[ue.question_id == qq].c1.isin(fq[qq])).mean()) for qq in both_q])
b = np.mean([float((~ue[ue.question_id == qq].c1.isin(set(tcodes[qq]))).mean()) for qq in both_q])
res['C_same_questions_proxy_fit_ref_vs_train_ref'] = [round(a, 3), round(b, 3), len(both_q)]

# ---------- final test regime estimate (error rows), convert to all rows using NONE share ~0.33
none_share = float((tr.root_code == 'NONE').mean())
uq_share = float((~test.question_id.isin(ntr.index)).mean())
est = {'A_subsample': res['A_test_predicted_new_incl_unseen(err rows)'], 'B_good_turing': res['B_test_predicted'],
       'C_claude_proxy(seen q)+unseen q': round(res['C_claude_proxy']['test_calibrated_seen_q'] * (1 - uq_share) + uq_share, 3)}
res['test_estimate_err_rows_new_code_incl_unseen_q'] = est
res['test_estimate_all_rows'] = {k: round(v * (1 - none_share), 3) for k, v in est.items()}
res['dev_equivalent_all_rows'] = round(float(obs_new.mean()) * len(de) / len(dev), 3)
json.dump(res, open(OUT / 'p3_question_regime.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
