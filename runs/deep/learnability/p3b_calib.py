"""Part 3b: fold-calibration of the Good-Turing estimator and a question-bootstrap CI for the test regime estimate."""
exec(open('runs/deep/learnability/p3_question_regime.py').read().split('# ---------- final test regime')[0].replace('print(', 'pass  # (', 0))
out = {}
ratios = []
for f in range(5):
    ref = tr[folds != f]; v = tr[(folds == f) & (tr.root_code != 'NONE')]
    rq = {qq: set(gg.root_code) for qq, gg in ref.groupby('question_id')}
    rc = {qq: gg.root_code.tolist() for qq, gg in ref.groupby('question_id')}
    o = np.mean([r.root_code not in rq.get(r.question_id, set()) for r in v.itertuples()])
    p = gt(rc, None, v.question_id).mean(); ratios.append(o / p)
out['B_fold_obs_over_pred'] = [round(float(r), 3) for r in ratios]
out['B_test_bias_corrected'] = round(float(res['B_test_predicted'] * np.mean(ratios)), 3)
# bootstrap A over train questions (re-estimate curve) and test questions
qs = tr.question_id.unique(); tq = test.question_id.to_numpy(); tqs = np.unique(tq)
vals = []
for b in range(200):
    pick = rng.choice(qs, len(qs)); tb = pd.concat([tr[tr.question_id == x].assign(question_id=f'{x}#{i}') for i, x in enumerate(pick)])
    c = curve(tb, M=10, reps=3); pn = {m: v[0] for m, v in c.items() if not np.isnan(v[0])}
    tpick = rng.choice(tqs, len(tqs)); w = test.question_id.value_counts()
    num = sum(w[x] * (1.0 if ntr.get(x, 0) == 0 else pn[min(int(ntr.get(x, 0)), max(pn))]) for x in tpick)
    vals.append(num / sum(w[x] for x in tpick))
out['A_test_bootstrap_ci95'] = [round(float(x), 3) for x in np.quantile(vals, [.025, .975])]
out['test_seen_q_new_code_share_of_err_rows_A'] = round(res['A_test_predicted_new_incl_unseen(err rows)'] - float((~test.question_id.isin(ntr.index)).mean()), 3)
json.dump(out, open(OUT / 'p3b_calib.json', 'w'), indent=1); print(out)
