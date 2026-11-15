"""W5 approach C CPU test: synthetic mal-rule rows (inject_malrules.py / gen_pool.py).
(a) realism: can a char TF-IDF LR tell synthetic rows from real gold error transcripts? (question-grouped CV AUC)
(b) utility: add them to the W5 proxy (A+B recipe = all_labelshift_pairbal_gen + gold-uncovered x3) and score dev.
Dev = fold 0 (301 rows, machine transcripts).  Synthetic donors are fit-fold gold NONE rows or Claude-NONE pool photos,
never dev traces.  CIs: question-grouped paired bootstrap (boot_ci).
Run: cd ROOT && PYTHONPATH=src:scripts:runs/arch/label-structure:runs/deep/test-W5 runs/deep/approach-W5/venv/bin/python runs/deep/test-W5/c_test.py
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.argv = sys.argv[:1]
src = open('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/approach-W5/w5_proxy.py').read().split('res = dict(diagnostics')[0]
exec(src)
from scipy.sparse import vstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
OUT2 = ROOT / 'runs/deep/test-W5'
R = lambda p: [json.loads(l) for l in open(p)]
syn_g = pd.DataFrame(R(OUT2 / 'synthetic.jsonl')); syn_p = pd.DataFrame(R(OUT2 / 'synthetic_pool.jsonl'))
syn_g['src'] = 'gold_donor'; syn_p['src'] = 'pool_donor'
syn = pd.concat([syn_g, syn_p], ignore_index=True)
assert not set(syn.donor) & set(dev.trace_id), 'dev trace used as donor'
INJ = sorted(set(syn.root_code))
Xsyn_t = [text(a, b, c) for a, b, c in zip(syn.topic_name, syn.question_text, syn.transcription_steps)]
Xsyn = vec.transform(Xsyn_t); ysyn = syn.root_code.map(ci).to_numpy()
res = dict(n_syn=len(syn), n_syn_gold_donor=len(syn_g), n_syn_pool_donor=len(syn_p), syn_codes=syn.root_code.value_counts().to_dict(),
           syn_questions=int(syn.question_id.nunique()))

# ---------- coverage: which dev / test rows can these rows even reach? ----------
te = pd.read_csv(ROOT / 'data/raw/test.csv')
unlq = set(pd.read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv', usecols=['question_id']).question_id)
synq = set(syn.question_id); synpair = set(zip(syn.question_id, syn.root_code))
injdev = (y != 'NONE') & np.isin(y, INJ)
npu = np.isin(sl, ['new_pair', 'unseen_q'])
pair_hit = np.array([(q, t) in synpair for q, t in zip(dev.question_id, y)])
res['coverage'] = dict(
    dev_err_rows_injectable_code=int(injdev.sum()), dev_newpair_unseen_rows=int(npu.sum()),
    dev_newpair_unseen_injectable=int((injdev & npu).sum()),
    dev_newpair_unseen_with_exact_synthetic_pair=int((npu & pair_hit).sum()),
    dev_err_rows_with_exact_synthetic_pair=int(((y != 'NONE') & pair_hit).sum()),
    test_rows_on_q_with_synthetic=int(te.question_id.isin(synq).sum()), test_rows=len(te),
    uncovered_test_q=int(te.loc[~te.question_id.isin(unlq), 'question_id'].nunique()),
    uncovered_test_rows_on_q_with_synthetic=int((~te.question_id.isin(unlq) & te.question_id.isin(synq)).sum()),
    uncovered_test_q_with_synthetic=int(te.loc[~te.question_id.isin(unlq) & te.question_id.isin(synq), 'question_id'].nunique()))
refS_pts = refS
res['coverage']['blendS_on_dev_newpair_unseen_injectable'] = round(float(refS_pts[injdev & npu].mean()), 4)
res['coverage']['max_overall_gain_if_all_newpair_unseen_injectable_rows_became_perfect'] = round(float((1 - refS_pts[injdev & npu]).sum() / len(y)), 4)
res['coverage']['max_overall_gain_if_exact_pair_rows_became_perfect'] = round(float((1 - refS_pts[(y != 'NONE') & pair_hit]).sum() / len(y)), 4)
print(json.dumps(res, indent=1, default=str), flush=True)

# ---------- (a) realism ----------
gold_err = fit[fit.root_code != 'NONE']
gold_none = fit[fit.root_code == 'NONE']
noteless = lambda t: re.sub(r'\[[^\]]*\]', ' ', str(t))
def auc_cv(pos_txt, neg_txt, groups, strip_notes=False):
    T = [noteless(t) if strip_notes else t for t in list(pos_txt) + list(neg_txt)]
    T = [noL(t).lower() for t in T]
    Y = np.r_[np.ones(len(pos_txt)), np.zeros(len(neg_txt))]
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    S = np.zeros(len(Y))
    for trn, tst in GroupKFold(5).split(T, Y, groups):
        vv = v.fit(np.array(T, dtype=object)[trn]); Xa = vv.transform(np.array(T, dtype=object)[trn]); Xb = vv.transform(np.array(T, dtype=object)[tst])
        m = LogisticRegression(C=4.0, max_iter=500, class_weight='balanced').fit(Xa, Y[trn]); S[tst] = m.predict_proba(Xb)[:, 1]
    return round(float(roc_auc_score(Y, S)), 3)
same_q = gold_err[gold_err.question_id.isin(syn_g.question_id)]
real_inj = gold_err[gold_err.root_code.isin(INJ)]
realism = {}
for strip in [False, True]:
    k = 'notes_stripped' if strip else 'raw'
    realism[k] = dict(
        syn_gold_vs_real_err_same_q=auc_cv(syn_g.transcription_steps, same_q.transcription_steps, list(syn_g.question_id) + list(same_q.question_id), strip),
        syn_all_vs_real_err_same_codes=auc_cv(syn.transcription_steps, real_inj.transcription_steps, list(syn.question_id) + list(real_inj.question_id), strip),
        reference_real_none_vs_real_err=auc_cv(gold_none.transcription_steps, gold_err.transcription_steps, list(gold_none.question_id) + list(gold_err.question_id), strip),
        syn_gold_vs_its_donors=auc_cv(syn_g.transcription_steps, fit.set_index('trace_id').loc[syn_g.donor, 'transcription_steps'], list(syn_g.question_id) * 2, strip))
res['realism_auc'] = realism
res['realism_n'] = dict(syn_gold=len(syn_g), real_err_same_q=len(same_q), real_err_injectable_codes=len(real_inj))
print(json.dumps(realism, indent=1), flush=True)

# ---------- (b) utility in the proxy ----------
fit_uncov = ~fit.question_id.isin(pq.index).to_numpy()
g3 = np.where(fit_uncov, 3., 1.)
wA = scheme('all_labelshift_pairbal_gen')
def fitpred3(w_pool, w_syn, ysyn_used):
    m = w_pool > 0; ms = w_syn > 0
    X = vstack([Xfit, Xpool[m], Xsyn[ms]]); yy = np.concatenate([yfit, ypool[m], ysyn_used[ms]])
    sw = np.concatenate([g3, w_pool[m], w_syn[ms]])
    lr = LogisticRegression(C=4.0, max_iter=400, tol=1e-3); lr.fit(X, yy, sample_weight=sw)
    P = np.full((Xdev.shape[0], K), 1e-6); P[:, lr.classes_] += lr.predict_proba(Xdev)
    pi_tr = np.bincount(yy, weights=sw, minlength=K) + .5; pi_tr /= pi_tr.sum()
    return normalize(P), pi_tr
rng = np.random.default_rng(11)
placebo = ysyn.copy(); rng.shuffle(placebo)          # same rows, codes permuted among the injected codes
isg = (syn.src == 'gold_donor').to_numpy()
RUNS = {'AB_base': (0., ysyn, None), 'AB+syn_gold_w0.5': (.5, ysyn, isg), 'AB+syn_all_w0.3': (.3, ysyn, None),
        'AB+syn_all_w0.5': (.5, ysyn, None), 'AB+syn_all_w1.0': (1., ysyn, None), 'AB+placebo_all_w0.5': (.5, placebo, None)}
injidx = np.array([ci[c] for c in INJ])
def extra(P):
    lpt = np.log(P[np.arange(len(y)), [ci[c] for c in y]])
    top3 = np.argsort(-P, 1)[:, :3]
    fp = np.isin(top3, injidx).any(1)
    return lpt, fp
PTS = {}; res['utility'] = {}
for name, (ws, yl, mask) in RUNS.items():
    w_syn = np.full(len(syn), ws)
    if mask is not None: w_syn = np.where(mask, ws, 0.)
    P, pi_tr = fitpred3(wA, w_syn, yl)
    for tag, PP in [('raw', P), ('posthoc_prior', normalize(P * pi_fit / pi_tr))]:
        a, pa = report(PP); F, ch = crossfit_fuse(PP); FS = sinkhorn(F, pi_fit, .5); fs, pfs = report(FS)
        lpt, fp = extra(PP)
        PTS[(name, tag)] = (pa, pfs, lpt, fp)
        res['utility'][f'{name}/{tag}'] = dict(
            alone=a, alone_map3_inj_err=round(float(pa[injdev].mean()), 4), alone_map3_inj_newpair_unseen=round(float(pa[injdev & npu].mean()), 4),
            alone_map3_exact_pair_rows=round(float(pa[(y != 'NONE') & pair_hit].mean()), 4),
            mean_logp_true_inj_err=round(float(lpt[injdev].mean()), 4),
            none_rows_with_injectable_code_in_top3=round(float(fp[y == 'NONE'].mean()), 4),
            fusedS=fs, fuse_w=ch, fusedS_minus_blendS=boot_ci(pfs, refS, qid))
    print(name, json.dumps({k: v for k, v in res['utility'][f'{name}/raw'].items() if k != 'alone'}), flush=True)
for (name, tag), (pa, pfs, lpt, fp) in PTS.items():
    b = PTS[('AB_base', tag)]; r = res['utility'][f'{name}/{tag}']
    r['vs_base'] = dict(
        inj_err_map3=boot_ci(pa[injdev], b[0][injdev], qid[injdev]),
        inj_newpair_unseen_map3=boot_ci(pa[injdev & npu], b[0][injdev & npu], qid[injdev & npu]),
        exact_pair_rows_map3=boot_ci(pa[(y != 'NONE') & pair_hit], b[0][(y != 'NONE') & pair_hit], qid[(y != 'NONE') & pair_hit]),
        inj_err_logp_true=boot_ci(lpt[injdev], b[2][injdev], qid[injdev]),
        none_rows_map3=boot_ci(pa[y == 'NONE'], b[0][y == 'NONE'], qid[y == 'NONE']),
        alone_map3=boot_ci(pa, b[0], qid), fusedS_map3=boot_ci(pfs, b[1], qid))
json.dump(res, open(OUT2 / 'c_test.json', 'w'), indent=1, default=str)
for k, r in res['utility'].items():
    print(k, '| alone', r['alone']['map3'], 'injErr', r['alone_map3_inj_err'], 'injNP', r['alone_map3_inj_newpair_unseen'], 'pair', r['alone_map3_exact_pair_rows'],
          'none', r['alone']['map3_none'], 'fp', r['none_rows_with_injectable_code_in_top3'], '| fusedS', r['fusedS']['map3'], r['fuse_w'],
          '| vs base injErr', r['vs_base']['inj_err_map3'], 'logp', r['vs_base']['inj_err_logp_true'], 'fusedS', r['vs_base']['fusedS_map3'], 'vs blendS', r['fusedS_minus_blendS'])
