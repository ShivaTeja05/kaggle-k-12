"""(1) How much does each model follow the question prior P(code|question)?

Question-only model = HierarchicalPrior (question counts + topic/global backoff) on fit rows, and a variant that
also counts Claude's first codes on unlabelled photos of the same question (the data the members trained on).
Per slice (none / seen_pair / new_pair / unseen_q): MAP@3, top-1 == prior mode, KL(model||prior), JS,
Pearson(log p_model, log p_prior) over codes, share of top-3 slots that are fit codes of the question.
Truth and Claude are the reference: a model that agrees with the prior more than the truth does is over-relying on it.
Also a label-free test check: agreement of the full-train test bundles with the full-train prior, vs the
leave-one-out rate at which the truth equals the prior mode in train.
"""
import json
import numpy as np
import pandas as pd
from k12.baseline import HierarchicalPrior
from k12.core import read_csv, topk, row_scores, normalize
from sc_common import load, dev_preds, test_preds, claude_dev, claude_unl, slices, OUT, ROOT, r, boot_ci

tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
y = dev.root_code.to_numpy()
S, qc = slices(dev, fit)
P = dev_preds(dev, classes)
PC, ccodes = claude_dev(dev, classes)
P['claude'] = PC
cl, unl = claude_unl()

prior_fit = HierarchicalPrior(classes).fit(fit).predict_proba(dev)
# fit + Claude first codes on the same questions (unlabelled photos), dev photos never included
cl_dev_free = cl[~cl.trace_id.isin(set(dev.trace_id))]
prior_fc = HierarchicalPrior(classes).fit(pd.concat([fit[['question_id', 'topic_name', 'root_code']],
                                                     cl_dev_free[['question_id', 'topic_name', 'root_code']]])).predict_proba(dev)
# Count-only question mode (fit rows), defined only where the question has fit rows
qcount = {q: g.root_code.value_counts() for q, g in fit.groupby('question_id')}


def kl(p, q):
    p = np.clip(p, 1e-9, 1); q = np.clip(q, 1e-9, 1)
    return (p * np.log(p / q)).sum(1)


def js(p, q):
    m = (p + q) / 2
    return .5 * kl(p, m) + .5 * kl(q, m)


def stats(Pm, prior, mask):
    top = topk(Pm, classes)
    pm = topk(prior, classes)[:, 0]
    seenq = np.array([q in qc for q in dev.question_id])
    m2 = mask & seenq
    lp, lq = np.log(np.clip(Pm, 1e-9, 1)), np.log(np.clip(prior, 1e-9, 1))
    pear = [np.corrcoef(lp[i], lq[i])[0, 1] for i in np.where(m2)[0]]
    slot_seen = [np.mean([c in qc[q] for c in top[i]]) for i, q in enumerate(dev.question_id) if m2[i]]
    top1_seen = [top[i, 0] in qc[q] for i, q in enumerate(dev.question_id) if m2[i]]
    return dict(n=int(mask.sum()), map3=r(row_scores(y[mask], top[mask]).mean()),
                top1_eq_prior_mode=r((top[:, 0] == pm)[m2].mean()) if m2.any() else None,
                top1_is_fit_code_of_q=r(np.mean(top1_seen)) if m2.any() else None,
                top3_slots_fit_codes_of_q=r(np.mean(slot_seen)) if m2.any() else None,
                kl_model_prior=r(kl(Pm, prior)[mask].mean()), js=r(js(Pm, prior)[mask].mean()),
                pearson_logp=r(np.mean(pear)) if pear else None)


res = {'slice_counts': pd.Series(S).value_counts().to_dict(), 'prior_map3': {}, 'models': {}}
for pname, prior in (('fit', prior_fit), ('fit+claude', prior_fc)):
    res['prior_map3'][pname] = {s: r(row_scores(y[S == s], topk(prior, classes)[S == s]).mean()) for s in
                                ['none', 'seen_pair', 'new_pair', 'unseen_q']} | {'all': r(row_scores(y, topk(prior, classes)).mean())}
for name in ['m32', 'm8', 'vl', 'blend', 'claude']:
    res['models'][name] = {s: stats(P[name], prior_fit, S == s) for s in ['none', 'seen_pair', 'new_pair', 'unseen_q']}
    res['models'][name]['error_rows_seen_q'] = stats(P[name], prior_fit, np.isin(S, ['seen_pair', 'new_pair']))
    res['models'][name]['all'] = stats(P[name], prior_fit, np.ones(len(S), bool))
# Truth reference: how often the truth is the prior mode / a fit code of the question
pm = topk(prior_fit, classes)[:, 0]
seenq = np.array([q in qc for q in dev.question_id])
res['truth_reference'] = {s: dict(truth_eq_prior_mode=r((y == pm)[(S == s) & seenq].mean()) if ((S == s) & seenq).any() else None)
                          for s in ['none', 'seen_pair', 'new_pair', 'all_seen_q']} | {
    'all_seen_q': dict(truth_eq_prior_mode=r((y == pm)[seenq].mean()))}
em = np.isin(S, ['seen_pair', 'new_pair'])
res['truth_reference']['error_rows_seen_q'] = dict(truth_eq_prior_mode=r((y == pm)[em].mean()),
                                                   truth_is_fit_code_of_q=r(np.mean([t in qc[q] for t, q in zip(y[em], dev.question_id[em])])))

# Prior-following "excess": model top-1 == prior mode minus truth == prior mode, on seen-question error rows
res['excess_prior_agreement_error_rows_seen_q'] = {
    n: r(res['models'][n]['error_rows_seen_q']['top1_eq_prior_mode'] - res['truth_reference']['error_rows_seen_q']['truth_eq_prior_mode'])
    for n in res['models']}

# When the model's top-1 is the prior mode on new-pair rows, it is wrong by construction; count the points lost.
for n in ['m32', 'm8', 'vl', 'blend', 'claude']:
    top = topk(P[n], classes)
    m = S == 'new_pair'
    hit_mode = top[m, 0] == pm[m]
    res['models'][n]['new_pair']['rows_top1_is_prior_mode'] = int(hit_mode.sum())
    res['models'][n]['new_pair']['map3_when_top1_is_prior_mode'] = r(row_scores(y[m][hit_mode], top[m][hit_mode]).mean()) if hit_mode.any() else None
    res['models'][n]['new_pair']['map3_otherwise'] = r(row_scores(y[m][~hit_mode], top[m][~hit_mode]).mean()) if (~hit_mode).any() else None

# Paired bootstrap: blend vs claude excess agreement on error rows of seen questions
d_bl = (topk(P['blend'], classes)[:, 0] == pm).astype(float)
d_cl = (topk(PC, classes)[:, 0] == pm).astype(float)
res['blend_minus_claude_prior_mode_rate_error_rows_seen_q'] = dict(zip(['delta', 'ci95'], boot_ci(d_bl[em], d_cl[em], dev.question_id[em].to_numpy())))

# Label-free test check: full-train test bundles vs full-train prior
test = read_csv(ROOT / 'data/raw/test.csv')
PT = test_preds(test, classes)
prior_tr = HierarchicalPrior(classes).fit(tr).predict_proba(test)
pmt = topk(prior_tr, classes)[:, 0]
qtr = tr.groupby('question_id').root_code.apply(set).to_dict()
seen_t = np.array([q in qtr for q in test.question_id])
res['test_label_free'] = {'rows': len(test), 'rows_seen_q': int(seen_t.sum())}
for n in ['m32', 'm8', 'vl', 'blend']:
    top = topk(PT[n], classes)
    res['test_label_free'][n] = dict(top1_eq_prior_mode=r((top[:, 0] == pmt)[seen_t].mean()),
                                     top1_is_train_code_of_q=r(np.mean([top[i, 0] in qtr[q] for i, q in enumerate(test.question_id) if seen_t[i]])),
                                     top3_slots_train_codes_of_q=r(np.mean([np.mean([c in qtr[q] for c in top[i]]) for i, q in enumerate(test.question_id) if seen_t[i]])))
    # same quantity on dev for direct comparison (dev model vs fit prior)
    topd = topk(P[n], classes)
    res['test_label_free'][n]['dev_counterpart_top1_eq_prior_mode'] = r((topd[:, 0] == pm)[seenq].mean())
# Expected truth rate on test, estimated by leave-one-out in train: truth == mode of other rows of the question
loo_mode, loo_seen, ns = [], [], []
for q, g in tr.groupby('question_id'):
    if len(g) < 2:
        continue
    for i in range(len(g)):
        others = g.root_code.drop(g.index[i])
        vc = others.value_counts()
        loo_mode.append(g.root_code.iloc[i] == vc.index[0]); loo_seen.append(g.root_code.iloc[i] in set(others)); ns.append(len(others))
res['train_loo_reference'] = dict(rows=len(loo_mode), truth_eq_mode_of_other_rows=r(np.mean(loo_mode)),
                                  truth_in_codes_of_other_rows=r(np.mean(loo_seen)), mean_other_rows=r(np.mean(ns), 2),
                                  note='dev-model counterpart is the fit prior; test models were trained on all 1,500 rows, so test rows have ~3.66 train rows per question vs 2.78 fit rows per dev question')
# Expected test new-pair share, estimated by LOO weighting test questions by their train row counts
ntr = tr.question_id.value_counts()
test_n = test.question_id.map(ntr).fillna(0)
res['test_train_rows_per_q'] = dict(mean=r(test_n.mean(), 2), share_unseen_q=r((test_n == 0).mean()))
json.dump(res, open(OUT / 's1_prior_agreement.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
