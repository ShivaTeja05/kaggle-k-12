from common import *
from k12.baseline import HierarchicalPrior
import json
from scipy.stats import entropy
tr, tax, classes, folds, dev, fit = load()
te = read_csv(ROOT/'data/raw/test.csv')
out = {}
qc = tr.groupby('question_id').size()
out['n_questions_train'] = int(len(qc)); out['rows_per_q'] = qc.describe().round(2).to_dict()
out['test_q_in_train_share'] = float(te.question_id.isin(tr.question_id).mean())
out['test_rows_per_q_mean'] = float(te.groupby('question_id').size().mean())
# per-question code distribution (all train)
E = []; ncodes = []; maj = []
for q, g in tr.groupby('question_id'):
    vc = g.root_code.value_counts()
    E.append(entropy(vc.values, base=2)); ncodes.append(len(vc)); maj.append(vc.iloc[0]/len(g))
    E_err = None
err = tr[tr.root_code != 'NONE']
Ee = [entropy(g.root_code.value_counts().values, base=2) for q, g in err.groupby('question_id') if len(g) >= 2]
out['per_q_entropy_bits_mean'] = float(np.mean(E)); out['per_q_distinct_codes_mean'] = float(np.mean(ncodes))
out['per_q_majority_share_mean'] = float(np.mean(maj))
out['per_q_error_only_entropy_bits_mean(q with>=2 err rows)'] = float(np.mean(Ee))
out['global_code_entropy_bits'] = float(entropy(tr.root_code.value_counts().values, base=2))
out['global_error_code_entropy_bits'] = float(entropy(err.root_code.value_counts().values, base=2))
# conditional entropy H(code|question) weighted by rows
H = sum(len(g)/len(tr) * entropy(g.root_code.value_counts().values, base=2) for q, g in tr.groupby('question_id'))
out['H(code|question)_in_sample_bits'] = float(H)
# dev truth among codes seen for that question in fit rows
fq = fit.groupby('question_id').root_code.apply(set).to_dict()
fqe = fit[fit.root_code != 'NONE'].groupby('question_id').root_code.apply(set).to_dict()
de = dev[dev.root_code != 'NONE']
out['dev_q_seen_in_fit'] = float(dev.question_id.isin(fit.question_id).mean())
out['dev_truth_in_fit_codes_for_q (all rows)'] = float(np.mean([r.root_code in fq.get(r.question_id, set()) for r in dev.itertuples()]))
out['dev_error_truth_in_fit_error_codes_for_q'] = float(np.mean([r.root_code in fqe.get(r.question_id, set()) for r in de.itertuples()]))
out['dev_error_rows_q_has_fit_error_rows'] = float(np.mean([r.question_id in fqe for r in de.itertuples()]))
out['fit_error_codes_per_q_mean(for dev err rows)'] = float(np.mean([len(fqe.get(r.question_id, ())) for r in de.itertuples()]))
# topic -> section
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
tr2 = tr.assign(sec=tr.root_code.map(sec)); e2 = tr2[tr2.root_code != 'NONE']
out['H(section)_err_bits'] = float(entropy(e2.sec.value_counts().values, base=2))
out['H(section|topic)_err_bits'] = float(sum(len(g)/len(e2)*entropy(g.sec.value_counts().values, base=2) for _, g in e2.groupby('topic_name')))
out['H(code|topic)_err_bits'] = float(sum(len(g)/len(e2)*entropy(g.root_code.value_counts().values, base=2) for _, g in e2.groupby('topic_name')))
out['H(code|question)_err_bits'] = float(sum(len(g)/len(e2)*entropy(g.root_code.value_counts().values, base=2) for _, g in e2.groupby('question_id')))
# prior alone and prior x blend on dev
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ref = row_scores(y, topk(P, classes))
prior = HierarchicalPrior(classes).fit(fit).predict_proba(dev)
out['prior_only_dev_map3'] = map3(y, prior, classes)
# is the blend's top-3 restricted to fit-seen codes? where does the blend miss
top = topk(P, classes)
miss = de.index[(row_scores(de.root_code.to_numpy(), top[de.index]) == 0)]
out['blend_err_rows_missed_top3'] = int(len(miss))
out['missed_truth_in_fit_codes_for_q'] = float(np.mean([dev.root_code[i] in fqe.get(dev.question_id[i], set()) for i in miss]))
# cross-fit prior weight (geometric) by question groups
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
def mix(w, a=2.0, b=8.0, mask=None):
    pr = HierarchicalPrior(classes, a, b).fit(fit).predict_proba(dev)
    return normalize(P * np.clip(pr, 1e-6, None) ** w)
grid = [(w, a) for w in (0, .1, .2, .3, .5) for a in (0.5, 2., 8.)]
cache = {c: mix(*c) for c in grid}
R = np.empty_like(P); choice = []
for f in range(5):
    tr_m = mf != f
    sc = [row_scores(y[tr_m], topk(cache[c][tr_m], classes)).mean() for c in grid]
    c = grid[int(np.argmax(sc))]; choice.append(c); R[~tr_m] = cache[c][~tr_m]
pts = row_scores(y, topk(R, classes))
d, ci = boot_ci(pts, ref, dev.question_id.to_numpy())
out['prior_geo_crossfit'] = dict(map3=float(pts.mean()), delta=d, ci95=ci, choices=[list(c) for c in choice],
                                 in_sample=[(list(c), float(row_scores(y, topk(cache[c], classes)).mean())) for c in grid])
print(json.dumps(out, indent=1))
json.dump(out, open('question_structure.json', 'w'), indent=1)
