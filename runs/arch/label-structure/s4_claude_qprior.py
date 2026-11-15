"""Question-conditioned code prior built from Claude labels on unlabelled photos (+ fit rows)."""
from common import *
import json
tr, tax, classes, folds, dev, fit = load()
unl = read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv')
te = read_csv(ROOT/'data/raw/test.csv')
rows = []
for line in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); a = r.get('answer') or {}
    rows.append((r['trace_id'], a.get('first_code'), a.get('second_code'), a.get('third_code')))
cl = pd.DataFrame(rows, columns=['trace_id', 'c1', 'c2', 'c3']).drop_duplicates('trace_id')
cl = cl.merge(unl[['trace_id', 'question_id']], on='trace_id', how='left')
out = {'claude_unl_rows': int(len(cl)), 'with_qid': int(cl.question_id.notna().sum()),
       'unlabelled_pool_rows': int(len(unl)), 'unl_questions': int(unl.question_id.nunique())}
cl = cl[cl.question_id.notna()]
out['claude_rows_per_dev_q_mean'] = float(dev.question_id.map(cl.groupby('question_id').size()).fillna(0).mean())
out['dev_q_with_claude_rows'] = float(dev.question_id.isin(cl.question_id).mean())
out['test_q_with_claude_rows'] = float(te.question_id.isin(cl.question_id).mean())
out['claude_first_NONE_share'] = float((cl.c1 == 'NONE').mean())
ci = {c: i for i, c in enumerate(classes)}
def table(frame_codes):
    T = {}
    for q, codes, w in frame_codes:
        if codes in ci:
            T.setdefault(q, np.zeros(len(classes)))[ci[codes]] += w
    return T
T1 = table(zip(cl.question_id, cl.c1, [1.]*len(cl)))
T123 = table(list(zip(cl.question_id, cl.c1, [.6]*len(cl))) + list(zip(cl.question_id, cl.c2, [.25]*len(cl))) + list(zip(cl.question_id, cl.c3, [.15]*len(cl))))
Tfit = table(zip(fit.question_id, fit.root_code, [1.]*len(fit)))
de = dev[dev.root_code != 'NONE']
def cov(T, frame, minw=0):
    return float(np.mean([(r.question_id in T) and T[r.question_id][ci[r.root_code]] > minw for r in frame.itertuples()]))
out['dev_err_truth_in_claude_first_codes_for_q'] = cov(T1, de)
out['dev_err_truth_in_claude_top3_codes_for_q'] = cov(T123, de)
out['dev_err_truth_in_fit_codes_for_q'] = cov(Tfit, de)
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
ref = row_scores(y, topk(P, classes))
top = topk(P, classes)
miss = np.flatnonzero((y != 'NONE') & (ref == 0))
out['blend_missed_err_rows'] = int(len(miss))
out['missed_truth_in_claude_first_codes_for_q'] = float(np.mean([dev.question_id[i] in T1 and T1[dev.question_id[i]][ci[y[i]]] > 0 for i in miss]))
# rank of truth inside per-question Claude distribution
def qprior(T, alpha):
    g = sum(T.values()); g = g / g.sum()
    out = []
    for q in dev.question_id:
        c = T.get(q, np.zeros(len(classes)))
        out.append((c + alpha * g) / (c.sum() + alpha))
    return np.asarray(out)
res = {}
for name, T in [('claude_first', T1), ('claude_top3', T123)]:
    for alpha in (1., 5.):
        pr = qprior(T, alpha)
        res[f'{name}_a{alpha}_alone'] = map3(y, pr, classes)
        res[f'{name}_a{alpha}_alone_err_rows'] = float(row_scores(y[y != 'NONE'], topk(pr[y != 'NONE'], classes)).mean())
out['prior_alone'] = res
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in dev.question_id])
grid = [(n, a, w) for n in ('claude_first', 'claude_top3') for a in (1., 5.) for w in (0, .05, .1, .2, .3, .5)]
cache = {}
for n, a, w in grid:
    pr = qprior(T1 if n == 'claude_first' else T123, a)
    cache[(n, a, w)] = normalize(P * np.clip(pr, 1e-4, None) ** w)
R = np.empty_like(P); ch = []
for f in range(5):
    m = mf != f
    sc = [row_scores(y[m], topk(cache[c][m], classes)).mean() for c in grid]
    c = grid[int(np.argmax(sc))]; ch.append(c); R[~m] = cache[c][~m]
pts = row_scores(y, topk(R, classes)); d, ci95 = boot_ci(pts, ref, dev.question_id.to_numpy())
out['crossfit_geo_mix'] = dict(map3=float(pts.mean()), delta=d, ci95=ci95, choices=[list(map(str, c)) for c in ch])
out['in_sample_grid'] = {str(c): float(row_scores(y, topk(cache[c], classes)).mean()) for c in grid}
print(json.dumps(out, indent=1))
json.dump(out, open('claude_qprior.json', 'w'), indent=1)
