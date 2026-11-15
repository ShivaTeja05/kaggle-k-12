"""T5. A deployable CPU 'student' that writes no reasoning of its own: for each dev row, retrieve the Claude-labelled
unlabelled photos of the SAME question whose Claude transcription is most similar to our machine transcript (tx_dev),
and (a) vote their codes, (b) average the reason-head output on their Claude reasoning.
Question: is the unseen-pair knowledge in Claude's traces reachable from the student's inputs (machine transcript +
question) at all? If retrieval from 26k Claude traces already reaches ~0.48 unseen-pair, the reasoning signal is
input-reachable; if it stays near the 8B's 0.44, the gain in reason_head_probe comes from Claude's view of the photo."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes); mf = qfolds(qid)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
pi = np.bincount(fit.root_code.map(ci), minlength=K) + .5; pi /= pi.sum()
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
pool = read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
U = []
for r in read_jsonl(ROOT/'runs/claude/unl_all_v4.jsonl'):
    a = r.get('answer') or {}
    if a.get('first_code') in ci and a.get('reasoning') and r['trace_id'] in pool.index:
        U.append(dict(trace_id=r['trace_id'], q=pool.at[r['trace_id'], 'question_id'], tx=a.get('transcription') or '',
                      codes=[a.get(k) for k in ('first_code', 'second_code', 'third_code')], text=claude_text(a)))
U = pd.DataFrame(U)
res = {'n_pool_labelled': len(U), 'overlap_with_train_ids': int(U.trace_id.isin(set(tr.trace_id)).sum()),
       'dev_rows_with_claude_neighbours': float(np.mean([q in set(U.q) for q in qid]))}
# coverage: does any same-question Claude label carry the gold code (first..third)?
bq = U.groupby('q').codes.apply(lambda s: set(c for cs in s for c in cs if c))
cov = np.array([yy in bq.get(q, set()) for q, yy in zip(qid, y)])
res['gold_code_among_same_question_claude_labels'] = dict(all=cov.mean(), unseen_pair=cov[up].mean())
# head on Claude reasoning (fit on all pool rows; dev photos are not in the pool)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
lr = LogisticRegression(C=4, max_iter=2000, class_weight='balanced').fit(vec.fit_transform(U.text), [c[0] for c in U.codes])
Hp = np.full((len(U), K), 1e-6); pp = lr.predict_proba(vec.transform(U.text))
for j, c in enumerate(lr.classes_): Hp[:, ci[c]] = pp[:, j]
Hp = normalize(Hp)  # note: in-sample for pool rows (optimistic for variant b)
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(list(U.tx) + [tx[t] for t in ids])
Ux = cv.transform(U.tx); Dx = cv.transform([tx[t] for t in ids])
byq = U.groupby('q').indices
shares = (.6, .25, .15)
out = {}
for k in (5, 15, 50):
    V = np.full((len(ids), K), 1e-4); Hk = np.full((len(ids), K), 1/K)
    for i, (t, q) in enumerate(zip(ids, qid)):
        ix = byq.get(q)
        if ix is None or len(ix) == 0: continue
        sim = (Ux[ix] @ Dx[i].T).toarray().ravel(); o = np.argsort(-sim)[:k]; w = np.maximum(sim[o], 1e-3)
        for wi, j in zip(w, ix[o]):
            for s, c in zip(shares, U.codes.iat[j]):
                if c in ci: V[i, ci[c]] += wi*s
        Hk[i] = (w[:, None]*Hp[ix[o]]).sum(0)/w.sum()
    for name, P in (('vote', V), ('head_on_neighbours', Hk)):
        Pm = normalize(normalize(P)*mask + 1e-9); a = row_scores(y, topk(Pm, classes))
        sc = {}
        for wf in (0, .2, .35, .5, .65):
            F = geo([(B, 1-wf), (Pm, wf)]) if wf else B
            for l in (0, .25, .5, .75):
                sc[(wf, l)] = row_scores(y, topk(sinkhorn(F, pi, l) if l else F, classes))
        o, ch = crossfit(sc, mf)
        bs, _ = crossfit({l: sc[(0, l)] for l in (0, .25, .5, .75)}, mf)
        out[f'{name}_k{k}'] = dict(alone_top10=a.mean(), unseen_pair=a[up].mean(), fused_joint_cf=o.mean(),
                                  ci_vs_ref=boot_ci(o, ref, qid), ci_vs_blend_sink_cf=boot_ci(o, bs, qid), cfg=[list(c) for c in ch])
        print(name, k, out[f'{name}_k{k}']['alone_top10'], out[f'{name}_k{k}']['unseen_pair'], o.mean(), flush=True)
res['students'] = out
dump(res, 't5_retrieval_student.json')
