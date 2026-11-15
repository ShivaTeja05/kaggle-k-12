"""CPU probe (kNN-LM / Eedi generate-retrieve-rerank idea): within-question nearest neighbours over labelled transcriptions.
Datastore: fit-fold gold rows (annotator transcription, gold code, weight 1) + 26k Claude-labelled unlabelled photos
(Claude transcription, Claude first code, weight wc). Query: machine transcript of each dev row (runs/pod/tx/tx_dev.json).
Fused into the dev blend as log B + w*log P_knn, w cross-fitted over 5 question folds. Aggregate metrics only."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
tr, tax, classes, folds, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
unl = pd.read_csv(ROOT/'data/pseudo/x/competition_data/unlabelled.csv', usecols=['trace_id', 'question_id'])
q_of = dict(zip(unl.trace_id, unl.question_id))
docs, labs, qs, src = [], [], [], []
for t, l, q in zip(fit.transcription_steps, fit.root_code, fit.question_id):
    docs.append(str(t)); labs.append(ci[l]); qs.append(q); src.append(0)
for line in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); a = r.get('answer') or {}; q = q_of.get(r['trace_id'])
    if q is None or a.get('first_code') not in ci or not a.get('transcription'): continue
    docs.append(a['transcription']); labs.append(ci[a['first_code']]); qs.append(q); src.append(1)
labs = np.array(labs); qs = np.array(qs); src = np.array(src)
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
D = vec.fit_transform(docs); Qm = vec.transform([tx[i] for i in ids])
byq = {q: np.where(qs == q)[0] for q in np.unique(qs)}
def knn(k, T, wc, alpha=.5):
    P = np.full((len(ids), K), 1. / K); has = np.zeros(len(ids), bool)
    for r, q in enumerate(qid):
        idx = byq.get(q)
        if idx is None: continue
        s = (D[idx] @ Qm[r].T).toarray().ravel()
        top = np.argsort(-s)[:k]; w = np.exp((s[top] - s[top].max()) / T) * np.where(src[idx[top]] == 1, wc, 1.)
        v = np.bincount(labs[idx[top]], weights=w, minlength=K) + alpha / K
        P[r] = v / v.sum(); has[r] = True
    return P, has
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
pi = np.bincount(fit.root_code.map(ci), minlength=K) + .5; pi /= pi.sum()
def sink(P, lam=.5):
    b = np.zeros(K)
    for _ in range(30): Q = normalize(P*np.exp(b)); b += np.log(pi/Q.mean(0))
    return normalize(P*np.exp(lam*b))
Bs = sink(B); refs = row_scores(y, topk(Bs, classes))
qu = np.unique(qid); rng = np.random.default_rng(7); rng.shuffle(qu); g = {q: i % 5 for i, q in enumerate(qu)}
mf = np.array([g[q] for q in qid]); W = [0, .1, .2, .35, .5, .75]
def fuse(base, P, w): lp = np.log(base) + w*np.log(P); return normalize(np.exp(lp - lp.max(1, keepdims=True)))
def crossfit(base, P):
    pts = np.zeros(len(ids)); ch = []
    for f in range(5):
        tr_ = mf != f
        best = max(W, key=lambda w: row_scores(y[tr_], topk(fuse(base[tr_], P[tr_], w), classes)).mean())
        ch.append(best); te = mf == f
        pts[te] = row_scores(y[te], topk(fuse(base[te], P[te], best), classes))
    return pts, ch
out = {'datastore_rows': int(len(docs)), 'claude_rows': int(src.sum()), 'ref': float(ref.mean()), 'ref_sinkhorn0.5': float(refs.mean()), 'configs': {}}
for k, T, wc in [(10, .05, .5), (20, .05, .5), (20, .1, 1.), (50, .1, .5), (20, .05, .2), (20, .05, 1.)]:
    P, has = knn(k, T, wc); alone = row_scores(y, topk(P, classes))
    p1, c1 = crossfit(B, P); p2, c2 = crossfit(Bs, P)
    out['configs'][f'k{k}_T{T}_wc{wc}'] = {
        'coverage': float(has.mean()), 'knn_alone': float(alone.mean()), 'knn_alone_covered': float(alone[has].mean()),
        'fused_ref': float(p1.mean()), 'delta_ref_ci': boot_ci(p1, ref, qid), 'w': c1,
        'fused_sinkhorn': float(p2.mean()), 'delta_sinkhorn_ci': boot_ci(p2, refs, qid), 'w_s': c2}
    print(k, T, wc, json.dumps(out['configs'][f'k{k}_T{T}_wc{wc}']))
json.dump(out, open(ROOT/'runs/research/knn-probe/knn_probe.json', 'w'), indent=1)
