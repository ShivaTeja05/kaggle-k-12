"""CPU check: is mechanism-text -> code mapping learnable question-agnostically, and how much does a verifier
that sees a (Claude-quality) rationale add when restricted to the blend's top-10? Trains TF-IDF+LR on the 26k
Claude reasonings of UNLABELLED photos (code ids masked), scores dev rationales from runs/claude/dev_opus.jsonl.
Dev rationales are a teacher oracle stand-in: this bounds the 'mapping' half, not what a student can generate."""
import sys, json, re
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
P = blend(ids, classes); ref = row_scores(y, topk(P, classes))
mask = re.compile(r'\b[A-Z]{3}-\d{2}\b|\bNONE\b')
def txt(a): return mask.sub(' CODE ', (a.get('reasoning') or '') + ' ROOTLINE ' + str(a.get('root_line') or ''))
X, Y = [], []
for line in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    a = json.loads(line).get('answer') or {}
    if a.get('first_code') in classes and a.get('reasoning'): X.append(txt(a)); Y.append(a['first_code'])
dv = {}
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); dv[r['trace_id']] = txt(r.get('answer') or {})
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
Xt = vec.fit_transform(X); lr = LogisticRegression(C=4, max_iter=2000).fit(Xt, Y)
Vd = lr.predict_proba(vec.transform([dv.get(t, '') for t in ids]))
V = np.full((len(ids), len(classes)), 1e-6); ci = {c: i for i, c in enumerate(classes)}
for j, c in enumerate(lr.classes_): V[:, ci[c]] = Vd[:, j] + 1e-6
V = normalize(V)
def sc(M): return row_scores(y, topk(M, classes))
top10 = np.argsort(-P, 1)[:, :10]; V10 = np.full_like(V, 1e-6)
for i in range(len(y)): V10[i, top10[i]] = V[i, top10[i]]
V10 = normalize(V10)
qs = np.unique(qid).copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
def geo(a, b, w):
    lp = np.log(a) + w*np.log(b); return normalize(np.exp(lp - lp.max(1, keepdims=True)))
cands = {w: sc(geo(P, V10, w)) for w in [0, .25, .5, 1, 2]}
out = np.empty(len(y)); ch = []
for f in range(5):
    m = mf != f; w = max(cands, key=lambda w: cands[w][m].mean()); ch.append(w); out[~m] = cands[w][~m]
d, c = boot_ci(out, ref, qid)
fitcodes = fit.groupby('question_id').root_code.agg(set).to_dict()
unseen_pair = np.array([(yy != 'NONE') and (yy not in fitcodes.get(q, set())) for yy, q in zip(y, qid)])
res = dict(n_train=len(Y), mapper_alone_full=float(sc(V).mean()), mapper_alone_top10=float(sc(V10).mean()),
           fused_crossfit=float(out.mean()), delta=d, ci95=c, w_choices=ch,
           unseen_pair_rows=int(unseen_pair.sum()), unseen_pair_ref=float(ref[unseen_pair].mean()),
           unseen_pair_fused=float(out[unseen_pair].mean()), seen_rows_ref=float(ref[~unseen_pair].mean()),
           seen_rows_fused=float(out[~unseen_pair].mean()))
print(json.dumps(res, indent=1)); json.dump(res, open('mechanism_map.json', 'w'), indent=1)
