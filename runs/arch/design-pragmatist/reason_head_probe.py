"""Kill test for 'classify after reasoning': can a cheap learned head map reasoning TEXT (code ids stripped) to the code,
across questions? Train TF-IDF+logreg on Claude's 26k unlabelled-photo reasoning -> first_code; apply to Claude's dev
reasoning; restrict to blend top-10 (+NONE); fuse with the blend (cross-fitted weight); Sinkhorn optional."""
import sys, json, re
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/design-pragmatist')
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}
CODE = re.compile(r'\b[A-Za-z]{2,3}-\d{2}\b|\bNONE\b')
def txt(a): return CODE.sub(' ', (a.get('root_line') or '') + ' || ' + (a.get('reasoning') or ''))
X, Y = [], []
for line in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); a = r.get('answer') or {}
    if a.get('first_code') in ci and a.get('reasoning'): X.append(txt(a)); Y.append(a['first_code'])
dv = {}
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); dv[r['trace_id']] = txt(r['answer'])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
Xt = vec.fit_transform(X); lr = LogisticRegression(C=4, max_iter=2000, class_weight='balanced').fit(Xt, Y)
Pd = np.full((len(ids), len(classes)), 1e-6)
pp = lr.predict_proba(vec.transform([dv[t] for t in ids]))
for j, c in enumerate(lr.classes_): Pd[:, ci[c]] = pp[:, j]
Pd = normalize(Pd)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
Pm = normalize(Pd*mask + 1e-9)
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
def sink(P, lam=.5):
    b = np.zeros(P.shape[1])
    for _ in range(30): Q = normalize(P*np.exp(b)); b += np.log(pi/Q.mean(0))
    return normalize(P*np.exp(lam*b))
qs = np.unique(qid); np.random.default_rng(7).shuffle(qs); g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
W = [0, .2, .35, .5, .65, .8]
def cf(sk):
    c = {}
    for w in W:
        lp = (1-w)*np.log(B) + w*np.log(Pm); P = normalize(np.exp(lp - lp.max(1, keepdims=True)))
        c[w] = row_scores(y, topk(sink(P) if sk else P, classes))
    out = np.empty(len(y)); ch = []
    for f in range(5):
        m = mf != f; w = max(W, key=lambda w: c[w][m].mean()); ch.append(w); out[~m] = c[w][~m]
    return out, ch
a_full = row_scores(y, topk(Pd, classes)); a_m = row_scores(y, topk(Pm, classes))
o1, ch1 = cf(False); o2, ch2 = cf(True)
res = dict(n_train=len(Y), head_alone_full=a_full.mean(), head_alone_top10=a_m.mean(), head_top10_unseen_pair=a_m[up].mean(),
           blend_unseen_pair=ref[up].mean(), fused_cf=o1.mean(), fused_cf_ci=boot_ci(o1, ref, qid), w_choices=ch1,
           fused_cf_unseen_pair=o1[up].mean(), fused_cf_sinkhorn=o2.mean(), fused_cf_sinkhorn_ci=boot_ci(o2, ref, qid), w_choices_sink=ch2)
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(ROOT/'runs/arch/design-pragmatist/reason_head_probe.json', 'w'), indent=1, default=float)
