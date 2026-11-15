"""Same TF-IDF head as reason_head_probe.py, applied to OFFLINE Qwen reasoning tails (runs/pod/probe235/*/responses.jsonl,
code ids stripped) instead of Claude's dev reasoning. Proxy for 'offline reasoning text + learned head'."""
import sys, json, re
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}
CODE = re.compile(r'\b[A-Za-z]{2,3}-\d{2}\b|\bNONE\b|CODES:.*$', re.S)
X, Y = [], []
for line in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    a = json.loads(line).get('answer') or {}
    if a.get('first_code') in ci and a.get('reasoning'):
        X.append(CODE.sub(' ', (a.get('reasoning') or '')[-700:])); Y.append(a['first_code'])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
lr = LogisticRegression(C=4, max_iter=2000, class_weight='balanced').fit(vec.fit_transform(X), Y)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
fitpairs = set(zip(fit.question_id, fit.root_code)); up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
def sink(P, lam=.5):
    b = np.zeros(P.shape[1])
    for _ in range(30): Q = normalize(P*np.exp(b)); b += np.log(pi/Q.mean(0))
    return normalize(P*np.exp(lam*b))
qs = np.unique(qid); np.random.default_rng(7).shuffle(qs); g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
W = [0, .2, .35, .5, .65, .8]
def cf(Pm, sk):
    c = {}
    for w in W:
        lp = (1-w)*np.log(B) + w*np.log(Pm); P = normalize(np.exp(lp - lp.max(1, keepdims=True)))
        c[w] = row_scores(y, topk(sink(P) if sk else P, classes))
    out = np.empty(len(y))
    for f in range(5):
        m = mf != f; w = max(W, key=lambda w: c[w][m].mean()); out[~m] = c[w][~m]
    return out
sinkref = row_scores(y, topk(sink(B), classes))
res = {'blend_sinkhorn_fixed0.5': sinkref.mean()}
for d in ['zeroshot32', 'fewshot32', 'zeroshot235', 'fewshot235']:
    rs = {json.loads(l)['trace_id']: json.loads(l).get('tail', '') for l in open(ROOT/f'runs/pod/probe235/{d}/responses.jsonl')}
    Pd = np.full((len(ids), len(classes)), 1e-6); pp = lr.predict_proba(vec.transform([CODE.sub(' ', rs.get(t, '')) for t in ids]))
    for j, c in enumerate(lr.classes_): Pd[:, ci[c]] = pp[:, j]
    Pm = normalize(normalize(Pd)*mask + 1e-9)
    own = aligned_bundle(ROOT/f'runs/pod/probe235/{d}/validation.npz', ids, classes)
    a = row_scores(y, topk(Pm, classes)); o = cf(Pm, True)
    res[d] = dict(own_codes=row_scores(y, topk(own, classes)).mean(), head_on_tail_top10=a.mean(), head_unseen_pair=a[up].mean(),
                  fused_cf_sinkhorn=o.mean(), vs_ref=boot_ci(o, ref, qid), vs_blend_sinkhorn=boot_ci(o, sinkref, qid))
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(ROOT/'runs/arch/design-pragmatist/offline_reason_probe.json', 'w'), indent=1, default=float)
