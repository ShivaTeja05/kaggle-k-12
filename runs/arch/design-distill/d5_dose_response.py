"""Dose-response: how good must a student's rationale be for the fusion gain to survive?
For a fraction (1-p) of dev rows, Claude's real rationale is replaced by Claude's rationale for a
DIFFERENT unlabelled photo of the same question (same topic if the question has none) = a fluent but
memorised, row-unspecific rationale (what a question-memorising student would write). Stage = LR x desc
(as d4), fused with the blend, weight cross-fitted on 5 question folds; 3 seeds."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from k12.core import read_csv, label_space, make_folds, aligned_bundle, row_scores, topk, normalize
tax = read_csv(ROOT/'data/raw/taxonomy_v1.csv'); classes = label_space(tax); ci = {c:i for i,c in enumerate(classes)}
train = read_csv(ROOT/'data/raw/train.csv'); fold = make_folds(train)
dev = train[fold==0].reset_index(drop=True)
ids = dev.trace_id.tolist(); y = dev.root_code.tolist(); qid = dev.question_id.values
unl = read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
CODE_RE = re.compile(r'\b(?:[A-Z][A-Za-z]{1,3}-\d{2}|NONE)\b'); cl = lambda t: CODE_RE.sub(' ', t or '')
U = {}
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') in ci and r['trace_id'] in unl.index: U[r['trace_id']] = a
D = {json.loads(l)['trace_id']: json.loads(l).get('answer') or {} for l in open(ROOT/'runs/claude/dev_opus.jsonl')}
tr_txt = [cl(a.get('reasoning')) for a in U.values()]; Y = np.array([ci[a['first_code']] for a in U.values()])
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000, sublinear_tf=True)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=200000, sublinear_tf=True)
Xtr = hstack([vw.fit_transform(tr_txt), vc.fit_transform(tr_txt)]).tocsr()
lr = LogisticRegression(C=8.0, max_iter=300).fit(Xtr, Y)
desc = {r.id: f"{r.name}. {r.description}" for r in tax.itertuples()}; desc['NONE'] = 'No error. The working is correct and matches the answer key. No major error.'
docs = [desc[c] for c in classes]
vd = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words='english').fit(docs + tr_txt[:20000]); Dm = vd.transform(docs)
def stage(texts):
    X = hstack([vw.transform(texts), vc.transform(texts)]).tocsr()
    p = np.full((len(texts), len(classes)), 1e-6); p[:, lr.classes_] += lr.predict_proba(X)
    return normalize(np.exp(np.log(normalize(p)+1e-9) + 40*(vd.transform(texts) @ Dm.T).toarray()))
uq_of = unl.loc[list(U)]; byq, byt = {}, {}
for t, q, tp in zip(uq_of.index, uq_of.question_id, uq_of.topic_name):
    byq.setdefault(q, []).append(t); byt.setdefault(tp, []).append(t)
P = [aligned_bundle(ROOT/b, ids, classes) for b in ['runs/pod/text32/validation_tx_dev_epoch_1.npz','runs/pod/tx/text_tx_dev_e3.npz','runs/pod/dev-distill/validation_epoch_2.npz']]
blend = normalize(np.exp(sum(w*np.log(p+1e-9) for w,p in zip((.35,.30,.35), P))))
sb = row_scores(y, topk(blend, classes)); uq = np.unique(qid)
real = [cl(D.get(t, {}).get('reasoning')) for t in ids]
grid = np.linspace(0, 1, 21); res = {}
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    outs, singles = [], []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        fake = []
        for r in dev.itertuples():
            pool = byq.get(r.question_id) or byt.get(r.topic_name) or list(U)
            fake.append(cl(U[pool[rng.integers(len(pool))]].get('reasoning')))
        keep = rng.random(len(ids)) < frac
        ps = stage([a if k else b for a, b, k in zip(real, fake, keep)])
        singles.append(row_scores(y, topk(ps, classes)).mean())
        qf = {q: i%5 for i,q in enumerate(rng.permutation(uq))}; df = np.array([qf[q] for q in qid]); cf = np.zeros(len(ids))
        f = lambda w, ix: normalize(np.exp((1-w)*np.log(blend[ix]+1e-9) + w*np.log(ps[ix]+1e-9)))
        for k in range(5):
            tr = np.flatnonzero(df!=k); te = np.flatnonzero(df==k)
            w = max(grid, key=lambda w: row_scores([y[i] for i in tr], topk(f(w, tr), classes)).mean())
            cf[te] = row_scores([y[i] for i in te], topk(f(w, te), classes))
        outs.append(cf.mean())
    res[str(frac)] = {'stage_single': round(float(np.mean(singles)),4), 'blend_plus_crossfit': round(float(np.mean(outs)),4), 'delta_vs_0.7182': round(float(np.mean(outs)-sb.mean()),4)}
    print(frac, res[str(frac)])
json.dump(res, open(ROOT/'runs/arch/design-distill/d5_dose_response.json','w'), indent=1)
