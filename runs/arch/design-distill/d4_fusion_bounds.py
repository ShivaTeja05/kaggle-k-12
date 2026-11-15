"""Fusion bounds: blend (optionally Sinkhorn-balanced is NOT applied here) + a rationale stage.
Stages: (i) LR x desc on Claude dev reasoning (bound for a Claude-quality rationale student with a shallow head),
(ii) Claude's own codes (bound for a perfect rationale+decision student). Weight cross-fitted, 5 question folds, 3 seeds."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from k12.core import read_csv, label_space, make_folds, aligned_bundle, row_scores, topk, normalize
from evaluate_claude import claude_scores
tax = read_csv(ROOT/'data/raw/taxonomy_v1.csv'); classes = label_space(tax); ci = {c:i for i,c in enumerate(classes)}
train = read_csv(ROOT/'data/raw/train.csv'); fold = make_folds(train)
dev = train[fold==0].reset_index(drop=True); fit = train[fold!=0]
ids = dev.trace_id.tolist(); y = dev.root_code.tolist(); qid = dev.question_id.values
unl = read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
CODE_RE = re.compile(r'\b(?:[A-Z][A-Za-z]{1,3}-\d{2}|NONE)\b'); cl = lambda t: CODE_RE.sub(' ', t or '')
U = {}
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') in ci and r['trace_id'] in unl.index: U[r['trace_id']] = a
D = {json.loads(l)['trace_id']: json.loads(l).get('answer') or {} for l in open(ROOT/'runs/claude/dev_opus.jsonl')}
tr_txt = [cl(a.get('reasoning')) for a in U.values()]; Y = np.array([ci[a['first_code']] for a in U.values()])
dv_txt = [cl(D.get(t, {}).get('reasoning')) for t in ids]
vw = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000, sublinear_tf=True)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=200000, sublinear_tf=True)
Xtr = hstack([vw.fit_transform(tr_txt), vc.fit_transform(tr_txt)]).tocsr(); Xdv = hstack([vw.transform(dv_txt), vc.transform(dv_txt)]).tocsr()
lr = LogisticRegression(C=8.0, max_iter=300).fit(Xtr, Y)
plr = np.full((len(ids), len(classes)), 1e-6); plr[:, lr.classes_] += lr.predict_proba(Xdv); plr = normalize(plr)
desc = {r.id: f"{r.name}. {r.description}" for r in tax.itertuples()}; desc['NONE'] = 'No error. The working is correct and matches the answer key. No major error.'
docs = [desc[c] for c in classes]
vd = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, stop_words='english').fit(docs + tr_txt[:20000])
S = (vd.transform(dv_txt) @ vd.transform(docs).T).toarray()
pstage = normalize(np.exp(np.log(plr+1e-9) + 40*S))
pcl, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
fc = fit.groupby('question_id').root_code.apply(set).to_dict()
bins = np.array(['NONE' if r.root_code=='NONE' else 'q_unseen' if r.question_id not in fc else 'code_seen_for_q' if r.root_code in fc[r.question_id] else 'code_unseen_for_q' for r in dev.itertuples()])
P = [aligned_bundle(ROOT/b, ids, classes) for b in ['runs/pod/text32/validation_tx_dev_epoch_1.npz','runs/pod/tx/text_tx_dev_e3.npz','runs/pod/dev-distill/validation_epoch_2.npz']]
blend = normalize(np.exp(sum(w*np.log(p+1e-9) for w,p in zip((.35,.30,.35), P))))
sb = row_scores(y, topk(blend, classes))
rng = np.random.default_rng(1); uq = np.unique(qid)
boots = [np.concatenate([np.flatnonzero(qid==q) for q in rng.choice(uq, len(uq))]) for _ in range(4000)]
def ci_(a, b):
    d = np.array([a[i].mean()-b[i].mean() for i in boots]); return [round(float(np.percentile(d,2.5)),4), round(float(np.percentile(d,97.5)),4)]
pb = lambda s: {b: round(float(s[bins==b].mean()),4) for b in ['code_seen_for_q','code_unseen_for_q','q_unseen','NONE']}
grid = np.linspace(0, 1, 21)
res = {'blend': [round(float(sb.mean()),4), pb(sb)]}
for name, ps in [('rationale_lr_x_desc', pstage), ('claude_codes', pcl)]:
    outs = []
    for seed in range(3):
        r2 = np.random.default_rng(100+seed); qf = {q: i%5 for i,q in enumerate(r2.permutation(uq))}; df = np.array([qf[q] for q in qid])
        cf = np.zeros(len(ids))
        for k in range(5):
            tr = np.flatnonzero(df!=k); te = np.flatnonzero(df==k)
            f = lambda w, ix: normalize(np.exp((1-w)*np.log(blend[ix]+1e-9) + w*np.log(ps[ix]+1e-9)))
            w = max(grid, key=lambda w: row_scores([y[i] for i in tr], topk(f(w, tr), classes)).mean())
            cf[te] = row_scores([y[i] for i in te], topk(f(w, te), classes))
        outs.append(cf)
    cf = np.mean(outs, 0)
    fixed = {str(w): round(float(row_scores(y, topk(normalize(np.exp((1-w)*np.log(blend+1e-9)+w*np.log(ps+1e-9))), classes)).mean()),4) for w in (0.2, 0.35, 0.5)}
    res[name] = {'single': round(float(row_scores(y, topk(ps, classes)).mean()),4), 'blend_plus_crossfit': round(float(cf.mean()),4),
                 'delta': round(float(cf.mean()-sb.mean()),4), 'ci': ci_(cf, sb), 'bins': pb(cf), 'fixed_weight_insample': fixed}
print(json.dumps(res, indent=1)); json.dump(res, open(ROOT/'runs/arch/design-distill/d4_fusion_bounds.json','w'), indent=1)
