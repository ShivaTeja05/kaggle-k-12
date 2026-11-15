"""W3 test 6 (proxy for approach A1): if the transcriber read like Claude, would a downstream text classifier gain?
A TF-IDF + logistic model is trained once (fit gold transcripts + Claude-labelled unlabelled photos with Claude transcripts,
the same input mix the text members see) and scored on dev three times: gold / Claude / machine transcripts.
Question-grouped paired bootstrap. Also the paired comparison for the 32B/8B/VL members on M6 rows.
Run with runs/deep/symbolic/venv/bin/python."""
import json, sys
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import load, ROOT, boot_ci
from k12.core import aligned_bundle, topk, row_scores, normalize
tr, tax, classes, folds, dev, fit = load(); ci = {c: i for i, c in enumerate(classes)}
y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
w1 = pd.read_csv(ROOT/'runs/deep/approach-W3/w1_rows.csv', keep_default_na=False).set_index('trace_id')
grp = w1.grp.reindex(dev.trace_id).to_numpy(); m6 = grp == 'M6'; err = y != 'NONE'
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
cl = {}
for l in open(ROOT/'runs/claude/dev_opus.jsonl'):
    j = json.loads(l); a = j.get('answer') or {}
    cl[j['trace_id']] = a.get('transcription', '')
# members on M6 rows (paired, question-grouped)
P = {k: aligned_bundle(ROOT/p, dev.trace_id.tolist(), classes) for k, p in
     [('32B', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8B', 'runs/pod/tx/text_tx_dev_e3.npz'), ('VL', 'runs/pod/dev-distill/validation_epoch_2.npz')]}
S = {k: row_scores(y, topk(v, classes)) for k, v in P.items()}
out = {'members': {}}
for sub, m in [('M6', m6), ('error rows not M6', err & ~m6), ('NONE rows', ~err)]:
    for a, b in [('VL', '32B'), ('VL', '8B')]:
        d, c = boot_ci(S[a][m], S[b][m], qid[m]); out['members'][f'{sub}: {a}-{b}'] = [round(S[a][m].mean(), 3), round(S[b][m].mean(), 3), round(d, 3), [round(x, 3) for x in c]]
print(json.dumps(out['members'], indent=1))
# proxy classifier
unl = pd.read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv', keep_default_na=False).set_index('trace_id')
rows = []
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    j = json.loads(l); a = j.get('answer') or {}
    if a.get('first_code') in ci and a.get('transcription') and j['trace_id'] in unl.index:
        rows.append((j['trace_id'], a['first_code'], a['transcription']))
c = pd.DataFrame(rows, columns=['trace_id', 'code', 'tx'])
dev_q = set(dev.question_id)
c['q'] = unl.question_id.reindex(c.trace_id).to_numpy()
rng = np.random.default_rng(0)
keep = (c.code != 'NONE') | (rng.random(len(c)) < 3000 / max(1, (c.code == 'NONE').sum()))
c = c[keep]
meta = unl.reindex(c.trace_id)
Xq = list(fit.question_text.astype(str) + ' || ' + fit.correct_answer.astype(str)) + list(meta.question_text.astype(str) + ' || ' + meta.correct_answer.astype(str))
Xt = list(fit.transcription_steps.astype(str)) + list(c.tx.astype(str))
Y = np.array([ci[v] for v in list(fit.root_code) + list(c.code)])
W = np.r_[np.full(len(fit), 3.0), np.ones(len(c))]
print('train rows', len(Y), '(fit', len(fit), '+ claude', len(c), ')')
vt = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000).fit(Xt)
vq = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(Xq)
dq = list(dev.question_text.astype(str) + ' || ' + dev.correct_answer.astype(str))
res = {}
for name, use_q in [('Q+T', True), ('T only', False)]:
    X = hstack([vt.transform(Xt), vq.transform(Xq)]).tocsr() if use_q else vt.transform(Xt)
    lr = LogisticRegression(C=4.0, max_iter=400).fit(X, Y, sample_weight=W)
    pts = {}
    for src, T in [('gold', list(dev.transcription_steps.astype(str))), ('claude', [cl.get(t, '') for t in dev.trace_id]), ('machine', [tx.get(t, '') for t in dev.trace_id])]:
        Xd = hstack([vt.transform(T), vq.transform(dq)]).tocsr() if use_q else vt.transform(T)
        p = np.zeros((len(dev), len(classes))); p[:, lr.classes_] = lr.predict_proba(Xd)
        pts[src] = row_scores(y, topk(normalize(p + 1e-9), classes))
    r = {s: {'all': round(v.mean(), 4), 'error': round(v[err].mean(), 4), 'M6': round(v[m6].mean(), 4)} for s, v in pts.items()}
    for a, b in [('claude', 'machine'), ('gold', 'machine'), ('gold', 'claude')]:
        r[f'{a}-{b}'] = {k: [round(x, 4) if not isinstance(x, list) else [round(z, 4) for z in x] for x in boot_ci(pts[a][m], pts[b][m], qid[m])]
                         for k, m in [('all', np.ones(len(y), bool)), ('error', err), ('M6', m6)]}
    res[name] = r
    print(name, json.dumps(r, indent=1))
out['proxy'] = res
json.dump(out, open(ROOT/'runs/deep/approach-W3/w6_reader_proxy.json', 'w'), indent=1)
