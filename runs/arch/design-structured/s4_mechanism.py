"""Oracle-mechanism check: if stage 2 produced a mechanism description as good as the annotator's
(misconceptions[root].text), can a cheap, question-agnostic mapper (TF-IDF word+char LR, trained on
fit-row annotator texts + Claude reasoning -> first_code) pick the code on unseen (question, code)
pairs? Also: mapper over taxonomy descriptions only (zero-shot nearest description)."""
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv'); classes = label_space(tax)
folds = make_folds(train); fit = train[folds != 0].reset_index(drop=True); dev = train[folds == 0].reset_index(drop=True)
y = dev.root_code.values; err = y != 'NONE'
seen = {}
for r in fit.itertuples(): seen.setdefault(r.question_id, set()).add(r.root_code)
bin_ = np.array(['NONE' if c == 'NONE' else ('code_seen' if c in seen.get(q, set()) else ('code_unseen' if q in seen else 'q_unseen')) for c, q in zip(y, dev.question_id)])
def root_txt(m):
    for a in (json.loads(m) if m else []):
        if a.get('root'): return a.get('text', '')
    return ''
err_codes = [c for c in classes if c != 'NONE']
Xf = [root_txt(m) for m in fit.misconceptions[fit.root_code != 'NONE']]; yf = fit.root_code[fit.root_code != 'NONE'].tolist()
Xc, yc = [], []
for l in open('runs/claude/unl_all_v4.jsonl'):
    a = json.loads(l).get('answer') or {}
    if a.get('first_code') not in (None, 'NONE') and a.get('reasoning'):
        Xc.append(a['reasoning']); yc.append(a['first_code'])
Xd = [root_txt(m) for m in dev.misconceptions]
def run(X, Y, w):
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150000))
    clf = LogisticRegression(C=4, max_iter=3000, class_weight='balanced').fit(vec.fit_transform(X), Y, sample_weight=w)
    p = clf.predict_proba(vec.transform(Xd)); P = np.full((len(Xd), len(err_codes)), 1e-6)
    for j, c in enumerate(clf.classes_): P[:, err_codes.index(c)] = p[:, j]
    return P / P.sum(1, keepdims=True)
def stats(P):
    s = row_scores(y, topk(P, err_codes))
    return {'err': round(float(s[err].mean()), 3), **{b: round(float(s[bin_ == b].mean()), 3) for b in ['code_seen', 'code_unseen', 'q_unseen']}}
out = {}
out['annot_text_gold_only'] = stats(run(Xf, yf, None))
out['annot_text_claude_only'] = stats(run(Xc, yc, None))
out['annot_text_gold+claude'] = stats(run(Xf + Xc, yf + yc, [3.]*len(Xf) + [1.]*len(Xc)))
# zero-shot: cosine to taxonomy name+description
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(list(tax.name + ' ' + tax.description) + Xf)
T = vec.transform(list(tax.name + ' ' + tax.description)); D = vec.transform(Xd)
S = (D @ T.T).toarray(); P = np.full((len(Xd), len(err_codes)), 1e-6)
for j, c in enumerate(tax.id): P[:, err_codes.index(c)] += np.exp(10 * S[:, j])
out['zero_shot_description_match'] = stats(P)
L = lambda p: aligned_bundle(p, dev.trace_id.tolist(), classes)
blend = normalize(np.exp(.35*np.log(L('runs/pod/text32/validation_tx_dev_epoch_1.npz'))+.30*np.log(L('runs/pod/tx/text_tx_dev_e3.npz'))+.35*np.log(L('runs/pod/dev-distill/validation_epoch_2.npz'))))
ei = [classes.index(c) for c in err_codes]; out['blend_errcond'] = stats(blend[:, ei])
print(json.dumps(out, indent=1)); json.dump(out, open('runs/arch/design-structured/s4_mechanism.json', 'w'), indent=1)
