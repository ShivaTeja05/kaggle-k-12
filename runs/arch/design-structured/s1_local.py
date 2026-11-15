"""CPU check for the structured-decomposition proposal.

Question: if the root line were located, does a cheap, question-agnostic classifier on that
line's text generalise to (question, code) pairs never seen in training, where the blend
collapses (0.43 vs Claude 0.71)? And does it add to the blend under cross-fitting?

Variants (all fit on non-dev data only, evaluated on the 301 dev rows, fold 0):
  L_gold  : TF-IDF char n-grams on the GOLD root line(s) + span  (oracle localisation)
  L_whole : same model on the whole MACHINE transcript (no localisation; realistic)
Training rows: 1,199 gold fit rows (error rows), optionally + Claude unlabelled rows
(root_line text -> first_code). Digits are optionally masked to stop number memorisation.
"""
import json, re, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from k12.core import aligned_bundle, label_space, make_folds, map3, normalize, read_csv, row_scores, topk

rng = np.random.default_rng(0)
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv')
classes = label_space(tax); ci = {c: i for i, c in enumerate(classes)}
section = dict(zip(tax.id, tax.section)); section['NONE'] = 'NONE'
folds = make_folds(train); fit = train[folds != 0].reset_index(drop=True); dev = train[folds == 0].reset_index(drop=True)
ids = dev.trace_id.tolist(); y = dev.root_code.values
A = aligned_bundle('runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes)
B = aligned_bundle('runs/pod/tx/text_tx_dev_e3.npz', ids, classes)
C = aligned_bundle('runs/pod/dev-distill/validation_epoch_2.npz', ids, classes)
blend = normalize(np.exp(.35*np.log(A)+.30*np.log(B)+.35*np.log(C)))
ref = row_scores(y, topk(blend, classes)); print('reference', ref.mean().round(4))
tx = json.load(open('runs/pod/tx/tx_dev.json'))

def lines(t):
    out = {}
    for l in str(t).split('\n'):
        m = re.match(r'\s*L(\d+)\s*\|\s*(.*)', l)
        if m: out[int(m.group(1))] = m.group(2)
    return out

def root_text(r):
    ms = json.loads(r.misconceptions) if r.misconceptions else []
    ls = lines(r.transcription_steps)
    for m in ms:
        if m.get('root'):
            nums = [int(x) for x in re.findall(r'\d+', str(m['step']))]
            if len(nums) == 2: nums = list(range(nums[0], nums[1]+1))
            body = ' '.join(ls.get(n, '') for n in nums)
            return (body + ' || ' + str(m.get('span', ''))).strip()
    return ''

MASK = '--mask' in sys.argv
def prep(s):
    s = str(s).lower()
    return re.sub(r'\d', '0', s) if MASK else s

fit_err = fit[fit.root_code != 'NONE']
Xg = [root_text(r) for r in fit_err.itertuples()]; yg = fit_err.root_code.tolist()
# Claude unlabelled: root_line text -> first_code (question-agnostic extra supervision)
Xc, yc = [], []
for l in open('runs/claude/unl_all_v4.jsonl'):
    a = json.loads(l).get('answer') or {}
    if a.get('first_code') not in (None, 'NONE') and a.get('root_line'):
        Xc.append(a['root_line']); yc.append(a['first_code'])
print('gold fit error rows', len(Xg), 'claude rows with root_line', len(Xc))

err_codes = [c for c in classes if c != 'NONE']
def fit_model(X, Y, w=None):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
    Z = vec.fit_transform([prep(x) for x in X])
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight='balanced')
    clf.fit(Z, Y, sample_weight=w)
    return vec, clf

def predict(vec, clf, X):
    p = clf.predict_proba(vec.transform([prep(x) for x in X]))
    out = np.full((len(X), len(err_codes)), 1e-6)
    for j, c in enumerate(clf.classes_): out[:, err_codes.index(c)] = p[:, j]
    return out / out.sum(1, keepdims=True)

dev_gold_root = [root_text(r) for r in dev.itertuples()]
dev_tx = [tx[t] for t in ids]
seen = {}
for r in fit.itertuples(): seen.setdefault(r.question_id, set()).add(r.root_code)
is_err = y != 'NONE'
bin_ = np.array(['NONE' if c == 'NONE' else ('code_seen' if c in seen.get(q, set()) else ('code_unseen' if q in seen else 'q_unseen'))
                 for c, q in zip(y, dev.question_id)])
ei = [ci[c] for c in err_codes]
blend_err = blend[:, ei] / blend[:, ei].sum(1, keepdims=True)

def errstats(P, name):
    t = topk(P, err_codes); s = row_scores(y, t)
    sec = np.array([section[t[i, 0]] == section[y[i]] for i in range(len(y))])
    res = {'all_err_map': s[is_err].mean(), 'sec_acc_err': sec[is_err].mean()}
    for b in ['code_seen', 'code_unseen', 'q_unseen']:
        res[b] = s[bin_ == b].mean()
    print(f'{name:28s}', {k: round(float(v), 3) for k, v in res.items()})
    return res

out = {'mask_digits': MASK, 'bin_counts': {b: int((bin_ == b).sum()) for b in set(bin_)}}
out['blend_errcond'] = errstats(blend_err, 'blend (error-conditional)')
models = {}
for tag, X, Y, w in [('gold_only', Xg, yg, None),
                     ('gold+claude', Xg + Xc, yg + yc, [3.0]*len(Xg) + [1.0]*len(Xc))]:
    vec, clf = fit_model(X, Y, w)
    models[tag] = (vec, clf)
    for vt, Xd in [('oracle_root', dev_gold_root), ('whole_tx', dev_tx)]:
        P = predict(vec, clf, Xd); out[f'{tag}|{vt}'] = errstats(P, f'{tag}|{vt}')
        np.save(f'runs/arch/design-structured/P_{tag}_{vt}{"_mask" if MASK else ""}.npy', P)

# Fusion into the blend: error-conditional geometric mix, NONE mass kept from the blend.
qs = dev.question_id.values; uq = np.unique(qs)
def fuse(P, w):
    e = normalize(np.exp((1-w)*np.log(blend_err) + w*np.log(P)))
    full = blend.copy(); pn = blend[:, ci['NONE']:ci['NONE']+1]
    full[:, ei] = (1-pn) * e
    return full
def crossfit(P, grid=np.round(np.arange(0, .61, .05), 2), reps=10):
    res = []
    for rep in range(reps):
        perm = np.random.default_rng(rep).permutation(uq); fq = {q: i % 5 for i, q in enumerate(perm)}
        f = np.array([fq[q] for q in qs]); sc = np.zeros(len(y))
        cache = {w: row_scores(y, topk(fuse(P, w), classes)) for w in grid}
        for k in range(5):
            tr = f != k
            wb = max(grid, key=lambda w: (cache[w][tr].mean(), -w))
            sc[~tr] = cache[wb][~tr]
        res.append(sc)
    return np.mean(res, 0), {float(w): float(cache[w].mean()) for w in grid}
def boot(d, n=4000):
    g = {q: np.flatnonzero(qs == q) for q in uq}; r = np.random.default_rng(1); m = []
    for _ in range(n):
        pick = r.choice(uq, len(uq)); idx = np.concatenate([g[q] for q in pick]); m.append(d[idx].mean())
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
for key in ['gold+claude|oracle_root', 'gold+claude|whole_tx', 'gold_only|oracle_root', 'gold_only|whole_tx']:
    tag, vt = key.split('|')
    P = np.load(f'runs/arch/design-structured/P_{tag}_{vt}{"_mask" if MASK else ""}.npy')
    sc, insample = crossfit(P)
    d = sc - ref
    out[f'fusion|{key}'] = {'crossfit_map': round(float(sc.mean()), 4), 'delta': round(float(d.mean()), 4), 'ci': boot(d),
                            'in_sample_by_w': {k: round(v, 4) for k, v in insample.items()},
                            'by_bin_delta': {b: round(float(d[bin_ == b].sum()/len(y)), 4) for b in set(bin_)}}
    print('fusion', key, out[f'fusion|{key}']['crossfit_map'], out[f'fusion|{key}']['delta'], out[f'fusion|{key}']['ci'])
json.dump(out, open(f'runs/arch/design-structured/s1_local{"_mask" if MASK else ""}.json', 'w'), indent=1, default=float)
