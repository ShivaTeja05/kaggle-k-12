"""Stage 1 of the CPU test: build the question-blind TF-IDF mechanism->code mapper exactly as in
design-structured/s4-s6 (gold fit annotator text x3 + all Claude unl_all_v4 reasoning, code ids/names
stripped) and cache:
  P_dev_claude   : mapper output on Claude's dev reasoning (stripped)            [301 x 75]
  P_pool_oof     : mapper output on a pool of Claude unlabelled reasonings whose question_id is a dev
                   question, produced OUT OF FOLD (2 folds: each half of the pool is mapped by a mapper
                   that never saw it), so replacement mechanisms are not memorised training texts.
Nothing here touches the network. Outputs go to runs/arch/test-structured/."""
import json, re, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from k12.core import label_space, make_folds, read_csv

OUT = 'runs/arch/test-structured/'
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv'); classes = label_space(tax)
folds = make_folds(train); fit = train[folds != 0].reset_index(drop=True); dev = train[folds == 0].reset_index(drop=True)
err_codes = [c for c in classes if c != 'NONE']
names = sorted(tax.name.str.lower().tolist(), key=len, reverse=True)

def strip(t):
    t = re.sub(r'\b[A-Z]{2,4}-\d{2}\b', ' ', str(t)); tl = t.lower()
    for n in names: tl = tl.replace(n, ' ')
    return tl

def root_txt(m):
    for a in (json.loads(m) if m else []):
        if a.get('root'): return a.get('text', '')
    return ''

Xf = [strip(root_txt(m)) for m in fit.misconceptions[fit.root_code != 'NONE']]
yf = fit.root_code[fit.root_code != 'NONE'].tolist()
unl = read_csv('data/semisupervised/competition_data/unlabelled.csv')
uq = dict(zip(unl.trace_id, unl.question_id))
Xc, yc, qc, tc = [], [], [], []
for l in open('runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') not in (None, 'NONE') and a.get('reasoning'):
        Xc.append(strip(a['reasoning'])); yc.append(a['first_code']); qc.append(uq.get(r['trace_id'])); tc.append(r['trace_id'])
cl = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open('runs/claude/dev_opus.jsonl')}
Xd = [strip((cl.get(t) or {}).get('reasoning', '')) for t in dev.trace_id]

def fit_map(X, Y, w, Xq):
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                     TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150000))
    clf = LogisticRegression(C=4, max_iter=3000, class_weight='balanced').fit(vec.fit_transform(X), Y, sample_weight=w)
    p = clf.predict_proba(vec.transform(Xq)); P = np.full((len(Xq), len(err_codes)), 1e-6)
    for j, c in enumerate(clf.classes_): P[:, err_codes.index(c)] = p[:, j]
    return P / P.sum(1, keepdims=True)

t0 = time.time()
P_dev = fit_map(Xf + Xc, yf + yc, [3.] * len(Xf) + [1.] * len(Xc), Xd)
print('full mapper', round(time.time() - t0), 's', flush=True)

devq = set(dev.question_id)
pool = np.array([i for i, q in enumerate(qc) if q in devq])
rng = np.random.default_rng(0); half = rng.permutation(len(pool)) % 2
P_pool = np.zeros((len(pool), len(err_codes)))
for h in (0, 1):
    held = set(pool[half == h].tolist())
    keep = [i for i in range(len(Xc)) if i not in held]
    Pq = fit_map(Xf + [Xc[i] for i in keep], yf + [yc[i] for i in keep], [3.] * len(Xf) + [1.] * len(keep),
                 [Xc[i] for i in pool[half == h]])
    P_pool[half == h] = Pq
    print('oof half', h, round(time.time() - t0), 's', flush=True)
np.savez(OUT + 't1_mapper_cache.npz', P_dev=P_dev, P_pool=P_pool,
         pool_q=np.array([qc[i] for i in pool]), pool_code=np.array([yc[i] for i in pool]),
         pool_trace=np.array([tc[i] for i in pool]), err_codes=np.array(err_codes), dev_ids=dev.trace_id.values)
acc = float(np.mean(np.array(err_codes)[P_pool.argmax(1)] == np.array([yc[i] for i in pool])))
json.dump({'n_claude_texts': len(Xc), 'n_gold_texts': len(Xf), 'pool_size': int(len(pool)),
           'pool_oof_top1_agreement_with_claude_code': round(acc, 4), 'seconds': round(time.time() - t0)},
          open(OUT + 't1_mapper.json', 'w'), indent=1)
print('done', len(pool), acc)
