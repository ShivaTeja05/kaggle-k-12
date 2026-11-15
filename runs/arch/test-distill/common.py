"""Shared setup for the adversarial R1D CPU test. Builds the rationale decoder (TF-IDF LR on 26k Claude unlabelled
rationales, code ids stripped) x taxonomy-description similarity, the 0.718 reference blend, bins, and a nested cross-fit
fusion evaluator (desc weight AND fusion weight both chosen inside the training folds)."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.sparse import hstack
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
OUT = ROOT / 'runs/arch/test-distill'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from k12.core import read_csv, label_space, make_folds, aligned_bundle, row_scores, topk, normalize
from evaluate_claude import claude_scores

tax = read_csv(ROOT/'data/raw/taxonomy_v1.csv'); classes = label_space(tax); ci = {c: i for i, c in enumerate(classes)}
train = read_csv(ROOT/'data/raw/train.csv'); fold = make_folds(train)
dev = train[fold == 0].reset_index(drop=True); fit = train[fold != 0]
ids = dev.trace_id.tolist(); y = dev.root_code.tolist(); yi = np.array([ci[c] for c in y]); qid = dev.question_id.values
CODE_RE = re.compile(r'\b(?:[A-Z][A-Za-z]{1,3}-\d{2}|NONE)\b'); cl = lambda t: CODE_RE.sub(' ', t or '')
unl = read_csv(ROOT/'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
U = {}
for l in open(ROOT/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') in ci and r['trace_id'] in unl.index: U[r['trace_id']] = a
U_ids = list(U); U_q = np.array([unl.loc[t, 'question_id'] for t in U_ids])
U_txt = [cl(U[t].get('reasoning')) for t in U_ids]; U_y = np.array([ci[U[t]['first_code']] for t in U_ids])
D = {json.loads(l)['trace_id']: json.loads(l).get('answer') or {} for l in open(ROOT/'runs/claude/dev_opus.jsonl')}
claude_first = np.array([ci.get(D.get(t, {}).get('first_code'), -1) for t in ids])
dev_txt = [cl(D.get(t, {}).get('reasoning')) for t in ids]

fc = fit.groupby('question_id').root_code.apply(set).to_dict()
bins = np.array(['NONE' if r.root_code == 'NONE' else 'q_unseen' if r.question_id not in fc else
                 'code_seen_for_q' if r.root_code in fc[r.question_id] else 'code_unseen_for_q' for r in dev.itertuples()])
unseen = np.isin(bins, ['code_unseen_for_q', 'q_unseen'])
P = [aligned_bundle(ROOT/b, ids, classes) for b in ['runs/pod/text32/validation_tx_dev_epoch_1.npz',
     'runs/pod/tx/text_tx_dev_e3.npz', 'runs/pod/dev-distill/validation_epoch_2.npz']]
blend = normalize(np.exp(sum(w*np.log(p+1e-9) for w, p in zip((.35, .30, .35), P))))

def rs(p, idx=None):
    """Row MAP@3 scores, vectorised (stable argsort like topk)."""
    t = yi if idx is None else yi[idx]
    if p.shape[0] != len(t): p = p[idx]
    o = np.argsort(-p, axis=1, kind='stable')[:, :3]
    return ((o == t[:, None]) / np.array([1., 2., 3.])).sum(1)
sb = rs(blend)
assert abs(sb.mean() - row_scores(y, topk(blend, classes)).mean()) < 1e-12

class Decoder:
    def __init__(self, texts, labels):
        self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
        self.vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
        X = hstack([self.vw.fit_transform(texts), self.vc.fit_transform(texts)]).tocsr()
        self.lr = LogisticRegression(C=8.0, max_iter=300).fit(X, labels)
    def lr_proba(self, texts):
        X = hstack([self.vw.transform(texts), self.vc.transform(texts)]).tocsr()
        p = np.full((len(texts), len(classes)), 1e-6); p[:, self.lr.classes_] += self.lr.predict_proba(X); return normalize(p)

desc = {r.id: f"{r.name}. {r.description}" for r in tax.itertuples()}
desc['NONE'] = 'No error. The working is correct and matches the answer key. No major error.'
DOCS = [desc[c] for c in classes]
VD = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english').fit(DOCS + U_txt[:20000])
DOCM = VD.transform(DOCS)
def desc_sim(texts): return (VD.transform(texts) @ DOCM.T).toarray()

uq = np.unique(qid)
_rng = np.random.default_rng(1)
BOOTS = [np.concatenate([np.flatnonzero(qid == q) for q in _rng.choice(uq, len(uq))]) for _ in range(4000)]
def ci_(a, b):
    d = np.array([a[i].mean() - b[i].mean() for i in BOOTS]); return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
def pb(s): return {b: round(float(s[bins == b].mean()), 4) for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen', 'NONE']} | {'unseen_pair_106': round(float(s[unseen].mean()), 4)}

WGRID = np.linspace(0, 1, 21); DGRID = [0, 10, 20, 40]
def fuse_crossfit(plr, S, base=blend, seeds=3, dgrid=DGRID):
    """Nested cross-fit: (desc weight, fusion weight) picked on 4 question folds, scored on the 5th; 5 folds x seeds."""
    stages = {d: normalize(np.exp(np.log(plr+1e-9) + d*S)) for d in dgrid}
    lb = np.log(base+1e-9); ls = {d: np.log(stages[d]+1e-9) for d in dgrid}
    outs, picks = [], []
    for seed in range(seeds):
        r2 = np.random.default_rng(100+seed); qf = {q: i % 5 for i, q in enumerate(r2.permutation(uq))}; df = np.array([qf[q] for q in qid])
        cf = np.zeros(len(ids))
        for k in range(5):
            tr = np.flatnonzero(df != k); te = np.flatnonzero(df == k)
            best = max(((d, w) for d in dgrid for w in WGRID), key=lambda dw: rs(np.exp((1-dw[1])*lb[tr] + dw[1]*ls[dw[0]][tr]), tr).mean())
            picks.append([best[0], round(float(best[1]), 2)])
            cf[te] = rs(np.exp((1-best[1])*lb[te] + best[1]*ls[best[0]][te]), te)
        outs.append(cf)
    cf = np.mean(outs, 0)
    st = stages[20 if 20 in stages else dgrid[0]]  # single-stage score reported at fixed desc weight 20 (not tuned)
    return {'stage_single_dw20': round(float(rs(st).mean()), 4), 'stage_top1_agree_claude': round(float((st.argmax(1) == claude_first).mean()), 4),
            'fused_crossfit': round(float(cf.mean()), 4), 'delta_vs_0.7182': round(float(cf.mean() - sb.mean()), 4), 'ci': ci_(cf, sb),
            'bins': pb(cf), 'picks_median_w': float(np.median([p[1] for p in picks]))}, cf

import joblib
def cached_decoder(name, texts, labels):
    f = OUT/f'decoder_{name}.joblib'
    if f.exists(): return joblib.load(f)
    d = Decoder(texts, labels); joblib.dump(d, f); return d
