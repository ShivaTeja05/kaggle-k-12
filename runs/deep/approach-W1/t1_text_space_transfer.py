"""T1: does a code's meaning transfer across questions once the error is written as text?
Same linear learner (TF-IDF + multinomial LR, no question text), three inputs:
  work  : the student's working (gold transcript for fit, machine transcript for dev) -> the current members' input
  gtext : the gold annotator's root-misconception text (oracle diagnosis)
  creason: Claude's reasoning (pool-trained mapper, applied to Claude dev reasoning)
Learning curve on dev error rows binned by fit count of the (question, gold code) pair. Flat curve = transfer."""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, boot_ci, ROOT
from k12.core import row_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

D = get_all(); dev, fit, y, classes = D['dev'], D['fit'], D['y'], D['classes']
OUT = ROOT / 'runs/deep/approach-W1'
pair = Counter(zip(fit.question_id, fit.root_code)); qseen = set(fit.question_id)
def binof(q, c):
    if c == 'NONE': return 'NONE'
    if q not in qseen: return 'q_unseen'
    k = pair[(q, c)]; return 'pair0' if k == 0 else 'pair1' if k == 1 else 'pair2' if k == 2 else 'pair3+'
bins = np.array([binof(q, c) for q, c in zip(dev.question_id, dev.root_code)])

def root_text(m):
    try: ms = json.loads(m)
    except Exception: return ''
    r = [x for x in ms if x.get('root')]
    return ' '.join(f"{x.get('span','')} . {x.get('text','')}" for x in r)
def tx(s): return ' '.join(l.split('|', 1)[-1] for l in str(s).splitlines())

class Mapper:
    def __init__(self, C=10.):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
        self.lr = LogisticRegression(C=C, max_iter=3000)
    def fit(self, X, yy, sw=None):
        self.lr.fit(hstack([self.w.fit_transform(X), self.c.fit_transform(X)]).tocsr(), yy, sample_weight=sw); return self
    def top3(self, X):
        p = self.lr.predict_proba(hstack([self.w.transform(X), self.c.transform(X)]).tocsr())
        cl = self.lr.classes_; return np.array([cl[np.argsort(-r)[:3]] for r in p])

res = {}
err = y != 'NONE'
def report(name, pred, mask):
    s = row_scores(y[mask], pred) if len(pred) == mask.sum() else None
    full = np.zeros(len(y)); full[mask] = s
    out = {b: [round(float(full[(bins == b) & mask].mean()), 3), int(((bins == b) & mask).sum())]
           for b in ['q_unseen', 'pair0', 'pair1', 'pair2', 'pair3+', 'NONE'] if ((bins == b) & mask).any()}
    res[name] = out; print(name, out); return full

# references on error rows
for k in ['blend_sk', 'claude']:
    report(k, D['preds'][k][err], err)

# A. gold diagnosis text -> code, trained on fit error rows only (no question, no working)
fe = fit[fit.root_code != 'NONE']; Xf = [root_text(m) for m in fe.misconceptions]
keep = [i for i, x in enumerate(Xf) if x.strip()]
mg = Mapper().fit([Xf[i] for i in keep], fe.root_code.to_numpy()[keep])
Xd = [root_text(m) for m in dev.misconceptions]
s_gtext = report('gtext_mapper(fit794)', mg.top3([Xd[i] for i in np.where(err)[0]]), err)

# B. same learner on the working (what our members see), fit error rows, no question text
mw = Mapper().fit([tx(s) for s in fe.transcription_steps], fe.root_code.to_numpy())
txd = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))
def mtx(t):
    v = txd.get(t, '');
    return tx(v if isinstance(v, str) else v.get('transcription', '') if isinstance(v, dict) else '\n'.join(v))
s_work = report('work_mapper(fit794,machine tx)', mw.top3([mtx(t) for t in dev.trace_id[err]]), err)
mw2 = Mapper().fit([tx(s) for s in fe.transcription_steps], fe.root_code.to_numpy())
s_workg = report('work_mapper(fit794,gold tx)', mw2.top3([tx(s) for s in dev.transcription_steps[err]]), err)

# C. Claude reasoning -> code, mapper trained on the Claude pool (reasoning only), applied to Claude dev reasoning
pool = [json.loads(l) for l in open(ROOT / 'runs/claude/unl_all_v4.jsonl')]
pool = [(r['answer'].get('reasoning', ''), r['answer'].get('first_code')) for r in pool if (r.get('answer') or {}).get('first_code') in classes]
pool = [p for p in pool if p[1] != 'NONE' and p[0]]
mc = Mapper(C=5.).fit([p[0] for p in pool], np.array([p[1] for p in pool]))
cd = {json.loads(l)['trace_id']: json.loads(l)['answer'].get('reasoning', '') for l in open(ROOT / 'runs/claude/dev_opus.jsonl')}
s_cr = report('creason_mapper(pool error rows)', mc.top3([cd.get(t, '') for t in dev.trace_id[err]]), err)
# same mapper, gold-text input (cross-style)
s_cg = report('pool_mapper_on_gold_text', mc.top3([Xd[i] for i in np.where(err)[0]]), err)

bs = row_scores(y, D['preds']['blend_sk']); cs = row_scores(y, D['preds']['claude'])
g = dev.question_id.to_numpy()
cis = {}
for nm, s in [('gtext', s_gtext), ('work_machine', s_work), ('creason', s_cr)]:
    for b in ['pair0', 'q_unseen']:
        m = bins == b
        cis[f'{nm}-blend_sk@{b}'] = boot_ci(s[m], bs[m], g[m])
        cis[f'{nm}-claude@{b}'] = boot_ci(s[m], cs[m], g[m])
    m = np.isin(bins, ['pair0', 'q_unseen']); n = np.isin(bins, ['pair1', 'pair2', 'pair3+'])
    cis[f'{nm}:new_minus_seen'] = [round(float(s[m].mean() - s[n].mean()), 3)]
cis['blend_sk:new_minus_seen'] = [round(float(bs[m].mean() - bs[n].mean()), 3)]
cis['claude:new_minus_seen'] = [round(float(cs[m].mean() - cs[n].mean()), 3)]
for k, v in cis.items(): print(k, v)
json.dump({'bins_by_model': res, 'contrasts': cis, 'n_pool_error_rows': len(pool), 'n_fit_text_rows': len(keep)},
          open(OUT / 't1_text_space_transfer.json', 'w'), indent=1)
