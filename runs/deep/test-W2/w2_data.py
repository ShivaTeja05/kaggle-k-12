"""W2 kill test, data prep (CPU only). Builds a CONTENT-BEARING candidate proxy dataset from data/v5.

Each train row (data/v5/cand_train.jsonl) and dev row (data/v5/cand_dev.jsonl, order 1) has 10 candidate codes.
Per (row, candidate) pair features:
  prior  : seen-for-question, log1p pair count, 'On this question' example cue, NONE, log global freq
           (identical to runs/deep/approach-W2/w2_probe.py; train counts are leave-own-row-out over fit gold + Claude
           first codes, dev counts are fit gold + Claude non-dev rows)
  content: cosine(char TF-IDF of the student working, char TF-IDF of the candidate's same-question example span)
Per row content vector phi (for a code-specific linear head W[code] . phi):
  char_wb 2-5 TF-IDF of the student working (L-prefixes stripped) + word 1-2 TF-IDF of the question text + bias.
Also stores the error-conditional LOO question prior per letter (for the secondary A3 factorised-PoE arm),
the A1 group flag (error row whose truth is unseen for its question, LOO), and dev slices.
Writes runs/deep/test-W2/cache.pkl.
"""
import json, re, ast, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/shortcut')
from sc_common import load, claude_unl, slices, ROOT

OUT = ROOT / 'runs/deep/test-W2'
L = 'ABCDEFGHIJ'
tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
cl, unl = claude_unl(); cl_ok = cl[~cl.trace_id.isin(set(dev.trace_id))]
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
qcount = defaultdict(Counter)
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount[q][c] += 1
g = Counter(fit.root_code) + Counter(cl_ok.root_code); tot = sum(g.values())
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))
noL = lambda t: re.sub(r'(?m)^\s*L\d+\s*\|\s?', '', str(t))
EXK = ' On this question, e.g. "'


def parse(prompt):
    q = re.search(r'\nQuestion: (.*?)\nAnswer key:', prompt, re.S).group(1)
    w = prompt.split('Student working (line by line):\n', 1)[1].split('\n\nCandidate error codes:', 1)[0]
    cands = {}
    for ln in prompt.split('\n'):
        if len(ln) > 2 and ln[1] == '.' and ln[0] in L:
            ex = ln.split(EXK, 1)[1].rstrip('"') if EXK in ln else ''
            cands[ln[0]] = dict(ex=ex, has_ex=EXK in ln)
    return q, noL(w), cands


def prior_feats(codes, cnt, cands):
    F = []
    for i, c in enumerate(codes):
        n = max(cnt.get(c, 0), 0); none = c == 'NONE'
        F.append([0 if none else float(n > 0), 0 if none else np.log1p(n), float(cands.get(L[i], {}).get('has_ex', False)),
                  float(none), 0 if none else np.log(g[c] / tot + 1e-6)])
    return np.array(F)


def err_lp(codes, cnt):  # same as runs/deep/approach-W2/w2_probe2.py err_lp: centred log p(code | q, error), NONE = 0
    ne = sum(v for k, v in cnt.items() if v > 0 and k != 'NONE'); ge = tot - g['NONE']
    lp = np.array([0. if c == 'NONE' else np.log((max(cnt.get(c, 0), 0) + 2 * (g[c] + 1) / (ge + 76)) / (ne + 2)) for c in codes])
    m = np.array([c != 'NONE' for c in codes]); lp[m] -= lp[m].mean(); return lp


T = []
for line in open(ROOT / 'data/v5/cand_train.jsonl'):
    x = json.loads(line); codes = x['codes']; truth = codes[L.index(x['target'][-1])]
    q = qmap[x['trace_id']]; cnt = Counter(qcount[q]); cnt[own.get(x['trace_id'])] -= 1
    qt, wk, cands = parse(x['prompt'])
    soft = np.zeros(10)
    for k, v in x['soft'].items(): soft[L.index(k)] = v
    soft /= soft.sum()
    T.append(dict(tid=x['trace_id'], q=q, src=x['source'], truth=truth, codes=codes, qt=qt, wk=wk,
                  ex=[cands.get(L[i], {}).get('ex', '') for i in range(10)], Xp=prior_feats(codes, cnt, cands), y=soft,
                  w0=x['weight'], newp=truth != 'NONE' and cnt.get(truth, 0) <= 0, lp=err_lp(codes, cnt)))
S, _ = slices(dev, fit)
D = []
for line, s, q, t, tid in zip(open(ROOT / 'data/v5/cand_dev.jsonl'), S, dev.question_id, dev.root_code, dev.trace_id):
    x = json.loads(line); assert x['trace_id'] == tid
    P = ast.literal_eval(x['prompts']) if isinstance(x['prompts'], str) else x['prompts']
    CP = ast.literal_eval(x['codes_per_prompt']) if isinstance(x['codes_per_prompt'], str) else x['codes_per_prompt']
    codes = CP[0]; qt, wk, cands = parse(P[0])
    D.append(dict(tid=tid, q=q, s=s, truth=t, codes=codes, qt=qt, wk=wk, ex=[cands.get(L[i], {}).get('ex', '') for i in range(10)],
                  Xp=prior_feats(codes, qcount[q], cands), lp=err_lp(codes, qcount[q])))

vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=30000, sublinear_tf=True, lowercase=True)
vq = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True, lowercase=True)
vc.fit([x['wk'] for x in T]); vq.fit([x['qt'] for x in T])


def phi(rows):
    W = vc.transform([x['wk'] for x in rows]); Q = vq.transform([x['qt'] for x in rows])
    return W, Q


def ex_sim(rows, Wk):
    exs = [e for x in rows for e in x['ex']]
    E = vc.transform([e if e else ' ' for e in exs])
    out = np.zeros((len(rows), 10))
    for i in range(len(rows)):
        out[i] = (E[i * 10:(i + 1) * 10] @ Wk[i].T).toarray().ravel()
        for k in range(10):
            if not rows[i]['ex'][k]: out[i, k] = 0.
    return out


Wt, Qt = phi(T); Wd, Qd = phi(D)
st, sd = ex_sim(T, Wt), ex_sim(D, Wd)
cache = dict(classes=classes, slices=S, dev_q=dev.question_id.to_numpy(), dev_y=dev.root_code.to_numpy(), dev_ids=dev.trace_id.tolist(),
             fit_codes=fit.root_code.tolist(),
             tr=dict(Xp=np.stack([x['Xp'] for x in T]), sim=st, y=np.stack([x['y'] for x in T]), w0=np.array([x['w0'] for x in T]),
                     newp=np.array([x['newp'] for x in T]), err=np.array([x['truth'] != 'NONE' for x in T]),
                     cidx=np.array([[ci[c] for c in x['codes']] for x in T]), lp=np.stack([x['lp'] for x in T]),
                     W=Wt.tocsr(), Q=Qt.tocsr(), q=np.array([x['q'] for x in T]), src=np.array([x['src'] for x in T]),
                     truth=np.array([x['truth'] for x in T])),
             dv=dict(Xp=np.stack([x['Xp'] for x in D]), sim=sd, cidx=np.array([[ci[c] for c in x['codes']] for x in D]),
                     lp=np.stack([x['lp'] for x in D]), W=Wd.tocsr(), Q=Qd.tocsr(),
                     truth_in=np.array([x['truth'] in x['codes'] for x in D])))
pickle.dump(cache, open(OUT / 'cache.pkl', 'wb'))
tr_ = cache['tr']
print('train rows', len(T), 'dev rows', len(D), 'W dims', Wt.shape[1], 'Q dims', Qt.shape[1])
print('train error rows', tr_['err'].sum(), 'LOO new-pair error rows', tr_['newp'].sum(),
      'error weight share', round(tr_['w0'][tr_['err']].sum() / tr_['w0'].sum(), 4))
print('dev truth in candidates', cache['dv']['truth_in'].mean().round(4), Counter(S))
print('example sim mean (train, rows with example)', st[st > 0].mean().round(3), 'dev', sd[sd > 0].mean().round(3))
