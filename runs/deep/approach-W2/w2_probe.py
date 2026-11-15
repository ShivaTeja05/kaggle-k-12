"""W2 CPU probe (numpy only). A 'prior-only' conditional-logit learner over each row's 10 candidates, using only
question-prior features (seen-for-question, log pair count, same-question example cue, NONE, log global code freq).
It measures how strongly a training mix REWARDS the question shortcut under different row weightings / a PoE offset,
compared with the strength that is actually optimal on dev (fitted on dev, question-grouped 5-fold cross-fit).
Also counts natural counterfactual pairs (same question, different error truth) in data/v5 for the paired-candidate idea.
Limitation: no content features, so dev MAP here is a pure-prior proxy, not a Stage B forecast."""
import json, ast, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/shortcut')
from sc_common import load, claude_unl, slices, ROOT, r
from common import boot_ci
OUT = ROOT / 'runs/deep/approach-W2'
tr, tax, classes, dev, fit = load()
cl, unl = claude_unl(); cl_ok = cl[~cl.trace_id.isin(set(dev.trace_id))]
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
qcount = defaultdict(Counter)
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount[q][c] += 1
g = Counter(fit.root_code) + Counter(cl_ok.root_code); tot = sum(g.values())
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))
L = 'ABCDEFGHIJ'

def cand_lines(prompt):
    out = {}
    for ln in prompt.split('\n'):
        if len(ln) > 2 and ln[1] == '.' and ln[0] in L:
            out[ln[0]] = ln
    return out

def feats(codes, cnt, prompt):
    cl_ = cand_lines(prompt); F = []
    for i, c in enumerate(codes):
        n = max(cnt.get(c, 0), 0); none = c == 'NONE'
        ex = 'On this question' in cl_.get(L[i], '')
        F.append([0 if none else float(n > 0), 0 if none else np.log1p(n), float(ex), float(none),
                  0 if none else np.log(g[c] / tot + 1e-6)])
    return np.array(F)

def logprior(codes, cnt):
    n = sum(v for v in cnt.values() if v > 0)
    return np.array([0. if c == 'NONE' else np.log((max(cnt.get(c, 0), 0) + 2 * g[c] / tot) / (n + 2)) for c in codes])

T = []
for line in open(ROOT / 'data/v5/cand_train.jsonl'):
    x = json.loads(line); codes = x['codes']; truth = codes[L.index(x['target'][-1])]
    q = qmap[x['trace_id']]; cnt = Counter(qcount[q]); cnt[own.get(x['trace_id'])] -= 1
    n = sum(v for v in cnt.values() if v > 0)
    p = (max(cnt.get(truth, 0), 0) + 2 * g[truth] / tot) / (n + 2)
    mode = max((k for k in cnt if cnt[k] > 0 and k != 'NONE'), key=lambda k: cnt[k], default=None)
    soft = np.zeros(10)
    for k, v in x['soft'].items(): soft[L.index(k)] = v
    T.append(dict(q=q, src=x['source'], truth=truth, codes=codes, X=feats(codes, cnt, x['prompt']), y=soft,
                  w0=x['weight'], p=p, pair=max(cnt.get(truth, 0), 0), mode=mode, lp=logprior(codes, cnt)))
S, _ = slices(dev, fit)
D = []
for line, s, q, t in zip(open(ROOT / 'data/v5/cand_dev.jsonl'), S, dev.question_id, dev.root_code):
    x = json.loads(line); P = ast.literal_eval(x['prompts']) if isinstance(x['prompts'], str) else x['prompts']
    CP = ast.literal_eval(x['codes_per_prompt']) if isinstance(x['codes_per_prompt'], str) else x['codes_per_prompt']
    cnt = qcount[q]; codes = CP[0]
    D.append(dict(q=q, s=s, truth=t, codes=codes, X=feats(codes, cnt, P[0]), lp=logprior(codes, cnt)))

def fit_cl(rows, wts, offset=0., iters=400, lr=.5, l2=1e-3):
    X = np.stack([r_['X'] for r_ in rows]); Y = np.stack([r_['y'] for r_ in rows]); w = np.asarray(wts, float)
    w = w / w.mean(); off = offset * np.stack([r_['lp'] for r_ in rows]) if offset else 0.
    th = np.zeros(X.shape[2]); m = np.zeros_like(th); v = np.zeros_like(th)
    for it in range(1, iters + 1):
        z = X @ th + off; z -= z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
        grad = np.einsum('n,nk,nkf->f', w, P - Y, X) / len(w) + l2 * th
        m = .9 * m + .1 * grad; v = .999 * v + .001 * grad ** 2
        th -= lr * .05 * (m / (1 - .9 ** it)) / (np.sqrt(v / (1 - .999 ** it)) + 1e-8)
    return th

def dev_scores(th, rows):
    out = []
    for r_ in rows:
        z = r_['X'] @ th; order = np.argsort(-z)[:3]; top = [r_['codes'][i] for i in order]
        out.append(next((1 / (k + 1) for k, c in enumerate(top) if c == r_['truth']), 0.))
    return np.array(out)

def summ(sc):
    return {k: r(sc[S == k].mean()) for k in ['none', 'seen_pair', 'new_pair', 'unseen_q']} | {'all': r(sc.mean())}

for r_ in D:
    y = np.zeros(10)
    if r_['truth'] in r_['codes']: y[r_['codes'].index(r_['truth'])] = 1
    else: y[:] = .1  # truth not reachable: uninformative target for the dev-optimal fit
    r_['y'] = y
FEATS = ['seen', 'log_pair', 'example_cue', 'none', 'log_global']
res = {'features': FEATS}
# dev-optimal prior strength: fit on dev, 5 question-grouped folds, scored out of fold
qs = np.array(sorted(set(dev.question_id))); rng = np.random.default_rng(0); rng.shuffle(qs)
fold = {q: i % 5 for i, q in enumerate(qs)}; F = np.array([fold[q] for q in dev.question_id])
oof = np.zeros(len(D)); ths = []
for k in range(5):
    th = fit_cl([D[i] for i in range(len(D)) if F[i] != k], np.ones((F != k).sum())); ths.append(th)
    oof[F == k] = dev_scores(th, [D[i] for i in range(len(D)) if F[i] == k])
res['dev_optimal'] = dict(theta=dict(zip(FEATS, np.round(np.mean(ths, 0), 3).tolist())), dev_map=summ(oof))
W = {
    'current': lambda x: x['w0'],
    'focal_g2': lambda x: x['w0'] * (1 if x['truth'] == 'NONE' else (1 - x['p']) ** 2),
    'focal_g4': lambda x: x['w0'] * (1 if x['truth'] == 'NONE' else (1 - x['p']) ** 4),
    'inv_sqrt_pair': lambda x: x['w0'] * (1 if x['truth'] == 'NONE' else 1 / np.sqrt(x['pair'] + 1)),
    'jtt_nonmodal_x2': lambda x: x['w0'] * (2 if x['truth'] != 'NONE' and x['truth'] != x['mode'] else 1),
    'focal_g4_plus_nonmodal_x2': lambda x: x['w0'] * (1 if x['truth'] == 'NONE' else (1 - x['p']) ** 4 * (2 if x['truth'] != x['mode'] else 1)),
}
base = None
for name, f in W.items():
    th = fit_cl(T, [f(x) for x in T]); sc = dev_scores(th, D)
    res[name] = dict(theta=dict(zip(FEATS, np.round(th, 3).tolist())), dev_map=summ(sc))
    if name == 'current': base = sc
    else:
        res[name]['delta_vs_current_newpair+unseenq'] = boot_ci(np.where(np.isin(S, ['new_pair', 'unseen_q']), sc, base), base, dev.question_id.values)
        res[name]['delta_vs_current_all'] = boot_ci(sc, base, dev.question_id.values)
for lam in [.5, 1.]:
    th = fit_cl(T, [x['w0'] for x in T], offset=lam); sc = dev_scores(th, D)
    res[f'poe_offset_lambda{lam}'] = dict(theta=dict(zip(FEATS, np.round(th, 3).tolist())), dev_map=summ(sc),
                                          delta_vs_current_all=boot_ci(sc, base, dev.question_id.values))
# natural counterfactual pairs (approach 3)
byq = defaultdict(list)
for i, x in enumerate(T):
    if x['truth'] != 'NONE': byq[x['q']].append(i)
has_cf = 0; in_cands = 0; injectable = 0; nerr = 0
for q, idx in byq.items():
    truths = Counter(T[i]['truth'] for i in idx)
    for i in idx:
        nerr += 1; others = set(truths) - {T[i]['truth']}
        if others:
            has_cf += 1
            if others & set(T[i]['codes']): in_cands += 1
            else: injectable += 1
res['counterfactual_pairs'] = dict(error_rows=nerr, rows_with_same_question_other_truth=has_cf,
                                   share=r(has_cf / nerr), other_truth_already_in_candidates=in_cands,
                                   needs_injection=injectable, questions=len(byq))
def tojs(o):
    if isinstance(o, dict): return {k: tojs(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [tojs(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(tojs(res), open(OUT / 'w2_probe.json', 'w'), indent=1); print(json.dumps(tojs(res), indent=1))
