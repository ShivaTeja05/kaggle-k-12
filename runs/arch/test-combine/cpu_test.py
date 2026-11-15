"""Adversarial CPU test of the rank-then-verify cascade proposal (runs/arch/design-combine).
No verifier checkpoint exists yet, so the proposal's cpu_test is run on (a) the closest existing *distilled* students
(8B text reasoners trained with 0/25/100% Claude-labelled rows: runs/pod/scaling/*) and other dev bundles used as
top-10 verifiers on the Sinkhorn-balanced blend, and (b) harder oracle simulations than verifier_sim.py
(correlated 'which rows the student gets right', and confidently-wrong instead of uninformative elsewhere).
All tuning (Sinkhorn lambda, fusion w) is cross-fitted jointly on 5 question-grouped folds inside dev (301 rows).
CIs: paired question-cluster bootstrap vs the 0.718 reference blend. Claude dev answers are used only as an oracle."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy(); n = len(y)
ci = {c: i for i, c in enumerate(classes)}; yi = np.array([ci[c] for c in y])
P = blend(ids, classes); ref = row_scores(y, topk(P, classes))
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
c1 = C.argmax(1)
fitcodes = fit.groupby('question_id').root_code.agg(set).to_dict()
up = np.array([(t != 'NONE') and (t not in fitcodes.get(q, set())) for t, q in zip(y, qid)])  # unseen-pair rows
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
qs = np.unique(qid).copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])

def sinkhorn(M, lam, iters=30):
    b = np.zeros(M.shape[1])
    for _ in range(iters):
        Q = normalize(M * np.exp(b)); b += np.log(pi_fit / Q.mean(0))
    return normalize(M * np.exp(lam * b))
def geo(ms, ws):
    lp = sum(w*np.log(m) for m, w in zip(ms, ws)); return normalize(np.exp(lp - lp.max(1, keepdims=True)))
def sc(M): return row_scores(y, topk(M, classes))
def recall(M, K): return float(np.mean([yi[i] in np.argsort(-M[i])[:K] for i in range(n)]))
LAMS = [0, .25, .5, .75]; WS = [0, .25, .5, 1, 2, 4]
BASE = {l: sinkhorn(P, l) for l in LAMS}
TOP10 = {l: np.argsort(-BASE[l], 1)[:, :10] for l in LAMS}

def restrict(V, top, floor=1e-6):
    R = np.full_like(V, floor)
    for i in range(n): R[i, top[i]] = V[i, top[i]] + 1e-4
    return normalize(R)
def crossfit(pts):  # pts: dict key -> per-row points; choose key on 4/5 question folds, score the 5th
    keys = list(pts); out = np.empty(n); ch = []
    for f in range(5):
        m = mf != f; k = max(keys, key=lambda k: pts[k][m].mean()); ch.append(k); out[~m] = pts[k][~m]
    d, c = boot_ci(out, ref, qid)
    return out, dict(map3=float(out.mean()), delta=d, ci95=c, choices=[str(x) for x in ch],
                     unseen_pair=float(out[up].mean()), seen_or_none=float(out[~up].mean()))
def fuse_pts(Vfull, lams=LAMS, ws=WS, gate_frac=None):
    pts = {}
    for l in lams:
        V10 = restrict(Vfull, TOP10[l])
        for w in ws:
            M = geo([BASE[l], V10], [1, w])
            if gate_frac is not None:
                conf = BASE[l].max(1); gate = conf <= np.quantile(conf, gate_frac); M = np.where(gate[:, None], M, BASE[l])
            pts[(l, w)] = sc(M)
    return pts

res = {'ref_map3': float(ref.mean()), 'n_dev': n, 'n_unseen_pair': int(up.sum()),
       'ref_unseen_pair': float(ref[up].mean()), 'claude_alone': float(sc(C).mean()),
       'claude_unseen_pair': float(sc(C)[up].mean()),
       'recall10': {f'lam{l}': recall(BASE[l], 10) for l in LAMS}, 'recall5_lam0': recall(P, 5)}
# Stage 0 alone (Sinkhorn lambda cross-fitted)
s0_out, res['stage0_sinkhorn_crossfit'] = crossfit({(l, 0): sc(BASE[l]) for l in LAMS})

# ---- oracle curve on the Sinkhorn base (reproduce + adversarial variants) ----
def oracle_V(know, wrong_mode, top, seed):
    """know: bool mask of rows where verifier = Claude. Elsewhere: 'uniform' (proposal) or 'confident_wrong':
    a Claude-shaped .6/.25/.15 distribution on 3 random candidates, never Claude's top-1."""
    rng = np.random.default_rng(seed); V = np.full((n, len(classes)), 1e-3)
    for i in range(n):
        cand = top[i]
        if know[i]: V[i, cand] = C[i, cand]
        elif wrong_mode == 'uniform': V[i, cand] = 1 / len(cand)
        else:
            pool = [c for c in cand if c != c1[i]]; pick = rng.choice(pool, 3, replace=False)
            V[i, cand] = 1e-3; V[i, pick] = [.6, .25, .15]
    return normalize(V)
def oracle_fused(r, mask_mode, wrong_mode, seeds=6):
    vals, upv = [], []
    for s in range(seeds):
        rng = np.random.default_rng(100 + s); pts = {}
        if mask_mode == 'random': know = rng.random(n) < r
        else:
            # correlated: verifier knows Claude preferentially where the blend already agrees with Claude ('easy_first')
            agree = (P.argmax(1) == c1).astype(float)
            key = (agree if mask_mode == 'easy_first' else 1 - agree) + rng.random(n) * 1e-3
            know = np.zeros(n, bool); know[np.argsort(-key)[:int(round(r * n))]] = True
        for l in LAMS:
            V = oracle_V(know, wrong_mode, TOP10[l], s)
            for w in WS: pts[(l, w)] = sc(geo([BASE[l], V], [1, w]))
        o, d = crossfit(pts); vals.append(d['map3']); upv.append(d['unseen_pair'])
        if mask_mode != 'random' and wrong_mode == 'uniform': break
    return float(np.mean(vals)), float(np.mean(upv))
curves = {}
for mm in ['random', 'easy_first', 'hard_first']:
    for wm in ['uniform', 'confident_wrong']:
        curves[f'{mm}|{wm}'] = {str(r): oracle_fused(r, mm, wm) for r in [0.25, 0.5, 0.75, 1.0]}
res['oracle_curves_sinkhorn_base'] = curves
res['blend_top1_agrees_claude'] = float(np.mean(P.argmax(1) == c1))
ref_curve = {float(k): v[0] for k, v in curves['random|uniform'].items()}
def implied_r(x):
    xs = [0.0] + sorted(ref_curve); ys = [res['stage0_sinkhorn_crossfit']['map3']] + [ref_curve[k] for k in sorted(ref_curve)]
    if x <= ys[0]: return 0.0
    return float(np.interp(x, ys, xs))

# ---- proxy verifiers: the proposal's cpu_test on the closest existing distilled students ----
proxies = {
 'scale0_tx (8B, 0% Claude rows)': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
 'scale25_tx (8B, 25% Claude rows)': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
 'scale100_tx (8B, 100% Claude rows)': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
 'scale100_gold (8B, 100% Claude, gold transcript)': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz',
 'text32_gold': 'runs/pod/text32/validation_gold_epoch_1.npz',
 'fewshot235': 'runs/pod/probe235/fewshot235/validation.npz',
 'dev-v3 VL': 'runs/pod/dev-v3/validation_epoch_3.npz',
 'dev-read VL': 'runs/pod/dev-read/validation_epoch_3.npz',
 'claude_oracle (teacher, not a student)': None,
}
pr = {}
for name, path in proxies.items():
    V = C if path is None else aligned_bundle(ROOT/path, ids, classes)
    pts = fuse_pts(V); out, d = crossfit(pts)
    V10 = restrict(V, TOP10[.5])
    agree10 = float(np.mean(V10.argmax(1) == c1))
    d.update(alone=float(sc(V).mean()), alone_unseen_pair=float(sc(V)[up].mean()), alone_in_top10=float(sc(V10).mean()),
             top1_agree_claude_in_top10=agree10, implied_r=implied_r(d['map3']),
             kill_rule_triggered=bool(d['unseen_pair'] < .50 or implied_r(d['map3']) < .25))
    # cascade variant: verifier applied only to the lowest-confidence 60%
    _, dc = crossfit(fuse_pts(V, gate_frac=.6)); d['cascade60'] = {k: dc[k] for k in ('map3', 'delta', 'ci95', 'unseen_pair')}
    pr[name] = d
res['proxy_verifiers_on_sinkhorn_blend'] = pr
# Equal-strength check: if the student had scale100's own top-1 agreement with Claude but random placement, what does the oracle curve predict?
res['note_agreement_vs_curve'] = {k: dict(agree=v['top1_agree_claude_in_top10'], fused=v['map3'], implied_r=v['implied_r']) for k, v in pr.items()}
print(json.dumps(res, indent=1))
json.dump(res, open(ROOT/'runs/arch/test-combine/cpu_test.json', 'w'), indent=1)
