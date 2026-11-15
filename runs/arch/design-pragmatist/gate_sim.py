"""What single-model quality must a top-10 candidate reasoner reach for blend fusion to hit 0.75/0.78/0.80?
Simulated student = Claude's ranking on a random fraction q of dev rows, the offline 8B scale_100 bundle elsewhere,
always restricted to the blend's top-10 candidates (+NONE). Fusion weight cross-fitted (5 question folds).
Oracle-style diagnostic: it uses Claude on dev, so it sets gate thresholds; it is not a deployable score."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
from evaluate_claude import claude_scores
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
S = aligned_bundle(ROOT/'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', ids, classes)
ci = {c: i for i, c in enumerate(classes)}; inone = ci['NONE']
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool)
mask[np.arange(len(B))[:, None], top10] = True; mask[:, inone] = True
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def sinkhorn(P, lam=.5, it=30):
    b = np.zeros(P.shape[1])
    for _ in range(it): Q = normalize(P*np.exp(b)); b += np.log(pi_fit/Q.mean(0))
    return normalize(P*np.exp(lam*b))
qs = np.unique(qid); rng0 = np.random.default_rng(7); rng0.shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
# bins from teacher-gap rows.csv
rows = pd.read_csv(ROOT/'runs/arch/teacher-gap/rows.csv').set_index('trace_id').loc[ids]
seen_q = set(fit.question_id); fitpairs = set(zip(fit.question_id, fit.root_code))
unseen_pair = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
W = [0, .2, .35, .5, .65, .8, 1.]
def fuse(St, w): 
    lp = (1-w)*np.log(B) + w*np.log(np.where(mask, St, 1e-6)); return normalize(np.exp(lp - lp.max(1, keepdims=True)))
def crossfit(St, sink=False):
    out = np.empty(len(y)); cache = {w: fuse(St, w) for w in W}
    if sink: cache = {w: sinkhorn(v) for w, v in cache.items()}
    pts = {w: row_scores(y, topk(cache[w], classes)) for w in W}
    for f in range(5):
        m = mf != f; w = max(W, key=lambda w: pts[w][m].mean()); out[~m] = pts[w][~m]
    return out
res = {'ref': float(ref.mean()), 'ref_unseen_pair': float(ref[unseen_pair].mean()), 'n_unseen_pair': int(unseen_pair.sum())}
res['ref_sinkhorn_fixed0.5'] = float(row_scores(y, topk(sinkhorn(B), classes)).mean())
for q in [0, .25, .5, .75, 1.]:
    acc = []
    for seed in range(5):
        rng = np.random.default_rng(seed); use = rng.random(len(y)) < q
        St = normalize(np.where(use[:, None], C, S) * mask + 1e-9)
        alone = row_scores(y, topk(St, classes)); cf = crossfit(St); cfs = crossfit(St, True)
        acc.append([alone.mean(), alone[unseen_pair].mean(), cf.mean(), cf[unseen_pair].mean(), cfs.mean(), boot_ci(cfs, ref, qid)])
    a = np.array([r[:5] for r in acc]); res[f'q={q}'] = dict(student_alone=a[:, 0].mean(), student_unseen_pair=a[:, 1].mean(),
        fused_cf=a[:, 2].mean(), fused_cf_unseen_pair=a[:, 3].mean(), fused_cf_sinkhorn=a[:, 4].mean(), ci_seed0_sinkhorn=acc[0][5])
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(ROOT/'runs/arch/design-pragmatist/gate_sim.json', 'w'), indent=1, default=float)
