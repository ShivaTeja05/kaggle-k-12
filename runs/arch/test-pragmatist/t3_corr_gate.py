"""T3. Attack gate_sim: it assumes the student copies Claude on a RANDOM fraction q of rows. A real 8B student trained
on the same data and reading the same machine transcripts will be right mostly where the blend is already right.
Regimes for choosing the rows on which the simulated student equals Claude (else the s100 8B), always masked to blend top-10:
  random  - the proposal's assumption
  easy    - rows weighted exp(+3*blend_points): student's gains overlap the blend's strengths
  prior   - only rows where Claude's top-1 is already in s100's top-5 (reasoning can only re-rank within the 8B's prior)
  hard    - rows weighted exp(-3*blend_points) (optimistic, for symmetry)
Fusion weight AND Sinkhorn lambda are jointly cross-fitted (5 question folds). We read the curves at the proposal's
gate (alone >= 0.715 and unseen-pair >= 0.48) and compare with blend+Sinkhorn (cross-fitted)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; mf = qfolds(qid)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
S = aligned_bundle(ROOT/'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', ids, classes)
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
LAMS = [0, .25, .5, .75]; W = [0, .2, .35, .5, .65, .8]
bs = {l: row_scores(y, topk(sinkhorn(B, pi, l) if l else B, classes)) for l in LAMS}
bsink_cf, _ = crossfit(bs, mf)
z = ref
s5 = np.argsort(-S, 1)[:, :5]; c1 = C.argmax(1)
in_prior = np.array([c in t for c, t in zip(c1, s5)])
weights = {'random': np.ones(len(y)), 'easy': np.exp(3*z), 'hard': np.exp(-3*z), 'prior': in_prior.astype(float) + 1e-9}


def choose(wt, q, rng):
    n = int(round(q*len(y)))
    if n == 0: return np.zeros(len(y), bool)
    n = min(n, int((wt > 1e-6).sum()))
    sel = rng.choice(len(y), n, replace=False, p=wt/wt.sum()); m = np.zeros(len(y), bool); m[sel] = True
    return m


def evaluate(St):
    sc = {}
    for w in W:
        F = geo([(B, 1-w), (St, w)]) if w else B
        for l in LAMS:
            sc[(w, l)] = row_scores(y, topk(sinkhorn(F, pi, l) if l else F, classes))
    o, _ = crossfit(sc, mf)
    return o


res = {'ref': ref.mean(), 'blend_sink_cf': bsink_cf.mean(), 'prior_eligible_rows': int(in_prior.sum())}
QS = [0, .1, .2, .3, .4, .5, .6, .7, .8, 1.]
for reg, wt in weights.items():
    curve = []
    for q in QS:
        acc = []
        for seed in range(5):
            use = choose(wt, q, np.random.default_rng(seed))
            St = normalize(np.where(use[:, None], C, S)*mask + 1e-9)
            a = row_scores(y, topk(St, classes)); o = evaluate(St)
            acc.append((a.mean(), a[up].mean(), o.mean(), (o - bsink_cf).mean(), use.mean()))
        a = np.mean(acc, 0)
        curve.append(dict(q=q, q_eff=a[4], alone=a[0], unseen=a[1], fused_joint_cf=a[2], delta_vs_blend_sink_cf=a[3]))
        print(reg, curve[-1], flush=True)
    passing = [c for c in curve if c['alone'] >= .715 and c['unseen'] >= .48]
    res[reg] = dict(curve=curve, first_gate_pass=passing[0] if passing else None)
# CI at the first gate-passing point of each regime (seed 0)
for reg, wt in weights.items():
    fp = res[reg]['first_gate_pass']
    if fp:
        use = choose(wt, fp['q'], np.random.default_rng(0))
        St = normalize(np.where(use[:, None], C, S)*mask + 1e-9); o = evaluate(St)
        res[reg]['gate_point_seed0'] = dict(fused=o.mean(), ci_vs_ref=boot_ci(o, ref, qid), ci_vs_blend_sink_cf=boot_ci(o, bsink_cf, qid))
dump(res, 't3_corr_gate.json')
