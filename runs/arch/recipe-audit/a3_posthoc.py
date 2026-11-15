"""Cross-fitted post-hoc corrections on the reference blend: label-shift prior correction and question priors.

Every tuned knob is chosen on 4/5 of dev questions and applied to the held-out 1/5 (grouped by question),
then compared row-by-row with the fixed .35/.30/.35 reference (0.7182) via a question bootstrap.
"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/recipe-audit')
from common import *
from k12.baseline import HierarchicalPrior

ref = np.load(OUT / 'blend_ref.npy'); ref_pts = points(ref)
P = {n: load(n) for n in ('q32_tx', 'q8_tx_e3', 'vl_distill')}
pi_train = np.load(OUT / 'pi_train_mix.npy')
pi_gold = np.bincount([ci[c] for c in fit.root_code], minlength=len(classes)) + .5; pi_gold /= pi_gold.sum()

qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
meta = dev.question_id.map({q: i % 5 for i, q in enumerate(qs)}).to_numpy()

def crossfit(fn, grid):
    out = np.zeros(len(dev)); picks = []
    for f in range(5):
        a, b = meta != f, meta == f
        sc = [points(fn(g)[a], truth[a]).mean() for g in grid]
        g = grid[int(np.argmax(sc))]; picks.append(g)
        out[b] = points(fn(g)[b], truth[b])
    full = [round(float(points(fn(g)).mean()), 4) for g in grid]
    d = out - ref_pts
    return {'crossfit_map3': round(float(out.mean()), 4), 'delta': round(float(d.mean()), 4),
            'ci95': [round(x, 4) for x in question_bootstrap(d)], 'picks': picks, 'in_sample_grid': dict(zip(map(str, grid), full))}

R = {'reference_map3': round(float(ref_pts.mean()), 4)}
# 1. Label-shift correction toward the gold prior (text models were trained on the Claude-skewed mix).
shift = np.log(pi_gold) - np.log(pi_train.clip(1e-6))
R['prior_shift_on_blend'] = crossfit(lambda b: normalize(ref * np.exp(b * shift)), [0, .1, .2, .3, .5, .75, 1.])
# also toward uniform-over-errors (NONE kept): divide by train mix only
R['divide_train_prior'] = crossfit(lambda b: normalize(ref * np.exp(-b * np.log(pi_train.clip(1e-6)))), [0, .1, .2, .3, .5])
# 2. Question prior from gold fit rows (existing HierarchicalPrior), geometric
qp = HierarchicalPrior(classes).fit(fit).predict_proba(dev)
R['gold_question_prior_geometric'] = crossfit(lambda a: normalize(ref * qp ** a), [0, .05, .1, .2, .3, .5])
R['gold_question_prior_linear'] = crossfit(lambda w: (1 - w) * ref + w * qp, [0, .05, .1, .2, .3])
# 3. Question prior from Claude labels on unlabelled photos of the same question (error codes only + NONE rate)
unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv')
qid = dict(zip(unl.trace_id, unl.question_id))
cl = claude_rank(ROOT / 'runs/claude/unl_all_v4.jsonl')
byq = defaultdict(lambda: np.zeros(len(classes)))
for t, r in cl.items():
    if t in qid and r[0] in ci:
        for s, c in zip((.7, .2, .1), r):
            if c in ci: byq[qid[t]][ci[c]] += s
dq = dev.question_id.tolist()
R['dev_rows_whose_question_has_claude_rows'] = round(float(np.mean([q in byq for q in dq])), 3)
R['dev_median_claude_rows_per_question'] = float(np.median([byq[q].sum() for q in dq if q in byq]))
test = read_csv(ROOT / 'data/raw/test.csv')
R['test_rows_whose_question_has_claude_rows'] = round(float(np.mean([q in byq for q in test.question_id])), 3)
R['test_rows_whose_question_in_train'] = round(float(np.mean(test.question_id.isin(set(train.question_id)))), 3)
glob = np.bincount([ci[c] for c in fit.root_code], minlength=len(classes)) + .5
def claude_qprior(k):
    rows = []
    for q in dq:
        v = byq.get(q, np.zeros(len(classes))).copy()
        v[ci['NONE']] = 0  # NONE rate in the unlabelled pool (71%) differs from labelled data; use error shape only
        e = (v + k * glob / glob.sum() * (v.sum() > 0)) if v.sum() else glob.copy()
        e = e / e.sum()
        rows.append(e)
    return np.asarray(rows)
cq = claude_qprior(2.)
err_mass = lambda p: p  # apply only to error codes
def apply_cq(a):
    adj = ref.copy(); m = np.ones(len(classes), bool); m[ci['NONE']] = False
    adj[:, m] = ref[:, m] * np.maximum(cq[:, m], 1e-6) ** a
    # keep each row's NONE vs error split unchanged
    e = ref[:, m].sum(1, keepdims=True)
    adj[:, m] = adj[:, m] / adj[:, m].sum(1, keepdims=True) * e
    return normalize(adj)
R['claude_question_error_prior'] = crossfit(apply_cq, [0, .1, .2, .3, .5, .75])
# Combined gold+claude prior
R['claude_q_prior_then_prior_shift'] = crossfit(lambda g: normalize(apply_cq(g[0]) * np.exp(g[1] * shift)),
                                                [(a, b) for a in (0, .2, .3, .5) for b in (0, .2, .5)])
# 4. Weight re-tuning check (blend weights) cross-fitted, for scale of tuning noise
grid = [(a, b, 1 - a - b) for a in (.2, .35, .5) for b in (.2, .3, .4) if 1 - a - b > .05]
R['blend_weights_retuned'] = crossfit(lambda w: geo([(P['q32_tx'], w[0]), (P['q8_tx_e3'], w[1]), (P['vl_distill'], w[2])]), grid)
(OUT / 'a3_posthoc.json').write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
