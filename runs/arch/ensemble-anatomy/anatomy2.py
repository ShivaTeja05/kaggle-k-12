"""Part 2: label ambiguity, question-conditioned candidate sets, cross-fitted blend tuning with paired CIs."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_claude import claude_scores
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk

OUT = Path('runs/arch/ensemble-anatomy')
train = read_csv('data/raw/train.csv')
classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
C = np.asarray(classes)
folds = make_folds(train)
fit = train.loc[folds != 0].reset_index(drop=True)
dev = train.loc[folds == 0].reset_index(drop=True)
ids, y = dev.trace_id.tolist(), dev.root_code.to_numpy()
N = len(dev)
qid = dev.question_id.to_numpy()
B = {'q32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 'q8': 'runs/pod/tx/text_tx_dev_e3.npz',
     'vl': 'runs/pod/dev-distill/validation_epoch_2.npz', 'q8_e2': 'runs/pod/tx/text_tx_dev_e2.npz',
     'q8_s100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
     'fs235': 'runs/pod/probe235/fewshot235/validation.npz', 'vl_v3': 'runs/pod/dev-v3/validation_epoch_3.npz'}
P = {k: aligned_bundle(v, ids, classes) for k, v in B.items()}
P['claude'] = claude_scores('runs/claude/dev_opus.jsonl', ids, classes)[0]
L = {k: np.log(np.clip(v, 1e-12, None)) for k, v in P.items()}


def geo(ws):
    lp = sum(w * L[k] for k, w in ws.items() if w)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


pts = lambda p: row_scores(y, topk(p, classes))
ref = pts(geo({'q32': .35, 'q8': .30, 'vl': .35}))
rep = {'reference_map3': float(ref.mean())}
rng = np.random.default_rng(7)


def boot(delta, n=4000):
    g = pd.DataFrame({'q': qid, 'd': delta}).groupby('q').d.agg(['sum', 'size']).to_numpy()
    s = rng.integers(0, len(g), (n, len(g)))
    b = g[s, 0].sum(1) / g[s, 1].sum(1)
    return [round(float(x), 4) for x in np.quantile(b, [.025, .975])]


# ---- A. label ambiguity: is a wrong top-1 one of the row's other annotated misconception categories? ----
def cats(s):
    try:
        return {m['category'] for m in json.loads(s)}
    except Exception:  # noqa
        return set()


cs = [cats(s) for s in dev.misconceptions]
amb = {}
for name in ('blend', 'claude'):
    p = geo({'q32': .35, 'q8': .30, 'vl': .35}) if name == 'blend' else P['claude']
    t3 = topk(p, classes, 3)
    wrong = (t3[:, 0] != y) & (y != 'NONE')
    in_other = np.array([t3[i, 0] in cs[i] - {y[i]} for i in range(N)])
    amb[name] = {'error_rows_top1_wrong': int(wrong.sum()),
                 'top1_is_another_annotated_category': int((wrong & in_other).sum())}
rep['label_ambiguity'] = amb

# ---- B. question-conditioned candidate sets (seen questions) ----
qcodes = fit.groupby('question_id').root_code.apply(list).to_dict()
seen = np.array([q in qcodes for q in qid])
truth_in_q = np.array([y[i] in qcodes.get(qid[i], []) for i in range(N)])
truth_in_q_err = np.array([y[i] in set(qcodes.get(qid[i], [])) - {'NONE'} for i in range(N)])
size_q = np.array([len(set(qcodes.get(q, []))) for q in qid])
blend = geo({'q32': .35, 'q8': .30, 'vl': .35})
r = np.argsort(-blend, 1, kind='stable')
rank = np.argmax(C[r] == y[:, None], 1) + 1
rep['question_candidates'] = {
    'dev_rows_seen_question': int(seen.sum()),
    'seen_rows_truth_in_fit_codes_for_question': float(truth_in_q[seen].mean()),
    'seen_error_rows_truth_in_fit_error_codes_for_question': float(truth_in_q_err[seen & (y != 'NONE')].mean()),
    'median_distinct_codes_per_seen_question': float(np.median(size_q[seen])),
    'mean_fit_rows_per_seen_question': float(np.mean([len(qcodes[q]) for q in qid[seen]])),
    'blend_map_seen_truth_in_qset': float(ref[seen & truth_in_q].mean()),
    'blend_map_seen_truth_not_in_qset': float(ref[seen & ~truth_in_q].mean()),
    'claude_map_seen_truth_in_qset': float(pts(P['claude'])[seen & truth_in_q].mean()),
    'claude_map_seen_truth_not_in_qset': float(pts(P['claude'])[seen & ~truth_in_q].mean()),
    'n_seen_truth_not_in_qset': int((seen & ~truth_in_q).sum()),
}
# question prior multiplier (cross-fitted: prior from fit rows only, no tuning on dev except weight -> CV)
alpha = 1.0
prior = np.full((N, len(classes)), 0.0)
for i, q in enumerate(qid):
    for c in qcodes.get(q, []):
        prior[i, classes.index(c)] += 1
prior_n = (prior + alpha) / (prior + alpha).sum(1, keepdims=True)
L['qprior'] = np.log(prior_n)

# ---- C. cross-fitted weight search (5 folds over dev, grouped by question) ----
uq = np.unique(qid)
perm = np.random.default_rng(11).permutation(len(uq))
qf = {q: i % 5 for i, q in zip(range(len(uq)), uq[perm])}
f5 = np.array([qf[q] for q in qid])
grid = np.arange(0, 1.01, .1)


def crossfit(members, grid_vals=grid):
    combos = [w for w in itertools.product(grid_vals, repeat=len(members)) if sum(w) > 0]
    scored = {}
    for w in combos:
        scored[w] = pts(geo(dict(zip(members, w))))
    out = np.zeros(N)
    chosen = []
    for k in range(5):
        tr = f5 != k
        best = max(combos, key=lambda w: (scored[w][tr].mean(), -abs(sum(w) - 1)))
        out[f5 == k] = scored[best][f5 == k]
        chosen.append([round(float(x), 2) for x in best])
    best_in = max(combos, key=lambda w: scored[w].mean())
    return out, chosen, best_in, float(scored[best_in].mean())


cf = {}
for members in (['q32', 'q8', 'vl'], ['q32', 'q8', 'vl', 'q8_s100'], ['q32', 'q8', 'vl', 'qprior'],
                ['q32', 'q8', 'vl', 'q8_s100', 'qprior'], ['q32', 'q8', 'vl', 'fs235'], ['q32', 'q8', 'vl', 'q8_e2']):
    g = grid if len(members) <= 4 else np.arange(0, 1.01, .2)
    out, chosen, best_in, in_map = crossfit(members, g)
    d = out - ref
    cf['+'.join(members)] = {'crossfit_map3': round(float(out.mean()), 4), 'delta_vs_0.718': round(float(d.mean()), 4),
                             'ci95': boot(d), 'fold_weights': chosen, 'in_sample_best_weights': [round(float(x), 2) for x in best_in],
                             'in_sample_best_map3_(optimistic)': round(in_map, 4)}
    print(members, cf['+'.join(members)])
rep['crossfit_blends'] = cf

# ---- D. how far is a NONE/error switch? confidence-calibrated NONE threshold, cross-fitted ----
ni = classes.index('NONE')
pn = blend[:, ni]
rep['none'] = {'true_none_rows': int((y == 'NONE').sum()), 'blend_none_top1': int((blend.argmax(1) == ni).sum()),
               'claude_none_top1': int((P['claude'].argmax(1) == ni).sum())}

(OUT / 'anatomy2.json').write_text(json.dumps(rep, indent=2) + '\n')
print(json.dumps({k: v for k, v in rep.items() if k != 'crossfit_blends'}, indent=1))
