"""Ensemble anatomy: reproduce the 0.718 dev blend and dissect where the gap to Claude (0.819) lives.

CPU only. Reads existing dev bundles; writes only to runs/arch/ensemble-anatomy/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_claude import claude_scores
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk

OUT = Path('runs/arch/ensemble-anatomy')
RNG = np.random.default_rng(2026)

train = read_csv('data/raw/train.csv')
tax = read_csv('data/raw/taxonomy_v1.csv')
classes = label_space(tax)
C = np.asarray(classes)
folds = make_folds(train)
fit = train.loc[folds != 0].reset_index(drop=True)
dev = train.loc[folds == 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
y = dev.root_code.to_numpy()
yi = np.array([classes.index(c) for c in y])
N = len(dev)

B = {
    'q32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
    'q8_e2': 'runs/pod/tx/text_tx_dev_e2.npz',
    'q8_e3': 'runs/pod/tx/text_tx_dev_e3.npz',
    'vl': 'runs/pod/dev-distill/validation_epoch_2.npz',
    # extras for the union/oracle view
    'q8_s100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
    'q8_s100_gold': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz',
    'q32_gold': 'runs/pod/text32/validation_gold_epoch_1.npz',
    'fs235': 'runs/pod/probe235/fewshot235/validation.npz',
    'vl_dev_v3': 'runs/pod/dev-v3/validation_epoch_3.npz',
    'vl_dev_cls3': 'runs/pod/dev-cls/validation_epoch_3.npz',
    'vl_dev_read': 'runs/pod/dev-read/validation_epoch_3.npz',
    'vl_dev_aux': 'runs/pod/dev-aux/validation_epoch_3.npz',
    'vl_old': 'runs/vision-development/k12/validation_epoch_3.npz',
}
P = {}
for k, path in B.items():
    try:
        P[k] = aligned_bundle(path, ids, classes)
    except Exception as e:  # noqa
        print('skip', k, e)
P['claude'] = claude_scores('runs/claude/dev_opus.jsonl', ids, classes)[0]


def geo(parts):
    lp = sum(w * np.log(np.clip(P[k], 1e-12, None)) for k, w in parts)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


def pts(p):
    return row_scores(y, topk(p, classes))


def rank_of_truth(p):
    order = np.argsort(-p, axis=1, kind='stable')
    return np.argmax(order == yi[:, None], axis=1) + 1  # 1-based


report = {'rows': N}
# ---------------- 1. reproduce ----------------
repro = {}
for name in ('q8_e2', 'q8_e3'):
    for w in [(.35, .30, .35)]:
        p = geo([('q32', w[0]), (name, w[1]), ('vl', w[2])])
        repro[f'geo_q32.35_{name}.30_vl.35'] = float(pts(p).mean())
report['reproduce'] = repro
best8 = max(('q8_e2', 'q8_e3'), key=lambda n: repro[f'geo_q32.35_{n}.30_vl.35'])
report['reproducing_8b_bundle'] = B[best8]
P['blend'] = geo([('q32', .35), (best8, .30), ('vl', .35)])
P['q8'] = P[best8]
core = ['q32', 'q8', 'vl']
S = {k: pts(P[k]) for k in P}
R = {k: rank_of_truth(P[k]) for k in P}
report['single_map3'] = {k: round(float(S[k].mean()), 4) for k in P}
print('reproduce', repro)
print({k: round(float(v.mean()), 4) for k, v in S.items()})

blend, cl = S['blend'], S['claude']
gap_rows = cl - blend
report['gap_total'] = float(gap_rows.mean())


# question-grouped paired bootstrap
def boot_ci(delta, groups=dev.question_id.to_numpy(), n=4000):
    g = pd.DataFrame({'q': groups, 'd': delta}).groupby('q').d.agg(['sum', 'size']).to_numpy()
    s = RNG.integers(0, len(g), (n, len(g)))
    b = g[s, 0].sum(1) / g[s, 1].sum(1)
    return [round(float(x), 4) for x in np.quantile(b, [.025, .975])]


report['gap_ci95'] = boot_ci(gap_rows)

# ---------------- 2. features ----------------
fam = lambda c: 'NONE' if c == 'NONE' else c.split('-')[0].upper()
sec_map = dict(zip(tax.id, tax.section))
nat_map = dict(zip(tax.id, tax.nature))
fit_freq = fit.root_code.value_counts()
seen_q = set(fit.question_id)
ranked = {k: topk(P[k], classes, 10) for k in ['blend', 'claude'] + core}
bt1, ct1 = ranked['blend'][:, 0], ranked['claude'][:, 0]


def nmis(s):
    try:
        return len(json.loads(s))
    except Exception:  # noqa
        return 0


def ncat(s):
    try:
        return len({m['category'] for m in json.loads(s)})
    except Exception:  # noqa
        return 0


D = pd.DataFrame({
    'trace_id': ids, 'question_id': dev.question_id, 'truth': y, 'grade': dev.grade.astype(int),
    'is_none': y == 'NONE', 'family': [fam(c) for c in y],
    'section': [sec_map.get(c, 'NONE') for c in y], 'nature': [nat_map.get(c, 'NONE') or 'blank' for c in y],
    'fit_freq': [int(fit_freq.get(c, 0)) for c in y], 'seen_q': [q in seen_q for q in dev.question_id],
    'final_correct': dev.final_answer_correct.astype(str), 'clean': dev.clean.astype(str),
    'n_misc': [nmis(s) for s in dev.misconceptions], 'n_cats': [ncat(s) for s in dev.misconceptions],
    'blend': blend, 'claude': cl, 'gap': gap_rows,
    'blend_top1': bt1, 'claude_top1': ct1,
    'rank_blend': R['blend'], 'rank_claude': R['claude'],
    **{f's_{k}': S[k] for k in core}, **{f'rank_{k}': R[k] for k in core},
})
D['freq_bin'] = pd.cut(D.fit_freq, [-1, 0, 20, 40, 50, 1e9], labels=['0', '1-20', '21-40', '41-50', '>50(NONE)'])
D['grade_band'] = pd.cut(D.grade, [3, 6, 9, 12], labels=['4-6', '7-9', '10-12'])
D['multi_cat'] = np.where(D.n_cats >= 2, 'multi-category', np.where(D.n_cats == 1, 'single-category', 'no-annotation'))


def split(col):
    g = D.groupby(col, observed=True).agg(n=('blend', 'size'), blend=('blend', 'mean'), claude=('claude', 'mean'),
                                          gap_pts=('gap', 'sum'))
    g['gap_share_of_total'] = g.gap_pts / D.gap.sum()
    g['gap_map_contrib'] = g.gap_pts / N
    g['blend_lost_map_contrib'] = D.groupby(col, observed=True).blend.apply(lambda s: (1 - s).sum() / N)
    return g.round(4).sort_values('gap_pts', ascending=False)


splits = {}
for col in ['is_none', 'section', 'family', 'nature', 'freq_bin', 'grade', 'grade_band', 'seen_q',
            'final_correct', 'clean', 'multi_cat']:
    splits[col] = split(col)
    print('\n==', col)
    print(splits[col].to_string())
report['splits'] = {k: v.reset_index().astype({v.index.name or 'index': str}).to_dict(orient='records')
                    for k, v in splits.items()}


# ---------------- 3. outcome matrix vs Claude ----------------
def outcome(b, c):
    if b == 1 and c == 1:
        return 'both_top1'
    if b == 0 and c == 0:
        return 'both_miss_top3'
    if c > b:
        return 'claude_better'
    if b > c:
        return 'blend_better'
    return 'tie_partial'


D['outcome'] = [outcome(b, c) for b, c in zip(D.blend, D.claude)]
oc = D.groupby('outcome').agg(n=('gap', 'size'), gap_pts=('gap', 'sum'), blend=('blend', 'mean'), claude=('claude', 'mean'))
oc['gap_map_contrib'] = oc.gap_pts / N
report['outcome_vs_claude'] = oc.round(4).reset_index().to_dict(orient='records')
print('\n== outcome vs claude\n', oc.round(4).to_string())

# finer: Claude top1-right & blend's rank of truth
cr = D[D.rank_claude == 1]
report['claude_top1_right_rows'] = int(len(cr))
report['blend_rank_of_truth_when_claude_top1'] = cr.rank_blend.clip(upper=11).value_counts().sort_index().to_dict()


# ---------------- 4. error-type taxonomy of top1 ----------------
def errtype(t, p):
    if p == t:
        return 'correct'
    if t == 'NONE':
        return 'false_alarm_on_NONE'
    if p == 'NONE':
        return 'missed_error_said_NONE'
    if fam(p) == fam(t):
        return 'right_family_wrong_code'
    if sec_map.get(p) == sec_map.get(t):
        return 'right_section_wrong_family'
    return 'wrong_section'


D['blend_err'] = [errtype(t, p) for t, p in zip(y, bt1)]
D['claude_err'] = [errtype(t, p) for t, p in zip(y, ct1)]
et = pd.crosstab(D.blend_err, D.claude_err, margins=True)
report['top1_error_crosstab_blend_rows_claude_cols'] = et.to_dict()
print('\n== top1 error crosstab (rows blend, cols claude)\n', et.to_string())
eg = D.groupby('blend_err').agg(n=('gap', 'size'), blend=('blend', 'mean'), claude=('claude', 'mean'), gap_pts=('gap', 'sum'))
eg['gap_map_contrib'] = eg.gap_pts / N
eg['blend_lost_map'] = D.groupby('blend_err').blend.apply(lambda s: (1 - s).sum() / N)
report['gap_by_blend_top1_errtype'] = eg.round(4).reset_index().to_dict(orient='records')
print('\n== gap by blend top1 error type\n', eg.round(4).to_string())

# ---------------- 5. oracles / recall ----------------
rec = {}
for k in core + ['blend', 'claude'] + [x for x in P if x not in core + ['blend', 'claude', best8]]:
    r = R[k]
    rec[k] = {'map3': round(float(S[k].mean()), 4), **{f'recall@{j}': round(float(np.mean(r <= j)), 4) for j in (1, 2, 3, 5, 10, 20)},
              'mean_rank_truth': round(float(r.mean()), 2)}
report['recall'] = rec
print('\n== recall')
print(pd.DataFrame(rec).T.to_string())

top3 = {k: topk(P[k], classes, 3) for k in P}
union3 = np.any(np.concatenate([top3[k] for k in core], 1) == y[:, None], 1)
union3_all = np.any(np.concatenate([top3[k] for k in P if k not in ('claude', 'blend')], 1) == y[:, None], 1)
best_single = np.max(np.stack([S[k] for k in core]), 0)
oracle = {
    'per_row_best_of_3_models_map3': float(best_single.mean()),
    'per_row_best_of_3_plus_blend_map3': float(np.maximum(best_single, blend).mean()),
    'union_top3_of_3_models_recall': float(union3.mean()),
    'union_top3_all_offline_bundles_recall': float(union3_all.mean()),
    'n_offline_bundles_in_union_all': len([k for k in P if k not in ('claude', 'blend')]) - 1,
    'blend_perfect_rerank_top3_map3(=recall@3)': float(np.mean(R['blend'] <= 3)),
    'blend_perfect_rerank_top5': float(np.mean(R['blend'] <= 5)),
    'blend_perfect_rerank_top10': float(np.mean(R['blend'] <= 10)),
    'claude_recall@3': float(np.mean(R['claude'] <= 3)),
    'claude_or_blend_best_per_row': float(np.maximum(cl, blend).mean()),
}
report['oracles'] = oracle
print('\n== oracles', json.dumps(oracle, indent=1))

# blend rank-of-truth distribution, points lost by bucket
bk = pd.cut(pd.Series(R['blend']), [0, 1, 2, 3, 5, 10, 20, 100], labels=['1', '2', '3', '4-5', '6-10', '11-20', '>20'])
rb = pd.DataFrame({'bucket': bk, 'lost': 1 - blend, 'claude': cl, 'gap': gap_rows}).groupby('bucket', observed=True).agg(
    n=('lost', 'size'), blend_lost_pts=('lost', 'sum'), claude_map=('claude', 'mean'), gap_pts=('gap', 'sum'))
rb['blend_lost_map'] = rb.blend_lost_pts / N
rb['gap_map_contrib'] = rb.gap_pts / N
report['blend_truth_rank_buckets'] = rb.round(4).reset_index().astype({'bucket': str}).to_dict(orient='records')
print('\n== blend truth-rank buckets\n', rb.round(4).to_string())

# ---------------- 6. oracle gates: what if we knew NONE / family ----------------
g = {}
p = P['blend'].copy()
none_i = classes.index('NONE')
isn = y == 'NONE'
q = p.copy()
q[isn] = 0; q[isn, none_i] = 1
q[~isn, none_i] = 0
g['oracle_none_gate'] = float(pts(normalize(q + 1e-12)).mean())
famarr = np.array([fam(c) for c in classes])
q = p * (famarr[None, :] == np.array([fam(c) for c in y])[:, None])
g['oracle_family_gate'] = float(pts(normalize(q + 1e-12)).mean())
secarr = np.array([sec_map.get(c, 'NONE') for c in classes])
q = p * (secarr[None, :] == np.array([sec_map.get(c, 'NONE') for c in y])[:, None])
g['oracle_section_gate'] = float(pts(normalize(q + 1e-12)).mean())
# claude's top1 family as a gate (shows value of a strong family router)
cf = np.array([fam(c) for c in ct1])
q = p * (famarr[None, :] == cf[:, None]) + 1e-6 * p
g['claude_top1_family_gate_on_blend'] = float(pts(normalize(q)).mean())
report['oracle_gates'] = g
print('\n== gates', g)

# ---------------- 7. Claude errors that we get right (complementarity) ----------------
report['complementarity'] = {
    'rows_claude_better': int((gap_rows > 0).sum()), 'rows_blend_better': int((gap_rows < 0).sum()),
    'pts_claude_better': float(gap_rows[gap_rows > 0].sum()), 'pts_blend_better': float(-gap_rows[gap_rows < 0].sum()),
    'agree_top1_rate': float(np.mean(bt1 == ct1)),
    'acc_when_agree_top1': float(np.mean((bt1 == y)[bt1 == ct1])),
    'claude_acc_when_disagree': float(np.mean((ct1 == y)[bt1 != ct1])),
    'blend_acc_when_disagree': float(np.mean((bt1 == y)[bt1 != ct1])),
}
print('\n== complementarity', report['complementarity'])

# blend confidence vs accuracy (does it know when it's wrong?)
conf = P['blend'].max(1)
qb = pd.qcut(conf, 5, labels=False)
cv = pd.DataFrame({'q': qb, 'conf': conf, 'blend': blend, 'claude': cl, 'gap': gap_rows}).groupby('q').agg(
    n=('blend', 'size'), conf=('conf', 'mean'), blend=('blend', 'mean'), claude=('claude', 'mean'), gap_pts=('gap', 'sum'))
report['confidence_quintiles'] = cv.round(4).reset_index().to_dict(orient='records')
print('\n== blend confidence quintiles\n', cv.round(4).to_string())

# per-model agreement & pairwise disagreement
t1 = {k: topk(P[k], classes, 1)[:, 0] for k in core}
report['pairwise_top1_agreement'] = {f'{a}-{b}': float(np.mean(t1[a] == t1[b])) for i, a in enumerate(core) for b in core[i + 1:]}
allagree = (t1['q32'] == t1['q8']) & (t1['q8'] == t1['vl'])
report['all3_agree'] = {'rate': float(allagree.mean()), 'acc_when_all_agree': float(np.mean((t1['q32'] == y)[allagree])),
                        'blend_map_when_not_all_agree': float(blend[~allagree].mean()),
                        'claude_map_when_not_all_agree': float(cl[~allagree].mean())}
print('\n== agreement', report['pairwise_top1_agreement'], report['all3_agree'])

# per-code table
pc = D[~D.is_none].groupby('truth').agg(n=('blend', 'size'), fit_freq=('fit_freq', 'first'), blend=('blend', 'mean'),
                                        claude=('claude', 'mean'), gap_pts=('gap', 'sum')).sort_values('gap_pts', ascending=False)
pc['gap_map_contrib'] = pc.gap_pts / N
pc.round(4).to_csv(OUT / 'per_code.csv')
# most common confusions (truth -> blend top1) where claude got it
conf_pairs = D[(D.blend_err != 'correct')].groupby(['truth', 'blend_top1']).size().sort_values(ascending=False).head(25)
report['top_confusions_truth_to_blend_top1'] = [{'truth': a, 'pred': b, 'n': int(n)} for (a, b), n in conf_pairs.items()]
pred_counts = pd.Series(bt1).value_counts()
report['blend_top1_overpredicted'] = {k: int(v) for k, v in (pred_counts - pd.Series(y).value_counts()).dropna().sort_values(ascending=False).head(12).items()}
print('\n== top confusions\n', conf_pairs.to_string())
print('\n== overpredicted (pred count - true count)\n', report['blend_top1_overpredicted'])

D.to_csv(OUT / 'rows.csv', index=False)
(OUT / 'anatomy.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
print('\nwrote', OUT)
