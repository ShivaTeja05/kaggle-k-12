"""Part 4b: stratify the 84 hard rows by what evidence exists anywhere in our data; score models and gap per stratum."""
from lcommon import *
D = get_all(); dev, tr, unl, fit = D['dev'], D['tr'], D['unl'], D['fit']
T = pd.read_csv(OUT / 'p4_transfer_rows.csv', keep_default_na=False)
used_ids = set(json.loads(l)['trace_id'] for l in open(ROOT / 'data/v3/claude_unl_v4.jsonl'))
res = {'used_pool_rows': len(used_ids), 'used_pool_error_rows': int(unl[unl.trace_id.isin(used_ids)].c1.ne('NONE').sum()),
       'pool_error_rows_total': int(unl.c1.ne('NONE').sum()), 'topics_train': int(tr.topic_name.nunique()), 'topics_test': int(D['test'].topic_name.nunique()),
       'unl_topics': int(unl.topic_name.nunique())}
h = T[T.regime.isin(['hard', 'unseen_q'])].copy()
def stratum(r):
    if r.cl26k_same_q >= 1:
        return 'A_same_q_claude_pair'
    if r.cl26k_same_topic >= 1 or r.fitgold_same_topic >= 1:
        return 'B_code_seen_in_topic_other_q'
    if r.cl26k_same_grade >= 1 or r.fitgold_same_grade >= 1:
        return 'C_code_seen_only_other_topics_same_grade'
    return 'D_code_seen_only_elsewhere_or_never'
h['stratum'] = h.apply(stratum, axis=1)
q = dev.set_index('trace_id').question_id
n = len(dev); out = {}
for reg in ['hard', 'unseen_q']:
    for s, g in h[h.regime == reg].groupby('stratum'):
        d, ci = boot_ci(g.s_claude.values, g.s_blend_sk.values, q[g.trace_id].values)
        out[f'{reg}|{s}'] = {'rows': int(len(g)), 'generic_share': round(float(g.generic.astype(str).eq('True').mean()), 2),
                             'blend_sk': round(float(g.s_blend_sk.mean()), 3), 'claude': round(float(g.s_claude.mean()), 3),
                             'scale_0': round(float(g.s_scale_0.mean()), 3), 'scale_100': round(float(g.s_scale_100.mean()), 3),
                             'gap_points_per_dev_row': round(float((g.s_claude - g.s_blend_sk).sum() / n), 4), 'claude_minus_blend_ci': [round(x, 3) for x in ci],
                             'codes': Counter(g.gold).most_common(6)}
res['strata'] = out
# topic-level DiD: does fit-gold same-topic evidence help trained models more than it helps Claude? (hard rows)
hh = T[T.regime == 'hard']; m = (hh.fitgold_same_topic >= 1).to_numpy(); hq = q[hh.trace_id].values
rng = np.random.default_rng(5); uq = np.unique(hq)
for k in ['blend_sk', 'scale_0', 'scale_100']:
    dlt = (hh[f's_{k}'] - hh.s_claude).to_numpy(); est = dlt[m].mean() - dlt[~m].mean(); bs = []
    for _ in range(3000):
        pick = rng.choice(uq, len(uq)); idx = np.concatenate([np.flatnonzero(hq == u) for u in pick]); mm = m[idx]
        if mm.any() and (~mm).any():
            bs.append(dlt[idx][mm].mean() - dlt[idx][~mm].mean())
    res[f'did_{k}_minus_claude_by_fitgold_same_topic'] = {'with': round(float(dlt[m].mean()), 3), 'without': round(float(dlt[~m].mean()), 3),
                                                          'did': round(float(est), 3), 'ci95': [round(float(x), 3) for x in np.quantile(bs, [.025, .975])]}
json.dump(res, open(OUT / 'p4b_strata.json', 'w'), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
