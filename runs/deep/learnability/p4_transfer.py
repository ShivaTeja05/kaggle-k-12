"""Part 4: transferability. For the 84 hard dev rows (seen question, gold code new for it) and 22 unseen-question
rows, does the Claude pool / fit gold contain the gold code on OTHER questions of the same topic / grade / section?
And does such cross-question evidence predict whether models get the row right (i.e. is transfer being learned)?"""
from lcommon import *

D = get_all(); y = D['y']; dev, fit, unl, tax, preds = D['dev'], D['fit'], D['unl'], D['tax'], D['preds']
q = dev.question_id.to_numpy(); top = dev.topic_name.to_numpy(); gr = dev.grade.astype(str).to_numpy(); n = len(y)
S = {k: score(y, v) for k, v in preds.items()}
fq = {qq: set(g.root_code) for qq, g in fit.groupby('question_id')}
hard = np.array([y[i] != 'NONE' and q[i] in fq and y[i] not in fq[q[i]] for i in range(n)])
unseen = np.array([y[i] != 'NONE' and q[i] not in fq for i in range(n)])
seenpair = np.array([y[i] != 'NONE' and q[i] in fq and y[i] in fq[q[i]] for i in range(n)])
used_ids = set(json.loads(l)['trace_id'] for l in open(ROOT / 'data/v3/claude_unl_v4.jsonl'))
U = unl.assign(used=unl.trace_id.isin(used_ids)); U['grade'] = U.grade.astype(str)
F = fit.assign(grade=fit.grade.astype(str))
sec = dict(zip(tax.id, tax.section)); U['sec1'] = U.c1.map(sec)

def counts(i, pool, code_col, any3=False):
    c, qq, t, g = y[i], q[i], top[i], gr[i]
    hit = pool[code_col].map(lambda cs: c in cs) if any3 else (pool[code_col] == c)
    other = pool.question_id != qq
    return dict(any_q=int((hit & other).sum()), same_topic=int((hit & other & (pool.topic_name == t)).sum()),
                same_grade=int((hit & other & (pool.grade == g)).sum()), same_q=int((hit & ~other).sum()),
                other_topic=int((hit & other & (pool.topic_name != t)).sum()),
                topic_questions_with_code=int(pool[hit & other & (pool.topic_name == t)].question_id.nunique()))
rows = []
for i in np.flatnonzero(hard | unseen | seenpair):
    r = dict(i=int(i), trace_id=dev.trace_id[i], regime='hard' if hard[i] else ('unseen_q' if unseen[i] else 'seen_pair'), gold=y[i], topic=top[i],
             generic=fam(y[i]) in GENERIC)
    for nm, pool, col, a3 in [('cl26k', U, 'c1', False), ('cl26k_any3', U, 'c3', True), ('clused', U[U.used], 'c1', False), ('fitgold', F, 'root_code', False)]:
        for k, v in counts(i, pool, col, a3).items():
            r[f'{nm}_{k}'] = v
    for k in ['blend_sk', 'claude', 'scale_0', 'scale_100', 'm32', 'm8', 'mvl']:
        r[f's_{k}'] = float(S[k][i])
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT / 'p4_transfer_rows.csv', index=False)

res = {}
for reg in ['hard', 'unseen_q', 'seen_pair']:
    t = T[T.regime == reg]; d = {'rows': int(len(t))}
    for nm in ['cl26k', 'cl26k_any3', 'clused', 'fitgold']:
        for k in ['same_q', 'same_topic', 'same_grade', 'any_q']:
            v = t[f'{nm}_{k}']
            d[f'{nm}_{k}'] = {'>=1': round(float((v >= 1).mean()), 3), '>=5': round(float((v >= 5).mean()), 3), '>=20': round(float((v >= 20).mean()), 3), 'median': float(v.median())}
        d[f'{nm}_topic_questions_with_code_median'] = float(t[f'{nm}_topic_questions_with_code'].median())
    d['generic_code_share'] = round(float(t.generic.mean()), 3)
    res[reg] = d

# Does cross-question same-topic evidence predict success on hard rows? (controls: same-q Claude count = 0)
h = T[(T.regime == 'hard')]
def split(mask, lab):
    out = {}
    for k in ['blend_sk', 'claude', 'scale_0', 'scale_100']:
        a, b = h[mask][f's_{k}'], h[~mask][f's_{k}']
        out[k] = [round(float(a.mean()), 3), round(float(b.mean()), 3)]
    out['rows'] = [int(mask.sum()), int((~mask).sum())]
    return {lab: out}
res['hard_rows_score_[with, without]'] = {}
med = h.clused_same_topic.median()
for mask, lab in [(h.clused_same_topic >= 5, 'used-pool same-topic other-q >=5'), (h.clused_same_topic >= 1, 'used-pool same-topic other-q >=1'),
                  (h.fitgold_same_topic >= 1, 'fit-gold same-topic other-q >=1'), (h.clused_same_q >= 1, 'used-pool same-q >=1'),
                  (h.generic, 'gold code generic')]:
    res['hard_rows_score_[with, without]'].update(split(mask.to_numpy(), lab))
# Difference-in-differences: gain from Claude labels (scale_100 - scale_0) vs evidence type, question-bootstrap
gain = (h.s_scale_100 - h.s_scale_0).to_numpy(); hq = q[h.i.to_numpy()]
rng = np.random.default_rng(11); uqs = np.unique(hq)
def did(mask):
    mask = mask.to_numpy(); est = gain[mask].mean() - gain[~mask].mean(); bs = []
    for _ in range(3000):
        pick = rng.choice(uqs, len(uqs)); idx = np.concatenate([np.flatnonzero(hq == u) for u in pick])
        m = mask[idx]
        if m.any() and (~m).any():
            bs.append(gain[idx][m].mean() - gain[idx][~m].mean())
    return {'gain_with': round(float(gain[mask].mean()), 3), 'gain_without': round(float(gain[~mask].mean()), 3), 'did': round(float(est), 3),
            'ci95': [round(float(x), 3) for x in np.quantile(bs, [.025, .975])], 'rows_with': int(mask.sum())}
sq0 = h.clused_same_q == 0
res['did_scale100_minus_scale0_on_hard_rows'] = {
    'same-q pair in used Claude pool (>=1)': did(h.clused_same_q >= 1),
    'same-topic other-q code in used pool >=5 (all hard rows)': did(h.clused_same_topic >= 5),
    'same-topic other-q code in used pool >=5 (only rows with no same-q pair)': None}
hh = h[sq0]; gain2 = (hh.s_scale_100 - hh.s_scale_0).to_numpy()
m2 = (hh.clused_same_topic >= 5).to_numpy()
res['did_scale100_minus_scale0_on_hard_rows']['same-topic other-q code in used pool >=5 (only rows with no same-q pair)'] = {
    'gain_with': round(float(gain2[m2].mean()), 3) if m2.any() else None, 'gain_without': round(float(gain2[~m2].mean()), 3), 'rows_with': int(m2.sum()), 'rows_without': int((~m2).sum())}
# Claude's own success vs evidence (Claude is not trained on these, so this measures intrinsic row easiness)
res['claude_score_hard_rows_by_same_q_pool'] = {'>=1': round(float(h[h.cl26k_same_q >= 1].s_claude.mean()), 3), '0': round(float(h[h.cl26k_same_q == 0].s_claude.mean()), 3)}
# Pool-wide: how topic-bound are codes? share of a code's Claude/gold occurrences in its top topic
def topic_conc(pool, col):
    out = {}
    for c, g in pool[pool[col] != 'NONE'].groupby(col):
        if len(g) >= 10:
            out[c] = float(g.topic_name.value_counts(normalize=True).iloc[0])
    return out
cg = topic_conc(U, 'c1'); fg = topic_conc(F, 'root_code')
res['code_topic_concentration_(share in top topic)'] = {'claude_pool_median': round(float(np.median(list(cg.values()))), 3),
                                                         'fit_gold_median': round(float(np.median(list(fg.values()))), 3),
                                                         'generic_codes_claude_median': round(float(np.median([v for c, v in cg.items() if fam(c) in GENERIC])), 3),
                                                         'specific_codes_claude_median': round(float(np.median([v for c, v in cg.items() if fam(c) not in GENERIC])), 3)}
# Hard-row gold codes: how many distinct questions does the used training pool show that code on (diversity of contexts)?
res['hard_rows_gold_code_context_diversity'] = {'used_pool_distinct_q_any_topic_median': float(np.median([U[U.used & (U.c1 == c)].question_id.nunique() for c in h.gold])),
                                                'fit_gold_distinct_q_median': float(np.median([F[F.root_code == c].question_id.nunique() for c in h.gold]))}
json.dump(res, open(OUT / 'p4_transfer.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
