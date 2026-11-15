"""(3) How easy is the question->code shortcut to learn from each training mix?

For each set: rows, questions, rows per question, within-question Simpson index of the label, leave-one-out
question-prior accuracy (predict a row's code from the OTHER rows of its question, smoothed to topic/global),
LOO cross-entropy reduction H(C) - H(C|Q)_LOO as a share of H(C) (how much of the classification loss question
identity alone can remove), and the LOO "new-pair" share (error rows whose code is absent from the other rows of the
question: the natural supply of examples that contradict the shortcut).
Also: coverage of dev/test questions by each set, and the candidate-student data (data/v5) shortcut rules.
"""
import json
from collections import Counter
import numpy as np
import pandas as pd
from k12.core import read_csv
from sc_common import load, claude_unl, slices, OUT, ROOT, r

tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
K = len(classes)
cl, unl = claude_unl()
qmap = unl.question_id.to_dict(); tmap = unl.topic_name.to_dict()
test = read_csv(ROOT / 'data/raw/test.csv')


def jsonl_set(path):
    rows = []
    for line in open(ROOT / path):
        x = json.loads(line)
        if x['trace_id'] in qmap:
            soft = x['soft']; code = max(soft, key=soft.get)
            rows.append(dict(trace_id=x['trace_id'], question_id=qmap[x['trace_id']], topic_name=tmap[x['trace_id']],
                             root_code=code, w=float(x.get('weight', 1.))))
    return pd.DataFrame(rows)


def loo(df, a=2., b=8.):
    """Leave-one-out hierarchical question prior on df itself."""
    y = df.root_code.map(ci).to_numpy()
    g = np.bincount(y, minlength=K) + .5; g = g / g.sum()
    tcount = {t: np.bincount(y[idx], minlength=K).astype(float) for t, idx in df.groupby('topic_name').indices.items()}
    qcount = {q: np.bincount(y[idx], minlength=K).astype(float) for q, idx in df.groupby('question_id').indices.items()}
    top1, map3, ll, newpair, nerr, nq = [], [], [], 0, 0, []
    for i, (q, t) in enumerate(zip(df.question_id, df.topic_name)):
        tc = tcount[t].copy(); tc[y[i]] -= 1
        qq = qcount[q].copy(); qq[y[i]] -= 1
        pt = (tc + b * g) / (tc.sum() + b)
        p = (qq + a * pt) / (qq.sum() + a)
        order = np.argsort(-p, kind='stable')[:3]
        top1.append(order[0] == y[i]); hits = np.where(order == y[i])[0]
        map3.append(1 / (hits[0] + 1) if len(hits) else 0.); ll.append(-np.log(p[y[i]]))
        nq.append(qq.sum())
        if classes[y[i]] != 'NONE':
            nerr += 1; newpair += qq[y[i]] == 0
    H = -(g * np.log(g)).sum()
    return dict(loo_top1=r(np.mean(top1)), loo_map3=r(np.mean(map3)), loo_ce_nats=r(np.mean(ll)), H_C_nats=r(H),
                share_of_H_removed_by_question=r(1 - np.mean(ll) / H), loo_new_pair_share_of_error_rows=r(newpair / max(1, nerr)),
                mean_other_rows_same_q=r(np.mean(nq), 2))


def describe(df, name):
    rpq = df.question_id.value_counts()
    within = []
    for q, g in df.groupby('question_id'):
        n = len(g)
        if n >= 2:
            c = g.root_code.value_counts().to_numpy()
            within.append(((c * (c - 1)).sum() / (n * (n - 1)), n))
    simpson = np.average([s for s, _ in within], weights=[n * (n - 1) / 2 for _, n in within]) if within else None
    maj = df.groupby('question_id').root_code.agg(lambda s: s.value_counts().iloc[0] / len(s))
    out = dict(rows=len(df), questions=int(df.question_id.nunique()), rows_per_q_mean=r(rpq.mean(), 2),
               rows_per_q_median=r(rpq.median(), 1), share_rows_in_q_with_10plus=r(df.question_id.map(rpq).ge(10).mean()),
               none_share=r((df.root_code == 'NONE').mean()), distinct_codes=int(df.root_code.nunique()),
               within_q_simpson_pairs=r(simpson), majority_share_mean_q_2plus=r(maj[rpq[maj.index] >= 2].mean()),
               dev_rows_whose_q_in_set=r(dev.question_id.isin(set(df.question_id)).mean()),
               test_rows_whose_q_in_set=r(test.question_id.isin(set(df.question_id)).mean()),
               rows_per_dev_q=r(dev.question_id.map(rpq).fillna(0).mean(), 2),
               rows_per_test_q=r(test.question_id.map(rpq).fillna(0).mean(), 2))
    out |= loo(df)
    return out


sets = {'fit_gold': fit[['trace_id', 'question_id', 'topic_name', 'root_code']],
        'train_all_gold': tr[['trace_id', 'question_id', 'topic_name', 'root_code']],
        'claude_unl_v1 (3,427; VL dev-distill, 32B-full)': jsonl_set('data/v3/claude_unl.jsonl'),
        'claude_unl_v2 (6,764; 32B explain run)': jsonl_set('data/v3/claude_unl_v2.jsonl'),
        'claude_unl_v4 (11,426; scale_100 8B)': jsonl_set('data/v3/claude_unl_v4.jsonl'),
        'claude_all_v4_first_code (26,202 raw)': cl[['trace_id', 'question_id', 'topic_name', 'root_code']]}
res = {name: describe(df, name) for name, df in sets.items()}
for k in ['claude_unl_v1 (3,427; VL dev-distill, 32B-full)', 'claude_unl_v2 (6,764; 32B explain run)', 'claude_unl_v4 (11,426; scale_100 8B)']:
    mix = pd.concat([sets['fit_gold'], sets[k][['trace_id', 'question_id', 'topic_name', 'root_code']]], ignore_index=True)
    res['MIX fit + ' + k] = describe(mix, k)

# Claude's labels: is Claude more or less concentrated per question than gold? (same questions, both sets with >=2 rows)
g2 = tr.groupby('question_id').root_code.apply(list); c2 = cl.groupby('question_id').root_code.apply(list)
def simp(lst):
    c = np.array(list(Counter(lst).values())); n = len(lst)
    return (c * (c - 1)).sum() / (n * (n - 1))
qs = [q for q in g2.index.intersection(c2.index) if len(g2[q]) >= 2 and len(c2[q]) >= 2]
res['gold_vs_claude_concentration_same_questions'] = dict(questions=len(qs), gold_simpson=r(np.mean([simp(g2[q]) for q in qs])),
                                                         claude_simpson=r(np.mean([simp(c2[q]) for q in qs])),
                                                         gold_none_share=r(np.mean([np.mean([c == 'NONE' for c in g2[q]]) for q in qs])),
                                                         claude_none_share=r(np.mean([np.mean([c == 'NONE' for c in c2[q]]) for q in qs])))
# Claude-mode agreement with gold mode on the same question
agree = [Counter(g2[q]).most_common(1)[0][0] == Counter(c2[q]).most_common(1)[0][0] for q in qs]
res['gold_vs_claude_concentration_same_questions']['mode_agreement'] = r(np.mean(agree))
# Error-only Simpson (excluding NONE) on both
qs_e = [q for q in qs if sum(c != 'NONE' for c in g2[q]) >= 2 and sum(c != 'NONE' for c in c2[q]) >= 2]
res['gold_vs_claude_concentration_same_questions'] |= dict(
    questions_error_only=len(qs_e), gold_error_simpson=r(np.mean([simp([c for c in g2[q] if c != 'NONE']) for q in qs_e])),
    claude_error_simpson=r(np.mean([simp([c for c in c2[q] if c != 'NONE']) for q in qs_e])))

# Do Claude rows on the dev questions carry the dev truth? per slice (supply of contradicting examples for new pairs)
S, qc = slices(dev, fit)
ccodes = {q: Counter(g.root_code) for q, g in cl[~cl.trace_id.isin(set(dev.trace_id))].groupby('question_id')}
cover = {}
for s in ['seen_pair', 'new_pair', 'unseen_q']:
    m = S == s
    has = [ccodes.get(q, Counter()).get(t, 0) > 0 for q, t in zip(dev.question_id[m], dev.root_code[m])]
    share = [ccodes.get(q, Counter()).get(t, 0) / max(1, sum(ccodes.get(q, Counter()).values())) for q, t in zip(dev.question_id[m], dev.root_code[m])]
    nrows = [sum(ccodes.get(q, Counter()).values()) for q in dev.question_id[m]]
    cover[s] = dict(truth_among_claude_first_codes_same_q=r(np.mean(has)), mean_share_of_claude_rows_with_truth=r(np.mean(share)),
                    claude_rows_per_q=r(np.mean(nrows), 1))
res['claude_rows_on_dev_questions'] = cover
json.dump(res, open(OUT / 's3_train_composition.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
