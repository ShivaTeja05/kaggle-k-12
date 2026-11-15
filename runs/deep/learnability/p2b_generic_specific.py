"""Part 2b: the generic-vs-specific boundary (largest error relation for every model). Direction, pairs, learnability."""
from lcommon import *

D = get_all(); y = D['y']; dev, fit, tr, preds, unl = D['dev'], D['fit'], D['tr'], D['preds'], D['unl']
q = dev.question_id.to_numpy(); n = len(y)
S = {k: score(y, v) for k, v in preds.items()}
G = lambda c: c != 'NONE' and fam(c) in GENERIC
res = {}
for k in ['blend_sk', 'claude', 'claude_fewshot', 'claude_fable']:
    t1 = preds[k][:, 0]
    gs = [(y[i], t1[i]) for i in range(n) if y[i] != t1[i] and 'NONE' not in (y[i], t1[i]) and G(y[i]) != G(t1[i])]
    gold_gen = [p for p in gs if G(p[0])]; gold_spec = [p for p in gs if not G(p[0])]
    res[k] = {'rows': len(gs), 'gold_generic_pred_specific': len(gold_gen), 'gold_specific_pred_generic': len(gold_spec),
              'top_pairs(gold->pred)': Counter(f'{a}->{b}' for a, b in gs).most_common(8)}
# Train-level: how often is the gold generic? and is it predictable from question history?
e = tr[tr.root_code != 'NONE']
res['train_err_generic_share'] = round(float(e.root_code.map(G).mean()), 3)
# Within question: do questions mix generic and specific gold roots?
mix = e.groupby('question_id').root_code.agg(lambda s: (s.map(G).any() and (~s.map(G)).any()))
res['questions_with>=2_err_rows'] = int((e.groupby('question_id').size() >= 2).sum())
res['share_of_those_mixing_generic_and_specific'] = round(float(mix[e.groupby('question_id').size() >= 2].mean()), 3)
# Claude's own generic rate on unlabelled vs gold
ue = unl[unl.c1 != 'NONE']
res['claude_pool_err_generic_share'] = round(float(ue.c1.map(G).mean()), 3)
de = y != 'NONE'
for k in ['blend_sk', 'claude']:
    t1 = preds[k][:, 0]; m = de & (t1 != 'NONE')
    res[f'{k}_dev_pred_generic_share_on_err_rows'] = round(float(np.mean([G(c) for c in t1[m]])), 3)
res['dev_gold_generic_share_on_err_rows'] = round(float(np.mean([G(c) for c in y[de]])), 3)
# Secondary annotations: when gold root is specific, is a generic code annotated as secondary (and vice versa)?
def anns(s):
    try:
        return json.loads(s)
    except Exception:
        return []
sp_with_gen_sec = gen_with_sp_sec = sp = gn = 0
for r in e.itertuples():
    a = anns(r.misconceptions); secs = [m['category'] for m in a if not m.get('root')]
    if G(r.root_code):
        gn += 1; gen_with_sp_sec += any(not G(c) for c in secs)
    else:
        sp += 1; sp_with_gen_sec += any(G(c) for c in secs)
res['train_specific_root_with_generic_secondary'] = round(sp_with_gen_sec / sp, 3)
res['train_generic_root_with_specific_secondary'] = round(gen_with_sp_sec / gn, 3)
# Points lost on generic/specific confusions, and how many of them involve CAL-01 / WOP / ANS / OTH
for k in ['blend_sk', 'claude']:
    t1 = preds[k][:, 0]
    m = np.array([y[i] != t1[i] and 'NONE' not in (y[i], t1[i]) and G(y[i]) != G(t1[i]) for i in range(n)])
    res[f'{k}_gen_spec_points_lost_per_row'] = round(float((1 - S[k])[m].sum() / n), 4)
    res[f'{k}_gen_spec_rows_gold_in_top3'] = round(float(np.mean([y[i] in preds[k][i] for i in np.flatnonzero(m)])), 3)
both = np.array([y[i] != preds['claude'][i, 0] and y[i] != preds['blend_sk'][i, 0] and 'NONE' not in (y[i], preds['claude'][i, 0]) and G(y[i]) != G(preds['claude'][i, 0]) for i in range(n)])
res['rows_gen_spec_miss_for_claude_and_blend_both'] = int(both.sum())
json.dump(res, open(OUT / 'p2b_generic_specific.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
