"""Label distributions: gold fit vs dev vs Claude pseudo; Claude rank calibration on dev; predicted-code skew."""
import json, sys
from collections import Counter
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/recipe-audit')
from common import *

R = {}
gold_all = Counter(train.root_code); R['train_counts_hist'] = dict(Counter(gold_all.values()).most_common(12))
R['train_none_share'] = gold_all['NONE'] / len(train)
pseudo = [json.loads(l) for l in (ROOT / 'data/v3/claude_unl_v4.jsonl').read_text().splitlines()]
pf = Counter(max(r['soft'], key=r['soft'].get) for r in pseudo)
allc = claude_rank(ROOT / 'runs/claude/unl_all_v4.jsonl')
af = Counter(v[0] for v in allc.values())
R['claude_all_rows'] = len(allc); R['claude_all_none_share'] = af['NONE'] / len(allc)
R['pseudo_v4_rows'] = len(pseudo); R['pseudo_v4_none_share'] = pf['NONE'] / len(pseudo)
nonnone_gold = {c: n for c, n in gold_all.items() if c != 'NONE'}
nonnone_ps = {c: n for c, n in pf.items() if c != 'NONE'}
tg, tp = sum(nonnone_gold.values()), sum(nonnone_ps.values())
R['pseudo_error_codes_top10_share'] = [(c, round(n / tp, 3), round(nonnone_gold.get(c, 0) / tg, 3)) for c, n in Counter(nonnone_ps).most_common(10)]
R['pseudo_top5_error_codes_cum_share'] = round(sum(n for _, n in Counter(nonnone_ps).most_common(5)) / tp, 3)
R['gold_top5_error_codes_cum_share'] = round(sum(n for _, n in Counter(nonnone_gold).most_common(5)) / tg, 3)
R['codes_with_<=5_pseudo_first'] = sorted(c for c in classes if c != 'NONE' and pf[c] <= 5)
R['codes_absent_in_pseudo'] = sorted(c for c in classes if c != 'NONE' and pf[c] == 0)
# Effective training mass of the 8B/32B text reasoner: 1199 gold (w=1, one-hot) + pseudo (w=.8, soft .7/.2/.1)
mass = np.zeros(len(classes))
for c in fit.root_code: mass[ci[c]] += 1
gold_mass = mass.sum()
for r in pseudo:
    for c, s in r['soft'].items(): mass[ci[c]] += r['weight'] * s
R['gold_share_of_classification_loss_mass'] = round(gold_mass / mass.sum(), 4)
pi_train = mass / mass.sum()
pi_gold = np.bincount([ci[c] for c in fit.root_code], minlength=len(classes)) + .5; pi_gold /= pi_gold.sum()
R['KL_gold_vs_trainmix'] = float(np.sum(pi_gold * np.log(pi_gold / pi_train.clip(1e-9))))
ratio = pi_train / pi_gold
R['train_mix_over_gold_prior_ratio_top'] = sorted([(classes[i], round(float(ratio[i]), 2)) for i in range(len(classes))], key=lambda x: -x[1])[:8]
R['train_mix_over_gold_prior_ratio_bottom'] = sorted([(classes[i], round(float(ratio[i]), 3)) for i in range(len(classes))], key=lambda x: x[1])[:8]
np.save(OUT / 'pi_train_mix.npy', pi_train)

# Claude rank calibration on dev: where is the gold code among Claude's ranks?
dc = claude_rank(ROOT / 'runs/claude/dev_opus.jsonl')
pos = Counter()
for t, y in zip(ids, truth):
    r = dc.get(t, [None] * 3)
    pos[r.index(y) + 1 if y in r else 0] += 1
n = len(ids)
R['claude_dev_gold_at_rank'] = {'1': pos[1] / n, '2': pos[2] / n, '3': pos[3] / n, 'miss': pos[0] / n}
R['claude_dev_map3_from_ranks'] = (pos[1] + pos[2] / 2 + pos[3] / 3) / n
# conditional on first==NONE vs error
for kind in ('NONE', 'ERR'):
    sel = [(t, y) for t, y in zip(ids, truth) if (dc[t][0] == 'NONE') == (kind == 'NONE')]
    p = Counter((dc[t].index(y) + 1) if y in dc[t] else 0 for t, y in sel)
    R[f'claude_dev_rank_given_first_{kind}'] = {'rows': len(sel), **{str(k): round(p[k] / len(sel), 3) for k in (1, 2, 3, 0)}}
# Predicted top-1 distribution of models vs truth on dev
pred = {}
for name in ('q32_tx', 'q8_tx_e3', 'vl_distill', 's0_tx', 's100_tx'):
    top = topk(load(name), classes)[:, 0]
    c = Counter(top)
    pred[name] = {'none_top1_share': round(c['NONE'] / n, 3), 'distinct_top1_codes': len(c),
                  'most_common': c.most_common(5)}
blend = geo([(load('q32_tx'), .35), (load('q8_tx_e3'), .30), (load('vl_distill'), .35)])
np.save(OUT / 'blend_ref.npy', blend)
top = topk(blend, classes)[:, 0]; c = Counter(top)
pred['blend'] = {'none_top1_share': round(c['NONE'] / n, 3), 'distinct_top1_codes': len(c), 'most_common': c.most_common(6)}
tc = Counter(truth); pred['truth'] = {'none_share': round(tc['NONE'] / n, 3), 'distinct': len(tc), 'most_common': tc.most_common(5)}
# over-prediction of Claude-frequent codes: precision for blend top-1 by code
over = []
for code, k in Counter(top).most_common(8):
    over.append((code, k, int(tc[code]), int(sum((top == code) & (truth == code)))))
pred['blend_top1_count_vs_true_count_vs_correct'] = over
R['dev_prediction_skew'] = pred
# Family-level error decomposition of the blend vs Claude
cl = np.array([dc[t][0] for t in ids])
for name, first in (('blend', top), ('claude', cl)):
    err = truth != 'NONE'
    R[f'{name}_decomp'] = {
        'top1_acc': round(float(np.mean(first == truth)), 3),
        'family_top1_acc_on_error_rows': round(float(np.mean([fam(a) == fam(b) for a, b in zip(first[err], truth[err])])), 3),
        'false_alarm_on_none': int(sum(~err & (first != 'NONE'))),
        'missed_error_said_none': int(sum(err & (first == 'NONE'))),
        'wrong_family': int(sum(err & (first != 'NONE') & np.array([fam(a) != fam(b) for a, b in zip(first, truth)]))),
        'right_family_wrong_code': int(sum(err & (first != truth) & np.array([fam(a) == fam(b) for a, b in zip(first, truth)])))}
(OUT / 'a2_labels.json').write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
