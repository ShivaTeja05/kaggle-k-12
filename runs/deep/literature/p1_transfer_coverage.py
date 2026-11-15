"""Literature-lens probe 1 (CPU, aggregate only).

Q: For dev error rows whose gold code is NEW for the question (vs fit gold), does the training data
contain that code at all on OTHER questions (same topic / same section / anywhere)?  This bounds what
label-semantics / cross-question transfer methods (description cross-encoders, counterfactual
augmentation, PoE/JTT debiasing) can learn from data we already have.
Also: size of the 'bias-conflicting' group (label != question's modal code) that JTT / group-DRO
style reweighting would act on, in fit gold and in Claude-labelled unlabelled rows.
"""
import json
import numpy as np, pandas as pd
from common import load, blend, boot_ci, ROOT
from k12.core import row_scores, topk

tr, tax, classes, folds, dev, fit = load()
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
lab = pd.read_csv(ROOT/'data/claude_dataset/labels.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
unl = lab[lab.split == 'unlabelled'].drop_duplicates('trace_id', keep='last')
unl = unl[unl.first_code.isin(set(classes))]
# topic for unlabelled rows comes from labels.csv topic_name
fitq = fit.groupby('question_id').root_code.apply(set).to_dict()
P = blend(dev.trace_id.tolist(), classes)
pts = row_scores(dev.root_code.to_numpy(), topk(P, classes))

rows = []
for i, r in dev.iterrows():
    y = r.root_code
    if y == 'NONE':
        continue
    seenq = r.question_id in fitq
    if seenq and y in fitq[r.question_id]:
        grp = 'seen_pair'
    elif seenq:
        grp = 'new_code_seen_q'
    else:
        grp = 'unseen_q'
    oth_fit = fit[fit.question_id != r.question_id]
    oth_unl = unl[unl.question_id != r.question_id]
    same_q_unl = unl[unl.question_id == r.question_id]
    rows.append(dict(
        grp=grp, map=pts[i],
        fit_same_topic=int(((oth_fit.topic_name == r.topic_name) & (oth_fit.root_code == y)).sum()),
        fit_any=int((oth_fit.root_code == y).sum()),
        unl_same_q_first=int((same_q_unl.first_code == y).sum()),
        unl_same_q_top3=int(((same_q_unl.first_code == y) | (same_q_unl.second_code == y) | (same_q_unl.third_code == y)).sum()),
        unl_same_topic_first=int(((oth_unl.topic_name == r.topic_name) & (oth_unl.first_code == y)).sum()),
        unl_any_first=int((oth_unl.first_code == y).sum()),
        section_generic=sec.get(y, '').split(' ')[0],
    ))
d = pd.DataFrame(rows)
out = {'n_by_group': d.grp.value_counts().to_dict(), 'blend_map_by_group': d.groupby('grp').map.mean().round(3).to_dict()}
cov = {}
for g, gd in d.groupby('grp'):
    cov[g] = {
        'n': len(gd),
        'share_code_in_fit_other_q_same_topic': round(float((gd.fit_same_topic > 0).mean()), 3),
        'share_code_in_fit_any_other_q': round(float((gd.fit_any > 0).mean()), 3),
        'median_fit_any_count': float(gd.fit_any.median()),
        'share_claude_same_q_first': round(float((gd.unl_same_q_first > 0).mean()), 3),
        'share_claude_same_q_top3': round(float((gd.unl_same_q_top3 > 0).mean()), 3),
        'share_claude_other_q_same_topic_first': round(float((gd.unl_same_topic_first > 0).mean()), 3),
        'share_claude_other_q_same_topic_first_ge10': round(float((gd.unl_same_topic_first >= 10).mean()), 3),
        'share_claude_any_other_q_first': round(float((gd.unl_any_first > 0).mean()), 3),
        'median_claude_any_other_q_first': float(gd.unl_any_first.median()),
    }
out['coverage'] = cov
# Generic codes (CAL/WOP/ANS/FRM/NOT/CHK/UNT/VIS/OTH) vs topic-specific among new-code rows
generic = {'CAL', 'WOP', 'ANS', 'FRM', 'NOT', 'CHK', 'UNT', 'VIS', 'OTH', 'REL'}
nd = d[d.grp != 'seen_pair'].copy()
dev_err = dev[dev.root_code != 'NONE']
nd['generic'] = [c.split('-')[0] in generic for c in dev_err.root_code[[g != 'seen_pair' for g in d.grp]]]
out['new_code_rows_generic_share'] = round(float(nd.generic.mean()), 3)
out['new_code_rows_map_generic_vs_specific'] = nd.groupby('generic').map.mean().round(3).to_dict()

# bias-conflicting group sizes (leave-one-out modal code per question)
def conflict_share(frame, col):
    s = 0; n = 0
    for q, g in frame.groupby('question_id'):
        if len(g) < 2:
            continue
        vc = g[col].value_counts()
        for c in g[col]:
            loo = vc.copy(); loo[c] -= 1
            mode = loo.idxmax() if loo.max() > 0 else None
            s += int(mode is not None and c != mode); n += 1
    return round(s / max(n, 1), 3), n
out['bias_conflicting_share_fit_gold'] = conflict_share(fit, 'root_code')
out['bias_conflicting_share_claude_unl_first'] = conflict_share(unl, 'first_code')
fit_err = fit[fit.root_code != 'NONE']
out['bias_conflicting_share_fit_gold_errors_only'] = conflict_share(fit_err, 'root_code')
json.dump(out, open(ROOT/'runs/deep/literature/p1_transfer_coverage.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
