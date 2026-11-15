"""Teacher-gap analysis: why Claude Opus (dev 0.819) beats the 0.718 geometric blend.

CPU only. Reads dev bundles, Claude's dev answers and train.csv; writes rows.csv + gap.json here.
"""
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from evaluate_claude import claude_scores

OUT = Path('runs/arch/teacher-gap')
train = read_csv('data/raw/train.csv')
tax = read_csv('data/raw/taxonomy_v1.csv')
classes = label_space(tax)
ci = {c: i for i, c in enumerate(classes)}
folds = make_folds(train)
dev = train.loc[folds == 0].reset_index(drop=True)
fit = train.loc[folds != 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
truth = dev.root_code.to_numpy()
fam = lambda c: 'NONE' if c == 'NONE' else c.split('-')[0].upper()
section = dict(zip(tax.id, tax.section)) | {'NONE': 'NONE'}

B32 = aligned_bundle('runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes)
B8 = aligned_bundle('runs/pod/tx/text_tx_dev_e3.npz', ids, classes)
BVL = aligned_bundle('runs/pod/dev-distill/validation_epoch_2.npz', ids, classes)
blend = normalize(np.exp(.35 * np.log(B32) + .30 * np.log(B8) + .35 * np.log(BVL)))
gold32 = aligned_bundle('runs/pod/text32/validation_gold_epoch_1.npz', ids, classes)
gold8_100 = aligned_bundle('runs/pod/scaling/scale_100/validation_gold_epoch_1.npz', ids, classes)
claude, _ = claude_scores('runs/claude/dev_opus.jsonl', ids, classes)
answers = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open('runs/claude/dev_opus.jsonl')}
tx = json.loads(Path('runs/pod/tx/tx_dev.json').read_text())

rb, rc = topk(blend, classes), topk(claude, classes)
sb, sc = row_scores(truth, rb), row_scores(truth, rc)
print(f'blend {sb.mean():.4f} claude {sc.mean():.4f}')


def boot(delta, groups=dev.question_id.to_numpy(), n=4000, seed=2026):
    """Paired bootstrap over questions of a per-row delta."""
    g = pd.DataFrame({'q': groups, 'd': delta}).groupby('q').d.agg(['sum', 'size']).to_numpy()
    s = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    b = g[s, 0].sum(1) / g[s, 1].sum(1)
    return [round(float(x), 4) for x in np.quantile(b, [.025, .975])]


def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = s.replace('−', '-').replace('×', 'x').replace('·', 'x').replace('÷', ':')
    return re.sub(r'\s+', '', s)


def lines(text):
    out = {}
    for ln in str(text).splitlines():
        m = re.match(r'\s*(L\d+)\s*\|?\s*(.*)', ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def claude_root_text(ans):
    rl = str(ans.get('root_line', '')).strip()
    m = re.match(r'(L\d+)\s*\|?\s*(.*)', rl)
    if not m:
        return None, rl
    tag, rest = m.group(1), m.group(2)
    if rest.strip():
        return tag, rest
    return tag, lines(ans.get('transcription', '')).get(tag, '')


train_freq = Counter(fit.root_code)
rows = []
for i, r in enumerate(dev.itertuples(index=False)):
    ann = json.loads(r.misconceptions or '[]')
    root = [a for a in ann if a['root']]
    nonroot_codes = {a['category'] for a in ann if not a['root']}
    ans = answers[r.trace_id]
    tag, ctext = claude_root_text(ans)
    rec = dict(trace_id=r.trace_id, question_id=r.question_id, grade=r.grade, topic=r.topic_name, truth=truth[i],
               final_answer_correct=r.final_answer_correct, train_freq=train_freq[truth[i]],
               blend=' '.join(rb[i]), claude=' '.join(rc[i]), s_blend=sb[i], s_claude=sc[i],
               m32=' '.join(topk(B32[i:i + 1], classes)[0]), m8=' '.join(topk(B8[i:i + 1], classes)[0]),
               mvl=' '.join(topk(BVL[i:i + 1], classes)[0]),
               s_gold32=row_scores(truth[i:i + 1], topk(gold32[i:i + 1], classes))[0],
               s_gold8=row_scores(truth[i:i + 1], topk(gold8_100[i:i + 1], classes))[0],
               p_truth_blend=blend[i, ci[truth[i]]], rank_truth_blend=int((blend[i] > blend[i, ci[truth[i]]]).sum()) + 1,
               blend_top1_nonroot_annot=rb[i][0] in nonroot_codes and rb[i][0] != truth[i],
               claude_top1_nonroot_annot=rc[i][0] in nonroot_codes and rc[i][0] != truth[i],
               n_major_nonroot=sum(a['severity'] == 'major' and not a['root'] for a in ann),
               claude_root_line=str(ans.get('root_line', ''))[:80], claude_reason=ans.get('reasoning', ''))
    # reading quality of the machine transcript vs gold
    rec['tx_sim'] = SequenceMatcher(None, norm(tx[r.trace_id]), norm(r.transcription_steps)).ratio()
    rec['claude_tx_sim'] = SequenceMatcher(None, norm(ans.get('transcription', '')), norm(r.transcription_steps)).ratio()
    # root-line agreement with the annotator's root step (content based, line numbers can shift)
    if root:
        g = root[0]
        gline = lines(r.transcription_steps).get(g['step'], '')
        rec['gold_step'], rec['gold_span'] = g['step'], g['span']
        if tag is None:
            rec['root_loc'] = 'claude_said_none'
        else:
            span, ct = norm(g['span']), norm(ctext)
            hit = (span and span in ct) or (ct and ct in norm(gline)) or SequenceMatcher(None, ct, norm(gline)).ratio() >= .6
            rec['root_loc'] = 'match' if hit else ('same_tag_only' if tag == g['step'] else 'other_line')
    else:
        rec['root_loc'] = 'gold_none'
        rec['claude_root_line_on_none'] = tag is not None
    ca = norm(r.correct_answer).strip('$')
    nums = re.findall(r'-?\d+(?:[.,]\d+)?', ca)
    rec['reason_mentions_key'] = bool(nums) and all(n in norm(ans.get('reasoning', '')) for n in nums[:2])
    rows.append(rec)
df = pd.DataFrame(rows)


def stage(top1, t):
    if t == 'NONE':
        return 'ok' if top1 == 'NONE' else 'false_alarm'
    if top1 == 'NONE':
        return 'missed_error'
    if top1 == t:
        return 'ok'
    if fam(top1) == fam(t):
        return 'sibling_same_family'
    if section[top1] == section[t]:
        return 'other_family_same_section'
    return 'other_section'


df['stage_blend'] = [stage(a.split()[0], t) for a, t in zip(df.blend, df.truth)]
df['stage_claude'] = [stage(a.split()[0], t) for a, t in zip(df.claude, df.truth)]
df['outcome'] = np.select([(df.s_claude > df.s_blend), (df.s_claude < df.s_blend)], ['claude_better', 'blend_better'], 'tie')
df.to_csv(OUT / 'rows.csv', index=False)

res = {'blend_map3': round(sb.mean(), 4), 'claude_map3': round(sc.mean(), 4),
       'claude_minus_blend': round((sc - sb).mean(), 4), 'claude_minus_blend_ci95': boot(sc - sb),
       'outcome_counts': df.outcome.value_counts().to_dict(),
       'points_won_by_claude': round(float((sc - sb)[sc > sb].sum()), 2),
       'points_won_by_blend': round(float((sb - sc)[sb > sc].sum()), 2)}

# 1. Error-stage decomposition: MAP points lost by top-1 stage error type, both systems
def lost(stage_col, score):
    return {k: {'rows': int((df[stage_col] == k).sum()), 'map_points_lost_per_301': round(float((1 - df[score])[df[stage_col] == k].sum() / len(df)), 4)}
            for k in ['false_alarm', 'missed_error', 'sibling_same_family', 'other_family_same_section', 'other_section', 'ok']}
res['stage_blend'] = lost('stage_blend', 's_blend')
res['stage_claude'] = lost('stage_claude', 's_claude')

# Where does the gap come from: per (blend stage) the net Claude gain
g = df.assign(d=sc - sb).groupby('stage_blend').d.agg(['sum', 'size'])
res['net_gain_by_blend_stage'] = {k: {'rows': int(v['size']), 'net_map_gain': round(v['sum'] / len(df), 4)} for k, v in g.iterrows()}
g = df.assign(d=sc - sb, kind=np.where(df.truth == 'NONE', 'truth_NONE', 'truth_error')).groupby('kind').d.agg(['sum', 'size'])
res['net_gain_by_truth_kind'] = {k: {'rows': int(v['size']), 'net_map_gain': round(v['sum'] / len(df), 4), 'claude_map': round(float(df.s_claude[(df.truth == 'NONE') == (k == 'truth_NONE')].mean()), 4), 'blend_map': round(float(df.s_blend[(df.truth == 'NONE') == (k == 'truth_NONE')].mean()), 4)} for k, v in g.iterrows()}

# 2. Rare codes: gap by training frequency of the true code (non-dev rows)
bins = pd.cut(df.train_freq, [-1, 0, 5, 20, 60, 10000], labels=['0', '1-5', '6-20', '21-60', 'NONE(>60)'])
res['by_train_freq'] = {str(k): {'rows': int(len(v)), 'blend': round(v.s_blend.mean(), 4), 'claude': round(v.s_claude.mean(), 4),
                                 'net_gain_per_301': round(float((v.s_claude - v.s_blend).sum() / len(df)), 4)}
                        for k, v in df.groupby(bins, observed=True)}

# 3. Root localisation: Claude root line vs annotator root step
err = df[df.truth != 'NONE']
res['claude_root_loc'] = err.root_loc.value_counts().to_dict()
res['claude_root_loc_vs_correct'] = {k: {'rows': int(len(v)), 'claude_top1_acc': round(float((v.claude.str.split().str[0] == v.truth).mean()), 3),
                                         'claude_map': round(v.s_claude.mean(), 3), 'blend_map': round(v.s_blend.mean(), 3)}
                                     for k, v in err.groupby('root_loc')}
res['nonroot_annotated_code_as_top1'] = {'blend': int(df.blend_top1_nonroot_annot.sum()), 'claude': int(df.claude_top1_nonroot_annot.sum()),
                                         'error_rows_with_other_major_annotations': int((err.n_major_nonroot > 0).sum())}
multi = err[err.n_major_nonroot > 0]
single = err[err.n_major_nonroot == 0]
res['multi_major_error_rows'] = {'rows': len(multi), 'blend': round(multi.s_blend.mean(), 4), 'claude': round(multi.s_claude.mean(), 4),
                                 'single_major_rows': len(single), 'blend_single': round(single.s_blend.mean(), 4), 'claude_single': round(single.s_claude.mean(), 4)}

# 4. Reading: is the gap explained by transcript quality? Does gold reading fix the rows Claude wins?
cw = df[df.outcome == 'claude_better']
bw = df[df.outcome == 'blend_better']
res['reading'] = {'tx_sim_mean_all': round(df.tx_sim.mean(), 3), 'tx_sim_claude_better_rows': round(cw.tx_sim.mean(), 3),
                  'tx_sim_other_rows': round(df[df.outcome != 'claude_better'].tx_sim.mean(), 3),
                  'claude_tx_sim_mean': round(df.claude_tx_sim.mean(), 3),
                  'gold32_map_all': round(df.s_gold32.mean(), 4), 'tx32_map_all': round(float(row_scores(truth, topk(B32, classes)).mean()), 4),
                  'gold8_100_map_all': round(df.s_gold8.mean(), 4),
                  'claude_better_rows_fixed_by_gold32': int((cw.s_gold32 >= cw.s_claude).sum()),
                  'claude_better_rows': len(cw)}
q = pd.qcut(df.tx_sim, 3, labels=['worst', 'mid', 'best'])
res['gap_by_tx_quality_tercile'] = {str(k): {'rows': int(len(v)), 'tx_sim': round(v.tx_sim.mean(), 3), 'blend': round(v.s_blend.mean(), 4), 'claude': round(v.s_claude.mean(), 4)} for k, v in df.groupby(q, observed=True)}

# 5. Answer key use: does final_answer_correct separate the systems on NONE detection?
res['by_final_answer_correct'] = {str(k): {'rows': int(len(v)), 'none_share': round(float((v.truth == 'NONE').mean()), 3),
                                           'blend': round(v.s_blend.mean(), 4), 'claude': round(v.s_claude.mean(), 4),
                                           'blend_false_alarm': int((v.stage_blend == 'false_alarm').sum()), 'claude_false_alarm': int((v.stage_claude == 'false_alarm').sum()),
                                           'blend_missed': int((v.stage_blend == 'missed_error').sum()), 'claude_missed': int((v.stage_claude == 'missed_error').sum())}
                                for k, v in df.groupby('final_answer_correct')}

# 6. Counterfactual "transplants" (no tuning): give the blend one Claude capability at a time.
def transplant(name, p):
    s = row_scores(truth, topk(p, classes))
    res.setdefault('transplants', {})[name] = {'map3': round(float(s.mean()), 4), 'delta_vs_blend': round(float((s - sb).mean()), 4), 'ci95': boot(s - sb)}

c1 = rc[:, 0]
isnone = np.array([c == 'NONE' for c in c1])
famidx = {f: np.array([fam(c) == f for c in classes]) for f in set(map(fam, classes))}
secidx = {s: np.array([section[c] == s for c in classes]) for s in set(section.values())}
p = blend.copy()  # (a) NONE/error gate from Claude
p[isnone, ci['NONE']] = 1.; p[~isnone, ci['NONE']] *= 1e-3
transplant('A_claude_none_gate', normalize(p))
p = blend.copy()  # (b) Claude's top-1 family (section-level for a coarser variant), blend ranks inside it
for i, c in enumerate(c1):
    keep = famidx[fam(c)].copy(); keep[ci['NONE']] = True
    p[i, ~keep] *= 1e-3
transplant('B_claude_family_gate_incl_none', normalize(p))
p = blend.copy()
for i, c in enumerate(c1):
    keep = secidx[section[c]].copy(); keep[ci['NONE']] = True
    p[i, ~keep] *= 1e-3
transplant('C_claude_section_gate', normalize(p))
p = blend.copy()  # (d) Claude's three codes as a candidate set, blend reorders them
for i in range(len(p)):
    keep = np.zeros(len(classes), bool); keep[[ci[c] for c in rc[i]]] = True
    p[i, ~keep] *= 1e-3
transplant('D_claude_candidates_blend_order', normalize(p))
p = blend.copy()  # (e) blend candidates, Claude reorders: Claude rank wins where it overlaps
for i in range(len(p)):
    for k, c in enumerate(rc[i]):
        if c in rb[i]:
            p[i, ci[c]] = 10. - k
transplant('E_blend_candidates_claude_order', normalize(p))
p = blend.copy()  # (f) within-family code choice from Claude, given the blend's family
for i in range(len(p)):
    f0 = fam(rb[i][0])
    for k, c in enumerate(rc[i]):
        if fam(c) == f0:
            p[i, ci[c]] = max(p[i, ci[c]], p[i].max() * (1.01 - .001 * k)); break
transplant('F_claude_sibling_choice_inside_blend_family', normalize(p))
for w in (.5, .7):
    transplant(f'G_geometric_claude_{w}', normalize(claude ** w * blend ** (1 - w)))

# 7. Rows where truth is not even in the blend's top-10 (knowledge / coverage gap)
res['truth_rank_in_blend'] = {'top3': int((df.rank_truth_blend <= 3).sum()), '4-10': int(((df.rank_truth_blend > 3) & (df.rank_truth_blend <= 10)).sum()),
                              '>10': int((df.rank_truth_blend > 10).sum())}
res['claude_better_truth_rank_in_blend'] = cw.rank_truth_blend.describe().round(2).to_dict()

# 8. Confusion pairs Claude resolves (blend top1 -> truth) on rows Claude wins
res['claude_better_pairs'] = Counter(f"{a.split()[0]}->{t}" for a, t in zip(cw.blend, cw.truth)).most_common(25)
res['blend_better_pairs'] = Counter(f"{a.split()[0]}->{t}" for a, t in zip(bw.claude, bw.truth)).most_common(25)
res['claude_better_stage'] = cw.stage_blend.value_counts().to_dict()
res['blend_better_stage_of_claude'] = bw.stage_claude.value_counts().to_dict()
res['reason_mentions_key_rate'] = {'claude_top1_right': round(float(df.reason_mentions_key[rc[:, 0] == truth].mean()), 3),
                                   'claude_top1_wrong': round(float(df.reason_mentions_key[rc[:, 0] != truth].mean()), 3)}
(OUT / 'gap.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
print(json.dumps(res, indent=1, default=str))
