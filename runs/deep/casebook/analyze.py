"""Quantify the mechanism mix from labels.csv (manual labels) joined with all_dev_rows.csv. CPU only."""
import json, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from common import load, blend, boot_ci, ROOT
from k12.core import aligned_bundle, topk, row_scores
OUT = ROOT/'runs/deep/casebook'
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy()
d = pd.read_csv(OUT/'all_dev_rows.csv', keep_default_na=False)
assert (d.trace_id == dev.trace_id).all()
lab = pd.read_csv(OUT/'labels.csv', keep_default_na=False)
err = d[d.truth != d.blend_top3.str.split().str[0]]
assert set(lab.trace_id) == set(err.trace_id) and len(lab) == len(err) == 112, (len(lab), len(err))
d = d.merge(lab, on='trace_id', how='left').fillna({'mech': 'OK', 'mech2': '', 'note': ''})
N = len(d); res = {}
names = {'M1': 'did not solve / unanchored', 'M2': 'picked a real secondary error (wrong line)', 'M3': 'right line, wrong error type (generic vs specific / conceptual vs procedural)',
         'M4': 'confusable sibling code', 'M5': "question's habitual code not supported by the work", 'M6': 'reading error on the root span (machine transcript)',
         'M7': 'label ambiguity / annotator convention / label artifact', 'M8': 'NONE confusion'}
res['definitions'] = names
res['precedence'] = 'M6 > M8 > M2 > M7 > M5 > M4 > M3 > M1 (primary label); secondary label recorded in mech2'
def table(sub, tag):
    rows = []
    for m in sorted(names):
        s = sub[sub.mech == m]
        rows.append(dict(mech=m, rows=len(s), share_rows=round(len(s)/max(1, len(sub[sub.mech != 'OK'])), 3),
                         blend_map=round(s.s_blend.mean(), 3) if len(s) else None, claude_map=round(s.s_claude.mean(), 3) if len(s) else None,
                         blend_points_lost=round((1 - s.s_blend).sum()/N, 4), gap_points_vs_claude=round((s.s_claude - s.s_blend).sum()/N, 4),
                         truth_in_top3=round((s.rank_truth <= 3).mean(), 3) if len(s) else None, median_rank=float(s.rank_truth.median()) if len(s) else None,
                         top1_in_q_fit=round(s.blend_top1_in_q_fit_codes.astype(str).eq('True').mean(), 3) if len(s) else None,
                         top1_nonroot_annot=round(s.top1_is_nonroot_annot.astype(str).eq('True').mean(), 3) if len(s) else None))
    t = pd.DataFrame(rows); print(f'\n== {tag} (n err={len(sub[sub.mech != "OK"])})'); print(t.to_string(index=False)); return t.to_dict('records')
res['overall'] = table(d, 'all 112 top-1 misses')
for sl in ['seen_q_new_code', 'unseen_q', 'seen_pair', 'none']:
    res[sl] = table(d[d.slice == sl], sl)
# gap decomposition with question-grouped bootstrap CI per mechanism (delta = claude - blend on mech rows, 0 elsewhere)
g = d.question_id.to_numpy()
res['gap_ci'] = {}
for m in sorted(names):
    msk = (d.mech == m).to_numpy()
    mean, ci = boot_ci(np.where(msk, d.s_claude, 0), np.where(msk, d.s_blend, 0), g)
    res['gap_ci'][m] = {'gap': round(mean, 4), 'ci': [round(x, 4) for x in ci]}
tot, tci = boot_ci(d.s_claude, d.s_blend, g); res['gap_total'] = {'gap': round(tot, 4), 'ci': [round(x, 4) for x in tci]}
lost_on_top1_hits = (1 - d[d.mech == 'OK'].s_blend).sum()/N
res['blend_points_lost_on_top1_hits'] = round(lost_on_top1_hits, 4)
print('\ngap', res['gap_total'], json.dumps(res['gap_ci']))
# Grouped families
grp = {'Decision failures (M1,M3,M4,M5,M2)': ['M1', 'M2', 'M3', 'M4', 'M5'], 'Reading (M6)': ['M6'], 'Label (M7)': ['M7'], 'NONE (M8)': ['M8']}
res['grouped'] = {}
for k, ms in grp.items():
    s = d[d.mech.isin(ms)]
    res['grouped'][k] = dict(rows=len(s), rows_84=int((s.slice == 'seen_q_new_code').sum()), lost=round((1 - s.s_blend).sum()/N, 4), gap=round((s.s_claude - s.s_blend).sum()/N, 4))
print(json.dumps(res['grouped'], indent=1))
# Secondary-label co-occurrence
res['mech2_counts'] = {m: g2.mech2[g2.mech2 != ''].value_counts().to_dict() for m, g2 in d[d.mech != 'OK'].groupby('mech')}
# Reading check: does the 32B model on GOLD transcripts recover M6 rows? (same model/epoch, only the transcript differs)
B32tx = aligned_bundle(ROOT/'runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes)
B32g = aligned_bundle(ROOT/'runs/pod/text32/validation_gold_epoch_1.npz', ids, classes)
s_tx = row_scores(truth, topk(B32tx, classes)); s_g = row_scores(truth, topk(B32g, classes))
d['s32_tx'], d['s32_gold'] = s_tx, s_g
res['reading_check_32b'] = {}
for m in ['M6'] + [x for x in sorted(names) if x != 'M6'] + ['OK']:
    s = d[d.mech == m]
    res['reading_check_32b'][m] = dict(rows=len(s), s32_tx=round(s.s32_tx.mean(), 3), s32_gold=round(s.s32_gold.mean(), 3), delta_points=round((s.s32_gold - s.s32_tx).sum()/N, 4))
print('\n32B tx vs gold-transcript by mech'); print(pd.DataFrame(res['reading_check_32b']).T.to_string())
m6 = (d.mech == 'M6').to_numpy()
mean, ci = boot_ci(np.where(m6, s_g, 0), np.where(m6, s_tx, 0), g); res['reading_check_32b']['M6_delta_ci'] = [round(mean, 4), [round(x, 4) for x in ci]]
mean, ci = boot_ci(s_g, s_tx, g); res['reading_check_32b']['all_rows_delta_ci'] = [round(mean, 4), [round(x, 4) for x in ci]]
print('M6 delta', res['reading_check_32b']['M6_delta_ci'], 'all', res['reading_check_32b']['all_rows_delta_ci'])
# Automated corroboration of M6: normalised gold root span present in machine transcript?
tx = json.loads((ROOT/'runs/pod/tx/tx_dev.json').read_text())
def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    for a, b in [('−', '-'), ('×', 'x'), ('·', 'x'), ('÷', ':'), ('^', ''), ('(', ''), (')', ''), ('[', ''), (']', ''), ('{', ''), ('}', '')]: s = s.replace(a, b)
    return re.sub(r'\s+', '', s)
def span_in_tx(r):
    sp = norm(r.gold_root_span)
    if not sp or len(sp) < 3 or sp.startswith('table') or 'l1' in sp: return None
    return sp in norm(tx.get(r.trace_id, ''))
d['root_span_in_tx'] = [span_in_tx(r) for r in d.itertuples()]
res['span_in_tx'] = {m: dict(n=int(d[(d.mech == m)].root_span_in_tx.notna().sum()), rate=round(float(d[(d.mech == m) & d.root_span_in_tx.notna()].root_span_in_tx.astype(bool).mean()), 3)) for m in sorted(names) + ['OK'] if d[(d.mech == m) & d.root_span_in_tx.notna()].shape[0]}
print('\nroot span verbatim in machine tx', res['span_in_tx'])
# "Corrective" reading: machine transcript makes final answer match key while the gold does not / NONE predicted on error rows
err_rows = d[(d.truth != 'NONE')]
pn = err_rows[err_rows.blend_top3.str.split().str[0] == 'NONE']
res['none_on_error_rows'] = dict(rows=len(pn), by_mech=pn.mech.value_counts().to_dict(), final_correct=pn.final_answer_correct.value_counts().to_dict())
print('NONE top-1 on error rows', res['none_on_error_rows'])
# What the blend's wrong top-1 codes are
e = d[d.mech != 'OK'].copy(); e['top1'] = e.blend_top3.str.split().str[0]
generic = {'CAL-01', 'OTH-00', 'ANS-01', 'ANS-02', 'WOP-01', 'WOP-02', 'FRM-01', 'VIS-01', 'NOT-01', 'NONE'}
res['wrong_top1_codes'] = e.top1.value_counts().head(12).to_dict()
res['wrong_top1_generic_share'] = round(e.top1.isin(generic).mean(), 3)
res['truth_generic_share_on_misses'] = round(e.truth.isin(generic).mean(), 3)
# VIS-01 habit: blend top-1 VIS-01 precision on dev vs truth frequency
t1 = d.blend_top3.str.split().str[0]
for c in ['VIS-01', 'CAL-01', 'OTH-00', 'STA-01', 'FRM-01', 'ANS-01', 'NONE']:
    res.setdefault('top1_precision', {})[c] = dict(pred=int((t1 == c).sum()), correct=int(((t1 == c) & (d.truth == c)).sum()), truth=int((d.truth == c).sum()))
print('top1 precision', res['top1_precision'])
# 84-row slice summary incl. rows where top-1 is right
s84 = d[d.slice == 'seen_q_new_code']
res['slice84'] = dict(rows=len(s84), top1_hits=int((s84.mech == 'OK').sum()), mech=s84[s84.mech != 'OK'].mech.value_counts().to_dict(),
                      lost_total=round((1 - s84.s_blend).sum()/N, 4), gap_total=round((s84.s_claude - s84.s_blend).sum()/N, 4))
print('slice84', res['slice84'])
# Confidence by mechanism (top-1 prob)
d['p1'] = d.blend_p3.str.split().str[0].astype(float)
res['top1_prob_by_mech'] = d.groupby('mech').p1.median().round(3).to_dict()
print('median top1 prob', res['top1_prob_by_mech'])
d.drop(columns=[]).to_csv(OUT/'casebook_labelled.csv', index=False)
pass
# --- extra: question-prior shortcut precision; Claude restricted to blend top-10 per mechanism; we-win anatomy
from evaluate_claude import claude_scores
P = blend(ids, classes); C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
top10 = np.argsort(-P, 1)[:, :10]; mask = np.full_like(C, 1e-9); np.put_along_axis(mask, top10, 1, 1)
Cr = C*mask; s_cr = row_scores(truth, topk(Cr/Cr.sum(1, keepdims=True), classes)); d['s_claude_top10'] = s_cr
res['claude_restricted_top10_by_mech'] = d.groupby('mech').s_claude_top10.mean().round(3).to_dict()
seen = d.slice.isin(['seen_q_new_code', 'seen_pair']) | ((d.slice == 'none') & (d.q_fit_codes != ''))
inq = d.blend_top1_in_q_fit_codes.astype(str).eq('True')
t1 = d.blend_top3.str.split().str[0]
res['q_prior_shortcut'] = {'seen_q_rows': int(seen.sum()),
    'top1_in_q_fit': dict(n=int((seen & inq).sum()), top1_acc=round(float((t1 == d.truth)[seen & inq].mean()), 3)),
    'top1_not_in_q_fit': dict(n=int((seen & ~inq).sum()), top1_acc=round(float((t1 == d.truth)[seen & ~inq].mean()), 3))}
nn = seen & inq & (t1 != 'NONE')
res['q_prior_shortcut']['non_NONE_top1_in_q_fit'] = dict(n=int(nn.sum()), top1_acc=round(float((t1 == d.truth)[nn].mean()), 3),
    wrong_of_which_M5=int((nn & (d.mech == 'M5')).sum()))
print('claude top10-restricted by mech', res['claude_restricted_top10_by_mech']); print('q prior shortcut', res['q_prior_shortcut'])
w = pd.read_csv(OUT/'we_win_labels.csv').merge(d, on='trace_id')
res['we_win'] = dict(rows=len(w), by_strength=w.strength.value_counts().to_dict(), by_slice=w.slice.value_counts().to_dict(),
    points=round((w.s_blend - w.s_claude).sum()/N, 4), by_strength_points={k: round((g2.s_blend - g2.s_claude).sum()/N, 4) for k, g2 in w.groupby('strength')},
    s1_top1_is_q_modal=int(sum(1 for r in w[w.strength == 'S1'].itertuples() if r.q_fit_codes.split()[0].split(':')[0] == r.truth)),
    claude_top1_on_wins=w.claude_top3.str.split().str[0].value_counts().head(6).to_dict(),
    claude_top1_NONE_or_OTH=int(w.claude_top3.str.split().str[0].isin(['NONE', 'OTH-00']).sum()))
print('we win', res['we_win'])
res['ok_rows_gap'] = round((d[d.mech == 'OK'].s_claude - d[d.mech == 'OK'].s_blend).sum()/N, 4)
print('gap on top1-hit rows', res['ok_rows_gap'])
d.to_csv(OUT/'casebook_labelled.csv', index=False)
pass
json.dump(res, open(OUT/'mechanisms.json', 'w'), indent=1, default=str)
