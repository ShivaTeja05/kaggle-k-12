"""Where the blend loses to Claude: code-frequency buckets, transcript mismatch, model correlation, root-vs-secondary."""
import json, sys, difflib
from collections import Counter
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/recipe-audit')
from common import *
from evaluate_claude import claude_scores

R = {}
ref = np.load(OUT / 'blend_ref.npy'); bp = points(ref)
cl, _ = claude_scores(ROOT / 'runs/claude/dev_opus.jsonl', ids, classes, shares=(.6, .25, .15)); cp = points(cl)
R['claude_dev_map3'] = round(float(cp.mean()), 4); R['blend_dev_map3'] = round(float(bp.mean()), 4)
gold_n = Counter(fit.root_code)
pseudo = [json.loads(l) for l in (ROOT / 'data/v3/claude_unl_v4.jsonl').read_text().splitlines()]
ps_n = Counter(max(r['soft'], key=r['soft'].get) for r in pseudo)
def bucket(c):
    if c == 'NONE': return 'NONE'
    g = gold_n[c]
    return 'gold<=2' if g <= 2 else 'gold3-10' if g <= 10 else 'gold11-30' if g <= 30 else 'gold>30'
b = np.array([bucket(c) for c in truth])
R['gap_by_gold_count_bucket'] = {k: {'rows': int((b == k).sum()), 'blend': round(float(bp[b == k].mean()), 3),
    'claude': round(float(cp[b == k].mean()), 3), 'lost_points_vs_claude_per_301': round(float((cp - bp)[b == k].sum() / len(bp)), 4)}
    for k in ('NONE', 'gold<=2', 'gold3-10', 'gold11-30', 'gold>30')}
pb = np.array(['NONE' if c == 'NONE' else 'pseudo<=20' if ps_n[c] <= 20 else 'pseudo21-150' if ps_n[c] <= 150 else 'pseudo>150' for c in truth])
R['gap_by_pseudo_count_bucket'] = {k: {'rows': int((pb == k).sum()), 'blend': round(float(bp[pb == k].mean()), 3),
    'claude': round(float(cp[pb == k].mean()), 3), 'lost_points_vs_claude_per_301': round(float((cp - bp)[pb == k].sum() / len(bp)), 4)}
    for k in ('NONE', 'pseudo<=20', 'pseudo21-150', 'pseudo>150')}
R['dev_rows_true_code_absent_from_fit_gold'] = int(sum((truth != 'NONE') & np.array([gold_n[c] == 0 for c in truth])))
R['dev_rows_true_code_absent_from_fit_gold_and_pseudo'] = int(sum((truth != 'NONE') & np.array([gold_n[c] + ps_n[c] == 0 for c in truth])))
# Claude teacher bias: dev first-code counts vs truth for the codes that dominate pseudo labels
c1 = topk(cl, classes)[:, 0]; tc = Counter(truth); cc = Counter(c1)
R['claude_dev_top1_vs_truth_counts'] = {c: {'claude_top1': cc[c], 'truth': tc[c], 'claude_precision': round(float(np.mean(truth[c1 == c] == c)), 2) if cc[c] else None}
                                        for c in ('NONE', 'ANS-01', 'WOP-01', 'CAL-01', 'REL-01', 'OTH-00', 'INT-04', 'KPF-04')}
# Transcript mismatch: gold->tx loss for the same model (scale_100) vs similarity of tx to gold
g = points(load('s100_gold')); t = points(load('s100_tx'))
tx = json.loads((ROOT / 'runs/pod/tx/tx_dev.json').read_text())
strip = lambda s: '\n'.join(l.split('|', 1)[-1].strip() for l in str(s).splitlines() if l.strip())
sim = np.array([difflib.SequenceMatcher(None, strip(r.transcription_steps), strip(tx[r.trace_id])).ratio() for r in dev.itertuples()])
np.save(OUT / 'tx_similarity.npy', sim)
q = np.quantile(sim, [.25, .5, .75])
R['tx_vs_gold_char_similarity_quartiles'] = [round(float(x), 3) for x in q]
for lo, hi, name in ((0, q[0], 'worst_quartile'), (q[0], q[2], 'middle_half'), (q[2], 1.01, 'best_quartile')):
    m = (sim >= lo) & (sim < hi)
    R[f'gold_to_tx_loss_{name}'] = {'rows': int(m.sum()), 'gold_map3': round(float(g[m].mean()), 3), 'tx_map3': round(float(t[m].mean()), 3)}
R['s100_gold_minus_tx'] = {'delta': round(float((g - t).mean()), 4), 'ci95': [round(x, 4) for x in question_bootstrap(g - t)]}
g32, t32 = points(load('q32_gold')), points(load('q32_tx'))
R['q32_gold_minus_tx'] = {'delta': round(float((g32 - t32).mean()), 4), 'ci95': [round(x, 4) for x in question_bootstrap(g32 - t32)]}
# Model correlation: top1 agreement and oracle
M = {n: load(n) for n in ('q32_tx', 'q8_tx_e3', 'vl_distill', 's100_tx')}
tops = {n: topk(p, classes)[:, 0] for n, p in M.items()}
pts = {n: points(p) for n, p in M.items()}
names = list(M)
R['pairwise_top1_agreement'] = {f'{a}|{b}': round(float(np.mean(tops[a] == tops[b])), 3) for i, a in enumerate(names) for b in names[i + 1:]}
R['pairwise_rowpoint_corr'] = {f'{a}|{b}': round(float(np.corrcoef(pts[a], pts[b])[0, 1]), 3) for i, a in enumerate(names) for b in names[i + 1:]}
R['claude_rowpoint_corr_with_blend'] = round(float(np.corrcoef(cp, bp)[0, 1]), 3)
R['oracle_best_of_3_blend_members'] = round(float(np.maximum.reduce([pts['q32_tx'], pts['q8_tx_e3'], pts['vl_distill']]).mean()), 4)
R['blend_with_s100_instead_of_q8'] = round(float(points(geo([(M['q32_tx'], .35), (M['s100_tx'], .30), (M['vl_distill'], .35)])).mean()), 4)
# Root vs secondary confusion: blend top1 equals an annotated non-root major error of that row
sec = []
for r, top in zip(dev.itertuples(), topk(ref, classes)[:, 0]):
    f = json.loads(str(r.misconceptions).replace('NaN', 'null'))
    s = {x['category'] for x in f if x.get('severity') == 'major' and not x.get('root')} - {r.root_code}
    sec.append((bool(s), top in s, top != r.root_code))
sec = np.array(sec)
R['dev_rows_with_secondary_major'] = int(sec[:, 0].sum())
R['blend_top1_is_a_secondary_major_code'] = int(sec[:, 1].sum())
fsec = sum(1 for r in fit.itertuples() if {x['category'] for x in json.loads(str(r.misconceptions).replace('NaN', 'null')) if x.get('severity') == 'major' and not x.get('root')} - {r.root_code})
R['fit_rows_with_secondary_major'] = fsec
# Hierarchical decode check: family marginal accuracy vs flat
families = sorted({fam(c) for c in classes}); F = np.array([[fam(c) == f for f in families] for c in classes], float)
fm = ref @ F; fam_true = np.array([fam(c) for c in truth]); err = truth != 'NONE'
R['blend_family_marginal_top1_acc_err_rows'] = round(float(np.mean(np.array(families)[fm.argmax(1)][err] == fam_true[err])), 3)
R['blend_family_of_top_code_acc_err_rows'] = round(float(np.mean(np.array([fam(c) for c in topk(ref, classes)[:, 0]])[err] == fam_true[err])), 3)
R['blend_true_family_in_top3_families_err_rows'] = round(float(np.mean([fam_true[i] in np.array(families)[np.argsort(-fm[i])[:3]] for i in np.flatnonzero(err)])), 3)
R['blend_recall_at_3'] = round(float(np.mean([truth[i] in topk(ref, classes)[i] for i in range(len(truth))])), 3)
R['blend_recall_at_10'] = round(float(np.mean([truth[i] in np.asarray(classes)[np.argsort(-ref[i])[:10]] for i in range(len(truth))])), 3)
R['claude_recall_at_3'] = round(float(np.mean([truth[i] in topk(cl, classes)[i] for i in range(len(truth))])), 3)
(OUT / 'a5_gaps.json').write_text(json.dumps(R, indent=1, default=str))
print(json.dumps(R, indent=1, default=str))
