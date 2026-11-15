from common import *
import json, re, collections
tr, tax, classes, folds, dev, fit = load()
sec = dict(zip(tax.id, tax.section)); sev = dict(zip(tax.id, tax.default_severity)); nat = dict(zip(tax.id, tax.nature))
M = [json.loads(x) if x else [] for x in tr.misconceptions]
out = {}
out['n_annotations_per_row'] = dict(collections.Counter(len(m) for m in M))
nroot = [sum(a.get('root') is True for a in m) for m in M]
out['n_root_annotations_per_row'] = dict(collections.Counter(nroot))
# consistency root annotation category vs root_code
agree = [any(a['category'] == c and a.get('root') for a in m) for m, c in zip(M, tr.root_code) if c != 'NONE']
out['err_rows_root_annot_matches_root_code'] = float(np.mean(agree))
none_rows = [(m, f) for m, c, f in zip(M, tr.root_code, tr.final_answer_correct) if c == 'NONE']
out['NONE_rows_with_nonroot_annotations'] = int(sum(len(m) > 0 for m, _ in none_rows))
out['NONE_nonroot_categories_top'] = collections.Counter(a['category'] for m, _ in none_rows for a in m).most_common(10)
out['NONE_nonroot_severity'] = dict(collections.Counter(a['severity'] for m, _ in none_rows for a in m))
def lineno(s):
    x = re.findall(r'L(\d+)', str(s)); return int(x[0]) if x else None
nlines = [len(str(t).splitlines()) for t in tr.transcription_steps]
pos, first, multi, sec_same, cat_same, cnt_distinct = [], [], [], [], [], []
nonroot_cats = collections.Counter(); pair = collections.Counter()
err_sev = collections.Counter(); nonroot_sev = collections.Counter()
for m, c, n in zip(M, tr.root_code, nlines):
    if c == 'NONE': continue
    roots = [a for a in m if a.get('root')]
    others = [a for a in m if not a.get('root')]
    for a in roots: err_sev[a['severity']] += 1
    for a in others: nonroot_sev[a['severity']] += 1
    if roots:
        rl = lineno(roots[0]['step'])
        if rl and n: pos.append(rl / n)
        ol = [lineno(a['step']) for a in others if lineno(a['step'])]
        if ol and rl: first.append(rl <= min(ol))
    if others:
        multi.append(1)
        for a in others:
            nonroot_cats[a['category']] += 1; pair[(c, a['category'])] += 1
            cat_same.append(a['category'] == c)
            sec_same.append(sec.get(a['category']) == sec.get(c))
    else:
        multi.append(0)
    cnt_distinct.append(len({a['category'] for a in m}))
out['err_rows_with_secondary_annotations_share'] = float(np.mean(multi))
out['err_rows_distinct_categories_mean'] = float(np.mean(cnt_distinct))
out['root_relative_line_position_quartiles'] = np.quantile(pos, [.1, .25, .5, .75, .9]).round(3).tolist()
out['root_is_earliest_annotated_error_share(rows with secondary)'] = float(np.mean(first))
out['secondary_same_code_as_root_share'] = float(np.mean(cat_same))
out['secondary_same_section_as_root_share'] = float(np.mean(sec_same))
out['root_severity'] = dict(err_sev); out['secondary_severity'] = dict(nonroot_sev)
out['secondary_categories_top'] = nonroot_cats.most_common(12)
out['root_secondary_pairs_top'] = [(f'{a}->{b}', n) for (a, b), n in pair.most_common(15)]
# code frequency structure
vc = tr.root_code.value_counts()
out['class_counts_hist'] = {int(k): v for k, v in collections.Counter(vc.values).items()}
out['codes_in_taxonomy_never_in_train'] = sorted(set(classes) - set(vc.index))
# ---- dev blend: where are lost points?
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
top = topk(P, classes); pts = row_scores(y, top)
Md = [json.loads(x) if x else [] for x in dev.misconceptions]
err = y != 'NONE'
fam = lambda c: c.split('-')[0]
lost = {'total_lost_points': float((1 - pts).sum()), 'none_rows_lost': float((1 - pts[~err]).sum()),
        'err_rows_lost': float((1 - pts[err]).sum())}
cats = collections.Counter()
for i in np.flatnonzero(err):
    t1 = top[i, 0]; secondary = {a['category'] for a in Md[i] if not a.get('root')}
    if pts[i] == 1: cats['hit@1'] += 1; continue
    if t1 == 'NONE': k = 'top1_NONE'
    elif t1 in secondary: k = 'top1_is_annotated_secondary_error'
    elif sec.get(t1) == sec.get(y[i]): k = 'top1_same_section_wrong_code'
    elif fam(t1) == fam(y[i]): k = 'top1_same_family'
    else: k = 'top1_other_section'
    cats[k] += 1
    cats[k + '_lost_pts'] += 1 - pts[i]
out['dev_blend_error_row_breakdown'] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in cats.items()}
out['dev_blend_lost'] = lost
# secondary codes in blend top-3 occupying slots
occ = [any(c in {a['category'] for a in Md[i] if not a.get('root')} and c != y[i] for c in top[i]) for i in range(len(y))]
out['dev_rows_where_secondary_code_in_top3'] = int(sum(occ))
# section-level accuracy & oracle section restriction
S = np.array([sec.get(c, 'NONE') for c in classes]); ys = np.array([sec.get(c, 'NONE') for c in y])
Ps = pd.DataFrame(P, columns=S).T.groupby(level=0).sum().T
secs = Ps.columns.to_numpy(); top_sec = secs[Ps.to_numpy().argmax(1)]
out['dev_section_top1_acc_all'] = float((top_sec == ys).mean())
out['dev_section_top1_acc_err'] = float((top_sec[err] == ys[err]).mean())
sr = np.argsort(-Ps.to_numpy(), 1)[:, :3]
out['dev_section_in_top3_err'] = float(np.mean([ys[i] in secs[sr[i]] for i in np.flatnonzero(err)]))
out['dev_code_top1_acc_err'] = float((top[err, 0] == y[err]).mean())
mask = (S[None, :] == ys[:, None]).astype(float)
Q = normalize(P * mask + 1e-12)
out['oracle_section_restricted_map3'] = map3(y, Q, classes)
out['oracle_section_restricted_err_map3'] = float(row_scores(y[err], topk(Q[err], classes)).mean())
# oracle family
F = np.array([fam(c) for c in classes]); yf = np.array([fam(c) for c in y])
Qf = normalize(P * (F[None, :] == yf[:, None]) + 1e-12)
out['oracle_family_restricted_map3'] = map3(y, Qf, classes)
fam_p = pd.DataFrame(P, columns=F).T.groupby(level=0).sum().T
out['dev_family_top1_acc_err'] = float((fam_p.columns.to_numpy()[fam_p.to_numpy().argmax(1)][err] == yf[err]).mean())
out['n_families'] = int(len(set(F)))
# confusion pairs (truth -> top1) on dev blend and on Claude dev
conf = collections.Counter((a, b) for a, b in zip(y, top[:, 0]) if a != b)
out['dev_blend_confusions_top'] = [(f'{a}->{b}', n) for (a, b), n in conf.most_common(20)]
cl = {}
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); cl[r['trace_id']] = (r.get('answer') or {}).get('first_code')
c1 = np.array([cl.get(t) for t in ids])
out['dev_claude_confusions_top'] = [(f'{a}->{b}', n) for (a, b), n in collections.Counter((a, b) for a, b in zip(y, c1) if a != b).most_common(15)]
out['dev_claude_top1_acc'] = float((c1 == y).mean()); out['dev_blend_top1_acc'] = float((top[:, 0] == y).mean())
both_wrong = (c1 != y) & (top[:, 0] != y)
out['dev_top1_wrong_blend_only'] = int(((c1 == y) & (top[:, 0] != y)).sum())
out['dev_top1_wrong_claude_only'] = int(((c1 != y) & (top[:, 0] == y)).sum())
out['dev_top1_wrong_both'] = int(both_wrong.sum())
# by train frequency of the truth code (fit rows)
fc = fit.root_code.value_counts()
bins = pd.cut(dev.root_code.map(fc).fillna(0), [-1, 0, 5, 15, 40, 1000], labels=['0', '1-5', '6-15', '16-40', '>40'])
c_pts = row_scores(y, [[c] for c in c1])
out['dev_by_fit_freq'] = {str(b): dict(n=int((bins == b).sum()), blend=float(pts[bins == b].mean()) if (bins == b).any() else None,
                                       claude_top1=float(c_pts[bins == b].mean()) if (bins == b).any() else None) for b in bins.cat.categories}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open('misconceptions_structure.json', 'w'), indent=1, default=str)
