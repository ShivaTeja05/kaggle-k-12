from common import *
import json, collections, itertools
tr, tax, classes, folds, dev, fit = load()
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
GEN = {'CROSS-STRAND', 'CATCH-ALL'}
kind = lambda c: 'NONE' if c == 'NONE' else ('generic' if sec[c] in GEN else 'specific')
out = {}
fac = tr.final_answer_correct.astype(str).eq('True'); cl = tr.clean.astype(str).eq('True')
out['P(NONE|clean)'] = float((tr.root_code[cl] == 'NONE').mean()); out['P(NONE|fac)'] = float((tr.root_code[fac] == 'NONE').mean())
out['P(NONE|~fac)'] = float((tr.root_code[~fac] == 'NONE').mean())
out['codes_when_fac_true_and_error'] = collections.Counter(tr.root_code[fac & (tr.root_code != 'NONE')]).most_common(10)
out['kind_share_train_err'] = dict(collections.Counter(kind(c) for c in tr.root_code if c != 'NONE'))
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
comps = [('32b', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8b', 'runs/pod/tx/text_tx_dev_e3.npz'), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz'), ('blend', None)]
S = np.array([sec[c] for c in classes]); ys = np.array([sec[c] for c in y]); err = y != 'NONE'
K = np.array([kind(c) for c in classes]); yk = np.array([kind(c) for c in y])
cl = {}
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); cl[r['trace_id']] = (r.get('answer') or {}).get('first_code')
c1 = np.array([cl.get(t) or 'NONE' for t in ids])
for n, p in comps:
    Pm = P if p is None else aligned_bundle(ROOT/p, ids, classes)
    t1 = np.asarray(classes)[Pm.argmax(1)]
    Ps = pd.DataFrame(Pm, columns=S).T.groupby(level=0).sum().T
    ts = Ps.columns.to_numpy()[Ps.to_numpy().argmax(1)]
    same = (ts == ys) & err
    Qin = normalize(Pm * (S[None, :] == ys[:, None]) + 1e-12)
    out[f'{n}'] = dict(map3=map3(y, Pm, classes), section_acc_err=float((ts[err] == ys[err]).mean()),
                       code_acc_given_section_top1_correct=float((t1[same] == y[same]).mean()),
                       within_true_section_top1_acc_err=float((np.asarray(classes)[Qin.argmax(1)][err] == y[err]).mean()),
                       kind_confusion=pd.crosstab(pd.Series(yk, name='truth'), pd.Series(np.array([kind(c) for c in t1]), name='pred')).to_dict())
out['claude'] = dict(top1=float((c1 == y).mean()), section_acc_err=float((np.array([sec.get(c, 'NONE') for c in c1])[err] == ys[err]).mean()),
                     kind_confusion=pd.crosstab(pd.Series(yk, name='truth'), pd.Series(np.array([kind(c) if c in sec else 'NONE' for c in c1]), name='pred')).to_dict())
# blend MAP by truth kind, and Claude's
pts = row_scores(y, topk(P, classes))
out['blend_map3_by_truth_kind'] = {k: float(pts[yk == k].mean()) for k in ('NONE', 'generic', 'specific')}
out['claude_top1_by_truth_kind'] = {k: float((c1 == y)[yk == k].mean()) for k in ('NONE', 'generic', 'specific')}
out['dev_kind_counts'] = dict(collections.Counter(yk))
# lost points share
out['lost_pts_by_truth_kind'] = {k: float((1 - pts[yk == k]).sum()) for k in ('NONE', 'generic', 'specific')}
# per generic code
gp = {c: (int((y == c).sum()), float(pts[y == c].mean()), float((c1 == y)[y == c].mean())) for c in sorted(set(y)) if kind(c) == 'generic'}
out['generic_codes_dev_n_blendmap_claudetop1'] = gp
print(json.dumps(out, indent=1, default=str))
json.dump(out, open('hierarchy.json', 'w'), indent=1, default=str)
