from common import *
from k12.core import save_bundle, write_submission
import json
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); P = blend(ids, classes)
t1 = np.asarray(classes)[P.argmax(1)]
M = [json.loads(x) if x else [] for x in dev.misconceptions]
fa = np.flatnonzero((y == 'NONE') & (t1 != 'NONE'))
out = {'dev_none_false_alarms': int(len(fa)),
       'false_alarms_on_rows_with_minor_annotations': int(sum(len(M[i]) > 0 for i in fa)),
       'false_alarm_code_equals_minor_annotation': int(sum(t1[i] in {a['category'] for a in M[i]} for i in fa)),
       'dev_none_rows_with_minor_annotations': int(sum(len(M[i]) > 0 for i in np.flatnonzero(y == 'NONE')))}
# test: balanced blend bundle (lambda 0.5), staged only
te = read_csv(ROOT/'data/raw/test.csv'); tids = te.trace_id.tolist()
ci = {c: i for i, c in enumerate(classes)}
pi = np.bincount(tr.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
lp = (.35*np.log(aligned_bundle(ROOT/'runs/pod/text32-full/test.npz', tids, classes)) + .30*np.log(aligned_bundle(ROOT/'runs/pod/text-full/test.npz', tids, classes))
      + .35*np.log(aligned_bundle(ROOT/'runs/pod/full-distill/test.npz', tids, classes)))
PT = normalize(np.exp(lp - lp.max(1, keepdims=True)))
b = np.zeros(len(classes))
for _ in range(50):
    b += np.log(pi / normalize(PT * np.exp(b)).mean(0))
QT = normalize(PT * np.exp(.5 * b))
a, c = topk(PT, classes), topk(QT, classes)
out['test_top1_changed_share'] = float((a[:, 0] != c[:, 0]).mean())
out['test_top3_set_changed_share'] = float(np.mean([set(x) != set(z) for x, z in zip(a, c)]))
out['test_NONE_top1_share_before_after'] = [float((a[:, 0] == 'NONE').mean()), float((c[:, 0] == 'NONE').mean())]
out['test_codes_never_top1_before_after'] = [int(len(set(classes) - set(a[:, 0]))), int(len(set(classes) - set(c[:, 0])))]
out['note'] = 'test components assumed: text32-full, text-full, full-distill at .35/.30/.35; verify these are the bundles behind public 0.7085 before use'
save_bundle('test_blend_balanced_l0.5.npz', tids, classes, QT, kind='sinkhorn_balanced_blend_test', lam=.5)
write_submission(te, QT, classes, 'staged_submission_balanced_l0.5.csv')
print(json.dumps(out, indent=1))
json.dump(out, open('none_falarm_and_test.json', 'w'), indent=1)
