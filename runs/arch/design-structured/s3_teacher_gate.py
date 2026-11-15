"""Diagnostic only (uses Claude on dev; not deployable): how much does an IMPERFECT but strong
stage-1 gate (Claude's top-1 kind/section/family) add to the blend, hard vs soft? Tells whether
a hierarchical gate needs teacher-level accuracy to pay off."""
import json, numpy as np
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from evaluate_claude import claude_scores
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv'); classes = label_space(tax)
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
folds = make_folds(train); dev = train[folds == 0].reset_index(drop=True); ids = dev.trace_id.tolist(); y = dev.root_code.values
L = lambda p: aligned_bundle(p, ids, classes)
blend = normalize(np.exp(.35*np.log(L('runs/pod/text32/validation_tx_dev_epoch_1.npz'))+.30*np.log(L('runs/pod/tx/text_tx_dev_e3.npz'))+.35*np.log(L('runs/pod/dev-distill/validation_epoch_2.npz'))))
ref = row_scores(y, topk(blend, classes)).mean()
Cl, _ = claude_scores('runs/claude/dev_opus.jsonl', ids, classes); ctop = topk(Cl, classes)[:, 0]
kind = lambda c: 'NONE' if c == 'NONE' else ('GEN' if sec[c] == 'CROSS-STRAND' else 'SPEC')
fns = {'kind3': kind, 'section': lambda c: sec[c], 'family': lambda c: c.split('-')[0]}
out = {'ref': ref}
for n, f in fns.items():
    arr = np.array([f(c) for c in classes])
    acc = np.mean([f(a) == f(b) for a, b in zip(ctop, y)]); out[n+'_claude_stage_acc'] = acc
    tacc = np.mean([f(a) == f(b) for a, b in zip(topk(blend, classes)[:, 0], y)]); out[n+'_blend_stage_acc'] = tacc
    for soft in [0.0, 0.1, 0.3]:
        m = np.array([[1.0 if arr[j] == f(ctop[i]) else soft for j in range(len(classes))] for i in range(len(y))])
        out[f'{n}_claudegate_soft{soft}'] = row_scores(y, topk(blend*m + 1e-12, classes)).mean()
print(json.dumps({k: round(float(v), 4) for k, v in out.items()}, indent=1))
json.dump(out, open('runs/arch/design-structured/s3_teacher_gate.json', 'w'), indent=1, default=float)
