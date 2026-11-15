"""Oracle value of each stage of a structured decomposition, applied as gates to the blend.
Stage gates: (a) NONE vs error, (b) generic (CROSS-STRAND) vs topic-specific, (c) section, (d) family.
Also: gate quality of the blend itself (AUC) for (b)."""
import json, numpy as np
from sklearn.metrics import roc_auc_score
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv'); classes = label_space(tax)
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
folds = make_folds(train); dev = train[folds == 0].reset_index(drop=True); ids = dev.trace_id.tolist(); y = dev.root_code.values
L = lambda p: aligned_bundle(p, ids, classes)
blend = normalize(np.exp(.35*np.log(L('runs/pod/text32/validation_tx_dev_epoch_1.npz'))+.30*np.log(L('runs/pod/tx/text_tx_dev_e3.npz'))+.35*np.log(L('runs/pod/dev-distill/validation_epoch_2.npz'))))
ref = row_scores(y, topk(blend, classes))
print(sorted(set(tax.section)))
kind = lambda c: 'NONE' if c == 'NONE' else ('GEN' if sec[c].upper().startswith('CROSS') else 'SPEC')
fam = lambda c: c.split('-')[0]
cls_kind = np.array([kind(c) for c in classes]); cls_sec = np.array([sec[c] for c in classes]); cls_fam = np.array([fam(c) for c in classes])
def gate(keyf, arr):
    m = np.array([[arr[j] == keyf(y[i]) for j in range(len(classes))] for i in range(len(y))])
    p = blend * m + 1e-12
    return row_scores(y, topk(p, classes)).mean()
out = {'ref': ref.mean()}
for name, f, arr in [('kind3', kind, cls_kind), ('section', lambda c: sec[c], cls_sec), ('family', fam, cls_fam)]:
    out['oracle_'+name] = gate(f, arr)
err = y != 'NONE'; tk = np.array([kind(c) for c in y])
pg = blend[:, cls_kind == 'GEN'].sum(1); ps = blend[:, cls_kind == 'SPEC'].sum(1)
out['n_err_GEN'] = int((tk == 'GEN').sum()); out['n_err_SPEC'] = int((tk == 'SPEC').sum())
out['auc_GEN_vs_SPEC_err_rows'] = roc_auc_score(tk[err] == 'GEN', (pg/(pg+ps))[err])
top = topk(blend, classes)[:, 0]; tkind = np.array([kind(c) for c in top])
out['kind_acc_err'] = float((tkind[err] == tk[err]).mean())
out['map_err_by_truekind'] = {k: float(ref[tk == k].mean()) for k in ['GEN', 'SPEC', 'NONE']}
out['top1_kind_confusion_err'] = {f'{a}->{b}': int(((tk == a) & (tkind == b) & err).sum()) for a in ['GEN', 'SPEC'] for b in ['GEN', 'SPEC', 'NONE']}
print(json.dumps(out, indent=1, default=float))
json.dump(out, open('runs/arch/design-structured/s2_gate.json', 'w'), indent=1, default=float)
