"""W3 test 7: upper bound for 'let a photo-reading model decide when the transcript lost the root span'.
Oracle (uses gold to pick rows): on error rows whose gold root span is absent from the machine transcript, swap the
blend's log-weights toward the VL image member (weight a); elsewhere keep the blend. Not deployable; a ceiling only."""
import json
import numpy as np, pandas as pd
from common import load, ROOT, boot_ci, blend
from k12.core import aligned_bundle, topk, row_scores, normalize
tr, tax, classes, folds, dev, fit = load(); ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
w2 = pd.read_csv(ROOT/'runs/deep/approach-W3/w2_rows.csv').set_index('trace_id').reindex(ids)
L = {k: np.log(aligned_bundle(ROOT/p, ids, classes)) for k, p in
     [('32', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8', 'runs/pod/tx/text_tx_dev_e3.npz'), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz')]}
def mix(a):
    lp = (1 - a) / .65 * (.35 * L['32'] + .30 * L['8']) + a * L['vl']; return normalize(np.exp(lp - lp.max(1, keepdims=True)))
ref = row_scores(y, topk(blend(ids, classes), classes))
lost = (y != 'NONE') & (w2.m_span_in.fillna(1).to_numpy() == 0)
out = {'rows_span_lost': int(lost.sum())}
for a in (.35, .5, .65, .8, 1.0):
    P = np.where(lost[:, None], mix(a), blend(ids, classes)); s = row_scores(y, topk(P, classes))
    out[f'a={a}'] = [round(float(s.mean()), 4)] + [round(x, 4) if not isinstance(x, list) else [round(z, 4) for z in x] for x in boot_ci(s, ref, qid)]
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT/'runs/deep/approach-W3/w7_photo_oracle.json', 'w'), indent=1)
