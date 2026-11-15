"""Per-member MAP@3 by mechanism; is the image model a fallback for reading failures (M6)?"""
import numpy as np, pandas as pd
from common import load, ROOT, boot_ci
from k12.core import aligned_bundle, topk, row_scores
tr, tax, classes, folds, dev, fit = load(); ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
d = pd.read_csv(ROOT/'runs/deep/casebook/casebook_labelled.csv', keep_default_na=False)
B = {'32b_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', '8b_tx': 'runs/pod/tx/text_tx_dev_e3.npz', 'vl_img': 'runs/pod/dev-distill/validation_epoch_2.npz'}
S, P = {}, {}
for k, p in B.items():
    P[k] = aligned_bundle(ROOT/p, ids, classes); S[k] = row_scores(y, topk(P[k], classes)); d[k] = S[k]
print(d.groupby('mech')[list(B) + ['s_blend', 's_claude']].mean().round(3).to_string())
ci = {c: i for i, c in enumerate(classes)}
# Disagreement signal: text members say NONE top-1 while the image model does not
t32 = np.array(topk(P['32b_tx'], classes))[:, 0]; tvl = np.array(topk(P['vl_img'], classes))[:, 0]
flag = (t32 == 'NONE') & (tvl != 'NONE')
print('\nflag text NONE & image not NONE:', int(flag.sum()), 'rows; truth NONE among them:', int((y[flag] == 'NONE').sum()),
      '; M6 rows among them:', int((d.mech[flag] == 'M6').sum()), 'of', int((d.mech == 'M6').sum()))
print('M6 rows with blend top-1 NONE where VL top-1 is not NONE:', int(((d.mech == 'M6') & (d.blend_top3.str.split().str[0] == 'NONE') & (tvl != 'NONE')).sum()))
