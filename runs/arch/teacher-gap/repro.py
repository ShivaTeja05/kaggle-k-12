import numpy as np
from k12.core import aligned_bundle, label_space, make_folds, map3, normalize, read_csv
train = read_csv('data/raw/train.csv'); classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
folds = make_folds(train); dev = train.loc[folds == 0].reset_index(drop=True); ids = dev.trace_id.tolist()
A = aligned_bundle('runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes)
C = aligned_bundle('runs/pod/dev-distill/validation_epoch_2.npz', ids, classes)
for e in ('e2', 'e3'):
    B = aligned_bundle(f'runs/pod/tx/text_tx_dev_{e}.npz', ids, classes)
    for w in [(.35, .30, .35), (.35, .35, .30), (.3, .35, .35)]:
        p = normalize(np.exp(w[0]*np.log(A)+w[1]*np.log(B)+w[2]*np.log(C)))
        print(e, w, round(map3(dev.root_code, p, classes), 4))
    print(e, 'single', round(map3(dev.root_code, B, classes), 4))
print('32B', map3(dev.root_code, A, classes), 'VL', map3(dev.root_code, C, classes))
