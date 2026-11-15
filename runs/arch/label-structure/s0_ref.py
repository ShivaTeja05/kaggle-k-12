from common import *
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
for v in ['text_tx_dev_e2', 'text_tx_dev_e3']:
    print(v, map3(y, blend(ids, classes, v), classes))
for n, p in [('32b', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('8b_e3', 'runs/pod/tx/text_tx_dev_e3.npz'), ('8b_e2', 'runs/pod/tx/text_tx_dev_e2.npz'), ('vl', 'runs/pod/dev-distill/validation_epoch_2.npz')]:
    print(n, map3(y, aligned_bundle(ROOT/p, ids, classes), classes))
print(len(dev), (y == 'NONE').sum())
