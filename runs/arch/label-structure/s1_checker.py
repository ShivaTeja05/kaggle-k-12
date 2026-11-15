from common import *
from checker import check
import json
tr, tax, classes, folds, dev, fit = load()
from sklearn.metrics import roc_auc_score
def evaluate(frame, col, name):
    out = {}
    for k in (1, 2, 3):
        res = [check(t, a, k) for t, a in zip(frame[col], frame.correct_answer)]
        s = np.array([r[0] for r in res]); y = frame.final_answer_correct.astype(str).eq('True').to_numpy()
        pred = s >= .999
        none = (frame.root_code == 'NONE').to_numpy()
        out[k] = dict(acc_vs_fac=float((pred == y).mean()), auc_vs_fac=float(roc_auc_score(y, s)),
                      precision_true=float(y[pred].mean()), recall_true=float(pred[y].mean()),
                      auc_vs_none=float(roc_auc_score(none, s)),
                      none_rate_if_pred_true=float(none[pred].mean()), none_rate_if_pred_false=float(none[~pred].mean()),
                      pred_true_share=float(pred.mean()))
        print(name, k, {a: round(b, 3) for a, b in out[k].items()})
    return out
R = {}
R['gold_fit'] = evaluate(fit, 'transcription_steps', 'gold-fit')
R['gold_dev'] = evaluate(dev, 'transcription_steps', 'gold-dev')
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
d2 = dev.assign(tx=dev.trace_id.map(tx))
print('missing tx', d2.tx.isna().sum())
R['machine_dev'] = evaluate(d2, 'tx', 'machine-dev')
json.dump(R, open('checker_metrics.json', 'w'), indent=1)
