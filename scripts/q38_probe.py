"""Small classifiers ("probes") on the untrained Qwen3.8-27B state saved by `k12.q38_finetune features`. Runs on this Mac.

For each saved layer: standardise, fit a multinomial logistic regression on the fit rows, score the dev rows. The best
layer (chosen by 5-fold cross-validation over questions inside the fit rows, never on dev) is written as a dev bundle, and,
when test features are given, refitted on all 1,500 rows and written as a test bundle. Never submits.

Usage: python scripts/q38_probe.py --features runs/.../ft_features_train_dev.npz [--test-features .../ft_features_test.npz] --out runs/q38_probe
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from k12.core import label_space, make_folds, read_csv, row_scores, save_bundle, topk

ROOT = Path(__file__).resolve().parents[1]


def fit_predict(Xa, ya, Xb, classes, C):
    scaler = StandardScaler().fit(Xa)
    model = LogisticRegression(C=C, max_iter=400).fit(scaler.transform(Xa), ya)
    P = np.full((len(Xb), len(classes)), 1e-6); P[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(scaler.transform(Xb))
    return P / P.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True); ap.add_argument('--test-features'); ap.add_argument('--out', default='runs/q38_probe')
    a = ap.parse_args()
    classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv')); train = read_csv(ROOT / 'data/raw/train.csv'); folds = make_folds(train)
    truth = dict(zip(train.trace_id, train.root_code)); qid = dict(zip(train.trace_id, train.question_id)); fold = dict(zip(train.trace_id, folds))
    b = np.load(ROOT / a.features); ids = [str(t) for t in b['ids']]; X = b['x'].astype(np.float32); layers = b['layers'].tolist()
    keep = [i for i, t in enumerate(ids) if ':' not in t]  # the question-blind repeats are training variants, not extra photos
    ids, X = [ids[i] for i in keep], X[keep]
    y = np.array([truth[t] for t in ids]); g = np.array([qid[t] for t in ids]); is_dev = np.array([fold[t] == 0 for t in ids])
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); report = {'rows': len(ids), 'dev_rows': int(is_dev.sum()), 'layers': {}}
    best = None
    for li, layer in enumerate(layers):
        for C in (.01, .1, 1.):
            cv = []
            for tr_i, va_i in GroupKFold(5).split(X[~is_dev, li], y[~is_dev], g[~is_dev]):
                P = fit_predict(X[~is_dev, li][tr_i], y[~is_dev][tr_i], X[~is_dev, li][va_i], classes, C)
                cv.append(row_scores(y[~is_dev][va_i], topk(P, classes)).mean())
            P = fit_predict(X[~is_dev, li], y[~is_dev], X[is_dev, li], classes, C)
            dev = float(row_scores(y[is_dev], topk(P, classes)).mean())
            report['layers'][f'{layer}|C={C}'] = {'cv_inside_fit': float(np.mean(cv)), 'dev': dev}
            if best is None or np.mean(cv) > best[0]:
                best = (float(np.mean(cv)), li, C, P)
    _, li, C, P = best
    report['chosen'] = {'layer': layers[li], 'C': C, 'cv_inside_fit': best[0], 'dev': float(row_scores(y[is_dev], topk(P, classes)).mean())}
    save_bundle(out / 'dev.npz', [t for t, d in zip(ids, is_dev) if d], classes, P, kind='q38_probe_dev', layer=layers[li], C=C)
    if a.test_features:
        t = np.load(ROOT / a.test_features); tids = [str(x) for x in t['ids']]
        PT = fit_predict(X[:, li], y, t['x'].astype(np.float32)[:, li], classes, C)
        save_bundle(out / 'test.npz', tids, classes, PT, kind='q38_probe_test', layer=layers[li], C=C); report['test_rows'] = len(tids)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n'); print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
