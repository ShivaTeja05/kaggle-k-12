"""Judge model: learn per candidate when to trust which model, with question-grouped cross-fitting.

Candidates are the union of each model's top-k codes. Features are each model's log
probability, reciprocal rank and top-1 flag, family-specific trust terms, and agreement.
The development score is cross-fitted; the judge refit on all development rows scores test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_working_verifier import summarize
from k12.baseline import HierarchicalPrior
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, save_bundle, write_submission
from k12.holdout_calibration import calibrated

FAMILIES = ['NONE', 'CAL', 'ANS', 'WOP', 'FRA', 'ALG', 'EXP', 'OTH', 'FRM']


def family(code):
    return code.split('-', 1)[0].upper()


def candidates_and_features(models, classes, top):
    names = list(models)
    rows = next(iter(models.values())).shape[0]
    orders = {n: np.argsort(-p, axis=1) for n, p in models.items()}
    ranks = {n: np.argsort(o, axis=1) + 1 for n, o in orders.items()}
    fam = np.asarray([family(c) for c in classes])
    features, positions = [], []
    for i in range(rows):
        pool = dict.fromkeys(j for n in names for j in orders[n][i, :top])
        for j in pool:
            row = []
            for n in names:
                logp, recip = np.log(max(models[n][i, j], 1e-9)), 1 / ranks[n][i, j]
                row += [logp, recip, ranks[n][i, j] == 1, models[n][i].max()]
                # Family-specific trust: lets the judge prefer a model for the error types it handles best.
                row += [logp * (fam[j] == f) for f in FAMILIES]
            row += [sum(ranks[n][i, j] == 1 for n in names), sum(ranks[n][i, j] <= 3 for n in names)]
            features.append(row)
            positions.append((i, j))
    return np.asarray(features, dtype=float), np.asarray(positions)


def scores_from(confidence, positions, shape, fallback):
    out = 1e-6 * normalize(fallback)
    out[positions[:, 0], positions[:, 1]] = confidence + 1e-3
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', nargs='+', required=True, help='name=development.npz')
    parser.add_argument('--test', nargs='*', default=[], help='name=test.npz (same names) to score test')
    parser.add_argument('--top', type=int, default=6)
    parser.add_argument('--c', type=float, default=.1)
    parser.add_argument('--output', default='artifacts/stacked')
    args = parser.parse_args()
    train = read_csv('data/raw/train.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    folds = make_folds(train)
    validation = train.loc[folds == 0].reset_index(drop=True)
    ids = validation.trace_id.tolist()
    dev = {s.split('=')[0]: aligned_bundle(s.split('=', 1)[1], ids, classes) for s in args.dev}
    fallback = normalize(np.exp(np.mean([np.log(p) for p in dev.values()], axis=0)))
    features, positions = candidates_and_features(dev, classes, args.top)
    truth = validation.root_code.to_numpy()
    targets = np.asarray(classes)[positions[:, 1]] == truth[positions[:, 0]]
    groups = validation.question_id.to_numpy()[positions[:, 0]]

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=args.c, max_iter=3000))

    confidence = np.empty(len(features))
    for fit, held in GroupKFold(n_splits=5).split(features, targets, groups):
        confidence[held] = model().fit(features[fit], targets[fit]).predict_proba(features[held])[:, 1]
    stacked = scores_from(confidence, positions, fallback.shape, fallback)
    reference = calibrated(aligned_bundle('runs/vision-development/k12/validation_epoch_3.npz', ids, classes),
                           HierarchicalPrior(classes).fit(train.loc[folds != 0]).predict_proba(validation), .25, 2.)
    report = {'models': list(dev), 'top': args.top, 'c': args.c,
              'coverage': float(np.mean([truth[i] in {classes[j] for j in positions[positions[:, 0] == i, 1]}
                                         for i in range(len(ids))])),
              'equal_weight_geometric': summarize(validation, fallback, classes, reference)['map3'],
              'stacked_crossfit': summarize(validation, stacked, classes, reference)}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_bundle(out / 'dev_crossfit.npz', ids, classes, stacked, kind='stacked_crossfit_development')
    if args.test:
        test = read_csv('data/raw/test.csv')
        tests = {s.split('=')[0]: aligned_bundle(s.split('=', 1)[1], test.trace_id.tolist(), classes) for s in args.test}
        if list(tests) != list(dev):
            raise SystemExit('Test bundles must use the same model names in the same order')
        judge = model().fit(features, targets)
        t_features, t_positions = candidates_and_features(tests, classes, args.top)
        t_fallback = normalize(np.exp(np.mean([np.log(p) for p in tests.values()], axis=0)))
        t_scores = scores_from(judge.predict_proba(t_features)[:, 1], t_positions, t_fallback.shape, t_fallback)
        save_bundle(out / 'test.npz', test.trace_id.tolist(), classes, t_scores, kind='stacked_test')
        write_submission(test, t_scores, classes, out / 'submission.csv')
        report['test_written'] = True
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    s = report['stacked_crossfit']
    print(json.dumps({k: v for k, v in report.items() if k != 'stacked_crossfit'}, indent=1))
    print(f"stacked (cross-fit) map3={s['map3']:.4f} root={s['root_error_map3']:.3f} none={s['none_map3']:.3f} "
          f"ci_vs_ref={np.round(s['paired_question_bootstrap_delta_95'], 3).tolist()}")


if __name__ == '__main__':
    main()
