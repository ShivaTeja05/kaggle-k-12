"""Weighted geometric blend of test bundles (optionally with a question prior) into a submission."""
import argparse
import json
from pathlib import Path

import numpy as np

from k12.baseline import HierarchicalPrior
from k12.core import aligned_bundle, fingerprint, label_space, normalize, read_csv, save_bundle, write_submission
from k12.holdout_calibration import calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+', help='name=path.npz:weight')
    parser.add_argument('--prior-weight', type=float, default=0.)
    parser.add_argument('--output', required=True)
    parser.add_argument('--development-map3', type=float, help='score of the same recipe on development rows')
    args = parser.parse_args()
    train, test = read_csv('data/raw/train.csv'), read_csv('data/raw/test.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    ids = test.trace_id.tolist()
    log_p, recipe = 0., {}
    for spec in args.inputs:
        name, rest = spec.split('=', 1)
        path, weight = rest.rsplit(':', 1)
        log_p = log_p + float(weight) * np.log(aligned_bundle(path, ids, classes))
        recipe[name] = {'path': path, 'weight': float(weight)}
    scores = normalize(np.exp(log_p - log_p.max(axis=1, keepdims=True)))
    if args.prior_weight:
        scores = calibrated(scores, HierarchicalPrior(classes).fit(train).predict_proba(test), args.prior_weight, 1.)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_bundle(out / 'test.npz', ids, classes, scores, kind='geometric_blend_test')
    write_submission(test, scores, classes, out / 'submission.csv')
    top = np.asarray(classes)[scores.argmax(1)]
    report = {'rows': len(test), 'recipe': recipe, 'prior_weight': args.prior_weight,
              'none_first_share': float(np.mean(top == 'NONE')),
              'development_map3': args.development_map3,
              'submission_sha256': fingerprint(out / 'submission.csv')}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
