"""Blend pod test predictions with the 4B and a question prior into a submission.

Uses the recipe chosen on development rows: geometric mix of the 8B (weight 0.7) and the
4B (0.3), then 10% hierarchical prior fitted on every labelled row.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from k12.baseline import HierarchicalPrior
from k12.core import aligned_bundle, fingerprint, label_space, normalize, read_csv, save_bundle, write_submission
from k12.holdout_calibration import calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_8b')
    parser.add_argument('--test-4b', default='runs/vision-development/k12/test.npz')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--weight-4b', type=float, default=.3)
    parser.add_argument('--prior-weight', type=float, default=.1)
    parser.add_argument('--output', default='artifacts/submission-8b-read-blend')
    args = parser.parse_args()
    data = Path(args.data)
    train, test = read_csv(data / 'train.csv'), read_csv(data / 'test.csv')
    classes = label_space(read_csv(data / 'taxonomy_v1.csv'))
    ids = test.trace_id.tolist()
    large, small = aligned_bundle(args.test_8b, ids, classes), aligned_bundle(args.test_4b, ids, classes)
    mixed = normalize(large ** (1 - args.weight_4b) * small ** args.weight_4b)
    prior = HierarchicalPrior(classes).fit(train).predict_proba(test)
    scores = calibrated(mixed, prior, args.prior_weight, 1.)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_bundle(out / 'test.npz', ids, classes, scores, kind='blended_test')
    write_submission(test, scores, classes, out / 'submission.csv')
    top = np.asarray(classes)[scores.argmax(1)]
    report = {'rows': len(test), 'inputs': {'test_8b': args.test_8b, 'test_4b': args.test_4b},
              'weight_4b': args.weight_4b, 'prior_weight': args.prior_weight,
              'none_first_share': float(np.mean(top == 'NONE')),
              'submission_sha256': fingerprint(out / 'submission.csv'),
              'development_map3_of_recipe': 0.6467,
              'warning': 'Recipe chosen among four blends on the same 301 development rows; slightly optimistic.'}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
