"""Compare pod development predictions with the submitted 4B blend on the same 301 rows."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_working_verifier import summarize
from k12.baseline import HierarchicalPrior
from k12.core import aligned_bundle, label_space, make_folds, map3, normalize, read_csv
from k12.holdout_calibration import calibrated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation', nargs='+', help='validation_epoch_N.npz bundles from the pod')
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--image-4b', default='runs/vision-development/k12/validation_epoch_3.npz')
    parser.add_argument('--output')
    args = parser.parse_args()
    data = Path(args.data)
    train = read_csv(data / 'train.csv')
    classes = label_space(read_csv(data / 'taxonomy_v1.csv'))
    folds = make_folds(train)
    fit, validation = train.loc[folds != 0], train.loc[folds == 0].reset_index(drop=True)
    ids = validation.trace_id.tolist()
    prior = HierarchicalPrior(classes).fit(fit).predict_proba(validation)
    image_4b = aligned_bundle(args.image_4b, ids, classes)
    reference = calibrated(image_4b, prior, .25, 2.)
    report = {'rows': len(validation), 'reference': 'Submitted 4B recipe (temperature 2, 25% prior)',
              'reference_map3': map3(validation.root_code, reference, classes), 'runs': {}}
    for path in args.validation:
        p = aligned_bundle(path, ids, classes)
        variants = {'raw': p, 'fixed_calibration': calibrated(p, prior, .25, 2.)}
        for weight in [.3, .5]:
            mixed = normalize((1 - weight) * np.sqrt(p) + weight * np.sqrt(image_4b))
            variants[f'with_4b_{weight}'] = calibrated(mixed, prior, .25, 1.)
        report['runs'][path] = {name: summarize(validation, scores, classes, reference)
                                for name, scores in variants.items()}
    text = json.dumps(report, indent=1)
    if args.output:
        Path(args.output).write_text(text + '\n')
    for path, variants in report['runs'].items():
        print(path)
        for name, metric in variants.items():
            print(f"  {name:20s} map3={metric['map3']:.4f} delta={metric['delta_vs_image']:+.4f} "
                  f"ci={np.round(metric['paired_question_bootstrap_delta_95'], 4).tolist()} "
                  f"root_err={metric['root_error_map3']:.4f} none={metric['none_map3']:.4f}")
    print('reference', round(report['reference_map3'], 4))


if __name__ == '__main__':
    main()
