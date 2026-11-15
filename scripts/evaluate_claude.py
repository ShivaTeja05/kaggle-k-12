"""Score Claude's development labels and their combination with our offline models."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_working_verifier import summarize
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from k12.holdout_calibration import calibrated


def claude_scores(path, ids, classes, shares=(.6, .25, .15)):
    """Rank-based pseudo-probabilities from Claude's three ranked codes."""
    index = {c: i for i, c in enumerate(classes)}
    answers = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        answers[record['trace_id']] = record['answer']
    p = np.full((len(ids), len(classes)), 1e-4)
    missing = 0
    for row, trace in enumerate(ids):
        answer = answers.get(trace, {})
        codes = [answer.get(k) for k in ('first_code', 'second_code', 'third_code')]
        if not any(codes):
            missing += 1
        for share, code in zip(shares, codes):
            if code in index and p[row, index[code]] < share:
                p[row, index[code]] = share
    return normalize(p), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('claude', default='runs/claude/dev_opus.jsonl', nargs='?')
    parser.add_argument('--others', nargs='*', default=[], help='name=path.npz development bundles to combine with')
    parser.add_argument('--baseline', help='another Claude dev run; deltas and the paired bootstrap are then against it')
    parser.add_argument('--report', default='reports/claude_dev_opus.json')
    args = parser.parse_args()
    train = read_csv('data/raw/train.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    folds = make_folds(train)
    validation = train.loc[folds == 0].reset_index(drop=True)
    ids = validation.trace_id.tolist()
    reference = calibrated(aligned_bundle('runs/vision-development/k12/validation_epoch_3.npz', ids, classes),
                           __import__('k12.baseline', fromlist=['HierarchicalPrior']).HierarchicalPrior(classes)
                           .fit(train.loc[folds != 0]).predict_proba(validation), .25, 2.)
    if args.baseline:
        reference = claude_scores(args.baseline, ids, classes)[0]
    claude, missing = claude_scores(args.claude, ids, classes)
    report = {'rows': len(ids), 'claude_missing_answers': missing, 'deltas_against': args.baseline or 'image model',
              'claude': summarize(validation, claude, classes, reference)}
    truth = validation.root_code.to_numpy()
    first = topk(claude, classes)[:, 0]
    family = lambda c: c.split('-')[0].upper()
    errors = truth != 'NONE'
    report['claude_failures'] = {
        'false_alarm_on_none': int(sum(~errors & (first != 'NONE'))),
        'missed_error_said_none': int(sum(errors & (first == 'NONE'))),
        'wrong_family': int(sum(errors & (first != 'NONE') & np.array([family(a) != family(b) for a, b in zip(first, truth)]))),
        'right_family_wrong_code': int(sum(errors & (first != truth) & np.array([family(a) == family(b) and a != b for a, b in zip(first, truth)])))}
    for spec in args.others:
        name, path = spec.split('=', 1)
        other = aligned_bundle(path, ids, classes)
        for weight in (.5, .7):
            mixed = normalize(claude ** weight * other ** (1 - weight))
            report[f'claude_{weight}_x_{name}'] = summarize(validation, mixed, classes, reference)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    for key, value in report.items():
        if isinstance(value, dict) and 'map3' in value:
            print(f"{key:32s} map3={value['map3']:.4f} top1={value['top1_accuracy']:.3f} "
                  f"root_err={value['root_error_map3']:.3f} none={value['none_map3']:.3f}")
        else:
            print(key, value)


if __name__ == '__main__':
    main()
