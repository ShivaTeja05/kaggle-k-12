"""Turn model scores on unlabelled photos into soft, class-balanced pseudo-label rows.

Keeps each row's top codes and their renormalized probabilities (not a hard guess),
drops low-confidence rows, and caps NONE so the extra data goes to error codes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from k12.core import read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scores', help='pod_predict bundle for the unlabelled subset')
    parser.add_argument('--output', default='data/v3/pseudo_v3.jsonl')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--min-confidence', type=float, default=.5)
    parser.add_argument('--none-share', type=float, default=.35, help='max share of kept rows whose top code is NONE')
    parser.add_argument('--max-per-code', type=int, default=300)
    parser.add_argument('--weight', type=float, default=.5)
    args = parser.parse_args()
    with np.load(args.scores, allow_pickle=False) as bundle:
        ids, classes, p = bundle['ids'], bundle['classes'], bundle['p'].astype(np.float64)
    order = np.argsort(-p, axis=1)[:, :args.top]
    confidence = p.max(axis=1)
    first = classes[order[:, 0]]

    keep = []
    by_code = Counter()
    for i in np.argsort(-confidence):
        if confidence[i] < args.min_confidence or by_code[first[i]] >= args.max_per_code:
            continue
        keep.append(i)
        by_code[first[i]] += 1
    errors = [i for i in keep if first[i] != 'NONE']
    nones = [i for i in keep if first[i] == 'NONE']
    allowed_none = int(args.none_share / (1 - args.none_share) * len(errors))
    keep = errors + nones[:allowed_none]

    rows = []
    for i in keep:
        top = p[i, order[i]]
        probs = top / top.sum()
        rows.append({'trace_id': str(ids[i]), 'weight': args.weight,
                     'soft': {str(classes[j]): float(q) for j, q in zip(order[i], probs)}})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    kept_first = Counter(str(first[i]) for i in keep)
    report = {'scored_rows': len(ids), 'kept_rows': len(rows), 'kept_error_rows': len(errors),
              'kept_none_rows': len(keep) - len(errors), 'distinct_top_codes': len(kept_first),
              'most_common_top_codes': kept_first.most_common(10),
              'settings': {k: v for k, v in vars(args).items() if k != 'scores'}}
    Path('reports/pseudo_v3.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
