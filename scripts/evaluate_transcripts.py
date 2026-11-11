"""Compare machine transcripts with teacher transcripts on labelled rows (aggregate only)."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from k12.core import read_csv


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def numbers(text):
    return re.findall(r'\d+(?:[.,]\d+)?', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('transcripts')
    parser.add_argument('--data', default='data/raw')
    args = parser.parse_args()
    machine = json.loads(Path(args.transcripts).read_text())
    train = read_csv(Path(args.data) / 'train.csv').set_index('trace_id')
    rates, recalls, precisions, line_ratio = [], [], [], []
    for trace, text in machine.items():
        gold = str(train.loc[trace, 'transcription_steps'])
        rates.append(distance(text, gold) / max(1, len(gold)))
        g, m = numbers(gold), numbers(text)
        if g:
            recalls.append(sum(min(g.count(x), m.count(x)) for x in set(g)) / len(g))
        if m:
            precisions.append(sum(min(g.count(x), m.count(x)) for x in set(m)) / len(m))
        line_ratio.append(len(text.splitlines()) / max(1, len(gold.splitlines())))
    report = {'rows': len(machine), 'character_error_rate_median': float(np.median(rates)),
              'character_error_rate_mean': float(np.mean(rates)),
              'number_recall': float(np.mean(recalls)), 'number_precision': float(np.mean(precisions)),
              'line_count_ratio_median': float(np.median(line_ratio))}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
