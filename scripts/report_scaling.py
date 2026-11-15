"""Claude-data scaling report: dev MAP@3 per data fraction, and 100% vs 0% with a paired question bootstrap."""
import json
from pathlib import Path

import numpy as np

from k12.core import aligned_bundle, label_space, make_folds, read_csv, row_scores, topk

ROOT = Path('runs/pod/scaling')


def paired_interval(validation, a, b, samples=2000):
    frame = validation.assign(delta=a - b).groupby('question_id').delta.agg(['sum', 'size']).to_numpy()
    draws = np.random.default_rng(2026).integers(0, len(frame), (samples, len(frame)))
    boot = frame[draws, 0].sum(axis=1) / frame[draws, 1].sum(axis=1)
    return np.quantile(boot, [.025, .975]).round(4).tolist()


def main():
    train = read_csv('data/raw/train.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    validation = train.loc[make_folds(train) == 0].reset_index(drop=True)
    ids, truth = validation.trace_id.tolist(), validation.root_code.to_numpy()
    points, report = {}, {'rows': len(ids), 'runs': {}}
    for share in (0, 25, 50, 100):
        run = ROOT / f'scale_{share}'
        if not run.exists():
            continue
        entry = {}
        for source, name in (('tx_dev', 'machine'), ('gold', 'teacher')):
            path = run / f'validation_{source}_epoch_1.npz'
            if path.exists():
                pts = row_scores(truth, topk(aligned_bundle(path, ids, classes), classes))
                entry[f'{name}_map3'] = round(float(pts.mean()), 4)
                points[(share, name)] = pts
        report['runs'][f'{share}%'] = entry
    for name in ('machine', 'teacher'):
        for share in (25, 50, 100):
            if (share, name) in points and (0, name) in points:
                report[f'{name}_{share}_vs_0'] = {
                    'delta': round(float((points[(share, name)] - points[(0, name)]).mean()), 4),
                    'ci95': paired_interval(validation, points[(share, name)], points[(0, name)])}
        if (100, name) in points and (25, name) in points:
            report[f'{name}_100_vs_25'] = {
                'delta': round(float((points[(100, name)] - points[(25, name)]).mean()), 4),
                'ci95': paired_interval(validation, points[(100, name)], points[(25, name)])}
    Path('reports/claude_scaling.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
