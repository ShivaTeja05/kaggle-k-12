"""Score a `head` bundle (ids, classes, p, final_ok) on the development rows: MAP@3 alone, overall and by slice
(data/v7/dev_slices.json), top-1, and the accuracy of the `final_ok` output against the teachers' final_answer_correct.

    ~/.cache/k12-api-venv/bin/python kaggle/q35-9b-ft/score_dev.py <bundle.npz> [--report out.json]
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bundle'); ap.add_argument('--report')
    a = ap.parse_args()
    with open(ROOT / 'data/raw/train.csv', newline='') as handle:
        rows = {r['trace_id']: r for r in csv.DictReader(handle)}
    slices = json.loads((ROOT / 'data/v7/dev_slices.json').read_text())
    z = np.load(a.bundle, allow_pickle=False)
    ids, classes, p = [str(x) for x in z['ids']], [str(x) for x in z['classes']], z['p']
    final = z['final_ok'] if 'final_ok' in z.files else None
    points, top1, by, flags = [], [], {}, []
    for k, tid in enumerate(ids):
        if tid not in slices:
            continue
        truth = rows[tid]['root_code']
        order = [classes[j] for j in np.argsort(-p[k])[:3]]
        score = 1. / (order.index(truth) + 1) if truth in order else 0.
        points.append(score); top1.append(order[0] == truth); by.setdefault(slices[tid], []).append(score)
        if final is not None:
            want = str(rows[tid]['final_answer_correct']).strip().lower() in ('true', '1', 'yes')
            flags.append((float(final[k]) >= .5) == want)
    rng = np.random.default_rng(2026); pts = np.asarray(points)
    boot = [pts[rng.integers(0, len(pts), len(pts))].mean() for _ in range(4000)] if len(pts) else [0.]
    out = {'bundle': a.bundle, 'rows_in_bundle': len(ids), 'dev_rows_scored': len(points), 'dev_rows_total': len(slices),
           'map3': round(float(pts.mean()), 4) if len(pts) else None, 'map3_ci95': [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
           'top1': round(float(np.mean(top1)), 4) if top1 else None,
           'by_slice': {s: {'rows': len(v), 'map3': round(float(np.mean(v)), 4)} for s, v in sorted(by.items())},
           'final_ok_accuracy': round(float(np.mean(flags)), 4) if flags else None,
           'final_ok_share_true_in_truth': round(float(np.mean([str(rows[t]['final_answer_correct']).strip().lower() in ('true', '1', 'yes') for t in ids if t in slices])), 4) if points else None,
           'mean_top_probability': round(float(p.max(1).mean()), 4) if len(ids) else None}
    print(json.dumps(out, indent=1))
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
