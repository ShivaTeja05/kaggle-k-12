"""Shortlist of codes for each development row, for the reasoner's v2 prompt (ARCHITECTURE_REFORM_2026.md 4.4.5). CPU.

S_i = top-8 of the submitted system  U  the three codes of the zero-shot answer (when the row has one)  U  {NONE}.
The submitted system is rebuilt exactly as scripts/fuse_vlm.py does it on dev: fuse(P0, P1, .5, .75, pi, 30).
"sent" is the confidence flag alone (top probability below --threshold); fuse_vlm.py's send flag is sent and
has_zero_shot. Codes are stored in alphabetical order, so the file carries no rank or score.

Development rows only: there is no test mode here. Prints the shortlist sizes and the recall of the true code by
slice, with the submitted system's MAP@3 by slice as a check that the rebuild is the system that was submitted.

Usage: python scripts/build_shortlist.py [--out data/v10/shortlist_dev.json]
"""
import argparse
import json

import numpy as np

import fuse_candidate as F
from fuse_candidate import fuse, geo, marginals, unseen_pairs
from fuse_vlm import RANKER_DEV
from k12.core import aligned_bundle, label_space, make_folds, read_csv, row_scores, topk
from score_vlm_dev import shares

ROOT = F.ROOT
SLICES = ('overall', 'no_error', 'seen_pair', 'hard')


def shortlists(SUB, S, used, classes, k=8):
    """Per row: the top-k codes of SUB, the union with the zero-shot codes (rows of S that have an answer) and NONE."""
    top = topk(SUB, classes, k)
    out = []
    for i in range(len(SUB)):
        zero = [c for c, s in zip(classes, S[i]) if s > 0] if used[i] > 0 else []
        out.append(sorted(set(top[i]) | set(zero) | {'NONE'}))
    return top, out


def four_slices(values, rows, masks):
    """{slice: (row count, mean of values)} over the rows selected, in the four-slice layout."""
    return {name: (int((rows & m).sum()), float(values[rows & m].mean()) if (rows & m).any() else float('nan'))
            for name, m in zip(SLICES, masks)}


def table(title, columns, lines):
    print(f'\n{title}')
    print(f"{'slice':<10}{'rows':>6}" + ''.join(f'{c:>16}' for c in columns))
    for name in SLICES:
        n = lines[columns[0]][name][0]
        print(f'{name:<10}{n:>6}' + ''.join(f'{lines[c][name][1]:>16.3f}' for c in columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero-shot', default='runs/deep/q38/pod_out/q38_dev_think.jsonl')
    ap.add_argument('--threshold', type=float, default=.817)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--out', default='data/v10/shortlist_dev.json')
    a = ap.parse_args()
    if 'test' in a.zero_shot or 'test' in a.out:
        raise SystemExit('Development rows only: no test-set shortlist is written by this script')
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
    masks = (np.ones(len(y), bool), none, seen, hard)
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER_DEV, ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30)
    S, used = shares(ROOT / a.zero_shot, ids, classes, first_only=True)
    have = used > 0; below = SUB.max(1) < a.threshold
    top, lists = shortlists(SUB, S, used, classes, a.top)

    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({t: {'sent': bool(b), 'codes': codes, 'has_zero_shot': bool(h)}
                               for t, b, codes, h in zip(ids, below, lists, have)}, indent=1) + '\n')

    size = np.array([len(c) for c in lists])
    stat = lambda m: {'rows': int(m.sum()), 'min': int(size[m].min()), 'median': float(np.median(size[m])),
                      'mean': round(float(size[m].mean()), 2), 'max': int(size[m].max())}
    groups = [('all dev rows', np.ones(len(y), bool)), ('rows Qwen3.8 answered', have), ('sent rows (top probability below threshold)', below),
              ('sent rows Qwen3.8 answered', below & have)]
    shown = str(out.relative_to(ROOT)) if ROOT in out.parents else str(out)  # --out may point outside the repository
    print(json.dumps({'out': shown, 'rows': len(ids), 'threshold': a.threshold, 'top': a.top,
                      'answered': int(have.sum()), 'sent': int(below.sum()), 'sent_and_answered': int((below & have).sum()),
                      'none_already_in_top': float(np.mean(['NONE' in t for t in top])),
                      'shortlist_size': {name: stat(m) for name, m in groups}}, indent=1))

    sub = row_scores(y, topk(SUB, classes))
    print('\nSubmitted system, MAP@3. Check: 0.755 / 0.916 / 0.871 / 0.500 on 301 rows; 0.775 / 0.916 / 0.872 / 0.502 on 270.')
    for name, m in groups[:2]:
        table(f'Submitted system, {name}', ['MAP@3'], {'MAP@3': four_slices(sub, m, masks)})

    in_top = np.array([t in r for t, r in zip(y, top)], float)
    in_top_none = np.array([t in r or t == 'NONE' for t, r in zip(y, top)], float)
    in_list = np.array([t in r for t, r in zip(y, lists)], float)
    columns = [f'top-{a.top}', f'top-{a.top} + NONE', 'shortlist']
    for name, m in groups:
        table(f'Recall of the true code, {name}',
              columns, {c: four_slices(v, m, masks) for c, v in zip(columns, (in_top, in_top_none, in_list))})


if __name__ == '__main__':
    main()
