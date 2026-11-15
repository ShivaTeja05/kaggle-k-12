"""Pick the next unlabelled photos to send to Claude: error-rich and spread across codes.

Each remaining photo is scored by its question's root-code distribution (Claude's labels
on already-labelled photos of that question plus the training labels, smoothed toward
the global mix). A greedy pass then takes photos whose expected codes are still scarce,
with a per-question cap so the batch does not pile onto a few questions.
CPU only; image sizes are read from the tar listing and the chosen images are extracted.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from k12.core import read_csv


def labelled(paths):
    first = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if 'first_code' in record['answer']:
                first[record['trace_id']] = record['answer']['first_code']
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers', nargs='+', default=['runs/claude/unl_opus.jsonl', 'runs/claude/unl2_opus.jsonl'])
    parser.add_argument('--pool', default='data/semisupervised/competition_data/unlabelled.csv')
    parser.add_argument('--count', type=int, default=9000)
    parser.add_argument('--cap', type=int, default=150, help='max photos per question')
    parser.add_argument('--smooth', type=float, default=5., help='pseudo-count toward the global code mix')
    parser.add_argument('--scarcity', type=float, default=10., help='value of a code ~ 1/sqrt(count + scarcity)')
    parser.add_argument('--output', default='data/pseudo/subset3')
    parser.add_argument('--tar', default='data/bundle/unl/k12_unlabelled.tar')
    parser.add_argument('--extract-to', default='data/pseudo/x', help='where claude_label --data finds images')
    parser.add_argument('--no-extract', action='store_true')
    args = parser.parse_args()

    pool = read_csv(args.pool)
    train = read_csv('data/raw/train.csv')
    first = labelled(args.answers)
    pool['claude'] = pool.trace_id.map(first)
    done = pool[pool.claude.notna()]
    rest = pool[pool.claude.isna()]

    seen = pd.concat([done[['question_id', 'claude']].rename(columns={'claude': 'code'}),
                      train[['question_id', 'root_code']].rename(columns={'root_code': 'code'})])
    codes = sorted(seen.code.unique())
    overall = seen.code.value_counts(normalize=True).reindex(codes).fillna(0).to_numpy()
    table = pd.crosstab(seen.question_id, seen.code).reindex(columns=codes).fillna(0)
    dist = (table.to_numpy() + args.smooth * overall) / (table.sum(1).to_numpy()[:, None] + args.smooth)
    dist = dict(zip(table.index, dist))
    none = codes.index('NONE')
    have = np.array([Counter(done.claude).get(c, 0) + Counter(train.root_code).get(c, 0) for c in codes], float)

    left = rest.groupby('question_id').size().to_dict()
    questions = [q for q in left if q in dist]
    taken, expected, picks = Counter(), np.zeros(len(codes)), []
    for _ in range(args.count):
        value = {q: np.delete(dist[q] / np.sqrt(have + args.scarcity), none).sum()
                 for q in questions if taken[q] < min(args.cap, left[q])}
        if not value:
            break
        best = max(value, key=value.get)
        taken[best] += 1
        picks.append(best)
        have += dist[best]
        expected += dist[best]

    # Photos are listed in pick order, so any prefix (claude_label --limit) is itself a balanced selection.
    # Within a question the labelled photos were a random draw, so take the remaining ones in a fixed random order.
    queues = {q: list(group.index) for q, group in rest.sample(frac=1., random_state=0).groupby('question_id')}
    chosen = rest.loc[[queues[q].pop(0) for q in picks]]
    ids = chosen.trace_id.tolist()
    out = Path(args.output)
    out.with_name(out.name + '_ids.json').write_text(json.dumps(ids))
    chosen.drop(columns='claude').to_csv(out.with_suffix('.csv'), index=False)

    train_counts = train.root_code.value_counts()
    rare = set(train_counts[train_counts <= 5].index)
    report = {'remaining_pool': len(rest), 'chosen': len(ids), 'questions': len(taken),
              'max_per_question': max(taken.values()),
              'expected_error_rows': round(len(ids) - expected[none]),
              'expected_error_share': round(1 - expected[none] / max(1, len(ids)), 3),
              'expected_rare_code_rows': round(sum(expected[codes.index(c)] for c in rare if c in codes), 1),
              'expected_top_codes': {codes[i]: round(expected[i]) for i in np.argsort(-expected)[:12]},
              'settings': {k: v for k, v in vars(args).items() if k not in ('answers',)}}
    out.with_name(out.name + '_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))

    if not args.no_extract:
        target = Path(args.extract_to)
        wanted = {f'competition_data/images/unlabelled/{f}' for f in chosen.image_file}
        present = {f for f in wanted if (target / f).exists()}
        with tarfile.open(args.tar) as tar:
            members = [m for m in tar if m.name in wanted - present]
            tar.extractall(target, members=members, filter='data')
        print(json.dumps({'extracted': len(members), 'already_present': len(present)}))


if __name__ == '__main__':
    main()
