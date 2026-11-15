"""Turn Claude's answers on unlabelled photos into distillation rows for pod_train --pseudo.

Each row keeps Claude's three ranked codes as a soft target and its transcription and
reasoning as an evidence-first reading target, in the dataset-v3 format.

Optional confidence weighting: on the 301 development photos, error answers longer than
~700 output tokens were right first-time only 47% of the time (75% for shorter ones), so
--long-tokens/--long-weight down-weight them; --rare-weight up-weights rows whose first
code has few training examples, where Claude is far stronger than our models.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from k12.core import label_space, read_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('answers', nargs='?', default='runs/claude/unl_opus.jsonl')
    parser.add_argument('--output', default='data/v3/claude_unl.jsonl')
    parser.add_argument('--report', default='reports/claude_distill.json')
    parser.add_argument('--shares', type=float, nargs=3, default=[.7, .2, .1])
    parser.add_argument('--weight', type=float, default=.8)
    parser.add_argument('--none-share', type=float, default=1 / 3,
                        help='cap NONE-first rows to this share; the unlabelled pool has far more correct work than test')
    parser.add_argument('--long-tokens', type=int, help='error answers with more output tokens count as low confidence')
    parser.add_argument('--long-weight', type=float, default=.5, help='weight of low-confidence error rows')
    parser.add_argument('--rare-count', type=int, default=5, help='codes with at most this many training rows are rare')
    parser.add_argument('--rare-weight', type=float, help='weight of rows whose first code is rare')
    args = parser.parse_args()
    taxonomy = read_csv('data/raw/taxonomy_v1.csv')
    names = taxonomy.set_index('id').name.to_dict()
    names['NONE'] = 'no major misconception'
    classes = set(label_space(taxonomy))
    train_counts = Counter(read_csv('data/raw/train.csv').root_code)
    rare = {c for c in classes if train_counts[c] <= args.rare_count and c != 'NONE'}
    rows, skipped, firsts, weights = [], Counter(), Counter(), Counter()
    for line in Path(args.answers).read_text().splitlines():
        record = json.loads(line)
        answer = record['answer']
        codes = [answer.get(k) for k in ('first_code', 'second_code', 'third_code')]
        if not codes[0] or not set(codes) <= classes:
            skipped[answer.get('stop_reason', 'invalid_codes')] += 1
            continue
        soft = {}
        for share, code in zip(args.shares, codes):
            soft[code] = soft.get(code, 0) + share
        first = codes[0]
        weight, kind = args.weight, 'base'
        if args.long_tokens and first != 'NONE' and record['usage']['output_tokens'] > args.long_tokens:
            weight, kind = args.long_weight, 'long'
        if args.rare_weight and first in rare:
            weight, kind = args.rare_weight, 'rare' if kind == 'base' else 'rare_long'
        head = (f'ROOT: {first} ({names[first]})' if first == 'NONE' else
                f"ROOT: {first} ({names[first]})\nWHERE: {answer['root_line'].strip()}\nWHY: {answer['reasoning'].strip()}")
        target = f"{head}\nREADING:\n{answer['transcription'].strip()}"
        rows.append({'trace_id': record['trace_id'], 'weight': weight, 'soft': soft, 'target': target})
        firsts[first] += 1
        weights[kind] += first != 'NONE'
    errors = [r for r in rows if not r['target'].startswith('ROOT: NONE')]
    nones = [r for r in rows if r['target'].startswith('ROOT: NONE')]
    keep_none = int(args.none_share / (1 - args.none_share) * len(errors))
    nones = sorted(nones, key=lambda r: -r['soft'].get('NONE', 0))[:keep_none]
    rows = errors + nones
    firsts['NONE'] = len(nones)
    out = Path(args.output)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'rows': len(rows), 'skipped': dict(skipped), 'none_first_share': firsts['NONE'] / max(1, len(rows)),
              'distinct_first_codes': len(firsts), 'most_common_first': firsts.most_common(8),
              'error_rows_by_weight_kind': dict(weights),
              'settings': {k: v for k, v in vars(args).items()}}
    # Rare-code coverage: how many Claude first-code rows each rare code gains over its training count.
    report['rare_codes'] = {c: {'train': train_counts[c], 'claude_first': firsts[c]} for c in sorted(rare)}
    report['rare_codes_without_claude_rows'] = sorted(c for c in rare if not firsts[c])
    report['codes_without_any_rows'] = sorted(c for c in classes if not firsts[c] and not train_counts[c])
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rare_codes'}, indent=1))


if __name__ == '__main__':
    main()
