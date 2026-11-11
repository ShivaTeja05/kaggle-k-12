"""Build the v3 training table: evidence-first teacher targets, soft labels, families and weights.

Every field is derived from the labelled training annotations only. Targets are
supervision; prediction inputs stay the photo, question, answer key, grade and topic.
Writes rows to data/v3/train_v3.jsonl (private) and counts to reports/dataset_v3.json.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

from k12.core import read_csv


def normalize(text):
    # Unicode composition and whitespace only; signs, decimals and ordering survive.
    return re.sub(r'[ \t]+', ' ', unicodedata.normalize('NFC', str(text))).strip()


def lines(transcript):
    found = {}
    for line in normalize(transcript).splitlines():
        match = re.match(r'\s*(L\d+)\s*\|\s*(.*)', line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def evidence_target(row, findings, names):
    """Root step and explanation first, so truncation keeps the decision-relevant part."""
    transcript = normalize(row.transcription_steps)
    if row.root_code == 'NONE':
        answer = 'matches' if str(row.final_answer_correct).lower() == 'true' else 'does not match'
        head = f'ROOT: NONE (no major misconception)\nFINAL ANSWER: {answer} the key'
    else:
        root = next((f for f in findings if f.get('root')), {})
        step = root.get('step') or '?'
        span = normalize(root.get('span') or lines(row.transcription_steps).get(step, ''))
        head = (f"ROOT: {row.root_code} ({names.get(row.root_code, '')})\n"
                f"WHERE: {step} | {span}\nWHY: {normalize(root.get('text', ''))}")
    return f'{head}\nREADING:\n{transcript}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--output', default='data/v3/train_v3.jsonl')
    parser.add_argument('--rare-floor', type=int, default=3, help='codes below this count get weight floor/count')
    args = parser.parse_args()
    data = Path(args.data)
    train = read_csv(data / 'train.csv')
    taxonomy = read_csv(data / 'taxonomy_v1.csv')
    names = taxonomy.set_index('id').name.to_dict()
    counts = train.root_code.value_counts()

    rows, stats = [], Counter()
    for row in train.itertuples(index=False):
        findings = json.loads(str(row.misconceptions).replace('NaN', 'null'))
        major = [f for f in findings if f.get('severity') == 'major']
        secondary = sorted({f['category'] for f in major if not f.get('root') and f.get('category') in names}
                           - {row.root_code})
        transcript = normalize(row.transcription_steps).lower()
        flags = {'unclear': bool(re.search(r'unclear|illegible|\?\?', transcript)),
                 'crossed_out': 'cross' in transcript,
                 'multiple_major': len(major) > 1,
                 'answer_correct_but_error': row.root_code != 'NONE' and str(row.final_answer_correct).lower() == 'true',
                 'answer_wrong_but_none': row.root_code == 'NONE' and str(row.final_answer_correct).lower() != 'true'}
        root = next((f for f in major if f.get('root')), None)
        flags['root_step_missing'] = row.root_code != 'NONE' and not (root and root.get('step'))
        target = evidence_target(row, major, names)
        rows.append({'trace_id': row.trace_id, 'root_code': row.root_code,
                     'family': row.root_code.split('-', 1)[0].upper(), 'has_error': row.root_code != 'NONE',
                     'secondary_codes': secondary,
                     'weight': float(max(1., args.rare_floor / counts[row.root_code])),
                     'target': target, 'flags': flags})
        stats.update(k for k, v in flags.items() if v)
        stats['rows_with_secondary'] += bool(secondary)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    lengths = np.asarray([len(r['target']) for r in rows])
    head_lengths = np.asarray([len(r['target'].split('\nREADING:')[0]) for r in rows])
    report = {'rows': len(rows), 'flag_counts': dict(stats),
              'families': len({r['family'] for r in rows}),
              'upweighted_rows': int(sum(r['weight'] > 1 for r in rows)),
              'target_chars_quantiles': np.quantile(lengths, [.5, .9, .99]).tolist(),
              'evidence_head_chars_quantiles': np.quantile(head_lengths, [.5, .9, .99]).tolist(),
              'note': 'Targets are supervision only. No test rows are read or labelled.'}
    Path('reports/dataset_v3.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    print('\nExample target:\n' + next(r['target'] for r in rows if r['root_code'] != 'NONE')[:600])


if __name__ == '__main__':
    main()
