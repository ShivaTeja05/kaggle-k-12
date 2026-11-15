"""Variant of the dataset-v3 training file whose WHY lines are Claude's explanations of the gold root code.

Claude (claude_label.py --explain) was given the annotators' transcription and root code and
asked only why that code is the root. Its explanation replaces the annotators' short WHY on
error rows; ROOT, WHERE, READING, NONE rows and the development fold are left unchanged.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/v3/train_v3.jsonl')
    parser.add_argument('--answers', default='runs/claude/train_explain_opus.jsonl')
    parser.add_argument('--output', default='data/v3/train_v3_claude.jsonl')
    args = parser.parse_args()
    dev = set(json.loads(Path('data/v3/dev_ids.json').read_text()))
    why = {}
    for line in Path(args.answers).read_text().splitlines():
        record = json.loads(line)
        if record['trace_id'] not in dev and record['answer'].get('reasoning', '').strip():
            why[record['trace_id']] = ' '.join(record['answer']['reasoning'].split())
    rows, replaced = [], 0
    for line in Path(args.train).read_text().splitlines():
        row = json.loads(line)
        lines = row['target'].split('\n')
        index = next((i for i, text in enumerate(lines) if text.startswith('WHY: ')), None)
        if row['trace_id'] in why and row['root_code'] != 'NONE' and index is not None:
            lines[index] = f"WHY: {why[row['trace_id']]}"
            row['target'] = '\n'.join(lines)
            replaced += 1
        rows.append(row)
    Path(args.output).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'rows': len(rows), 'why_replaced': replaced, 'explanations': len(why)}))


if __name__ == '__main__':
    main()
