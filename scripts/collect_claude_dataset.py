"""Gather every Claude-labelled photo and its answer into one self-contained folder.

data/claude_dataset/
  images/{train,unlabelled,questions}/   hard links to the original photos (no extra disk space)
  labels.jsonl   one row per (run, photo): question metadata, image path, Claude's answer, gold code for train
  labels.csv     the same without the long text fields
  summary.json   counts per run and split
Rerun after new batches are collected; it only adds what is missing. Never includes test photos.
Competition data: keep this folder private and off any upload or cloud sync.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import pandas as pd

from k12.core import read_csv

RUNS = {  # run name -> (split, model)
    'dev_opus': ('train', 'claude-opus-5-5'), 'dev_fable52': ('train', 'claude-fable-5-2'),
    'train_explain_opus': ('train', 'claude-opus-5-5'),
    'unl_opus': ('unlabelled', 'claude-opus-5-5'), 'unl2_opus': ('unlabelled', 'claude-opus-5-5'),
    'unl3_opus': ('unlabelled', 'claude-opus-5-5'), 'unl3b_opus': ('unlabelled', 'claude-opus-5-5')}
SOURCES = {'train': Path('data/bundle/x/competition_data'), 'unlabelled': Path('data/pseudo/x/competition_data')}


def link(source, target):
    if target.exists():
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/claude_dataset')
    args = parser.parse_args()
    out = Path(args.output)
    tables = {'train': read_csv(SOURCES['train'] / 'train.csv').set_index('trace_id'),
              'unlabelled': read_csv(SOURCES['unlabelled'] / 'unlabelled.csv').set_index('trace_id')}
    dev = set(json.loads(Path('data/v3/dev_ids.json').read_text()))
    rows, linked, missing = [], 0, Counter()
    for run, (split, model) in RUNS.items():
        path = Path('runs/claude') / f'{run}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            record = json.loads(line)
            meta = tables[split].loc[record['trace_id']]
            source = SOURCES[split] / 'images' / split / meta.image_file
            if not source.exists():
                missing[run] += 1
                continue
            linked += link(source, out / 'images' / split / meta.image_file)
            question = str(meta.question_image_file).strip()
            if question:
                linked += link(SOURCES[split] / 'images/questions' / question, out / 'images/questions' / question)
            answer = record['answer']
            rows.append({
                'run': run, 'model': model, 'split': split, 'trace_id': record['trace_id'],
                'dev_fold0': record['trace_id'] in dev, 'question_id': meta.question_id, 'grade': int(meta.grade),
                'topic_name': meta.topic_name, 'question_text': meta.question_text,
                'correct_answer': meta.correct_answer, 'image': f'images/{split}/{meta.image_file}',
                'question_image': f'images/questions/{question}' if question else '',
                'gold_root_code': meta.root_code if split == 'train' else '',
                'first_code': answer.get('first_code', meta.root_code if run == 'train_explain_opus' else ''),
                'second_code': answer.get('second_code', ''), 'third_code': answer.get('third_code', ''),
                'root_line': answer.get('root_line', ''), 'reasoning': answer.get('reasoning', ''),
                'transcription': answer.get('transcription', ''), 'stop_reason': answer.get('stop_reason', ''),
                'mode': record['mode'], 'usd': round(record['usd'], 5),
                'output_tokens': record['usage']['output_tokens']})
    out.mkdir(parents=True, exist_ok=True)
    (out / 'labels.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    frame = pd.DataFrame(rows)
    frame.drop(columns=['reasoning', 'transcription', 'question_text']).to_csv(out / 'labels.csv', index=False)
    summary = {'rows': len(rows), 'photos': int(frame.trace_id.nunique()), 'newly_linked_files': linked,
               'missing_images': dict(missing), 'test_photos': 0,
               'by_run': frame.groupby('run').agg(rows=('trace_id', 'size'), usd=('usd', 'sum'),
                                                  error_first=('first_code', lambda s: int((s != 'NONE').sum())))
                      .round(2).to_dict('index')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
