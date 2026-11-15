"""Build chat requests for an offline open-weight vision model, with the same prompt and photos Claude was given.

One JSON line per student photo: {"trace_id", "messages"} in the OpenAI chat format (images as data URLs), plus one
header line holding the answer schema. Default: the development fold of the training photos.

Test photos (--split test --ids FILE) are for our own offline model on our own pod only, never for an API; the user
approved that run on 14 Nov 2026. --parts N writes N files (rows dealt out in order) for uploading in pieces.

Usage: python scripts/build_vlm_requests.py --out data/v9/q38_dev_requests.jsonl
       python scripts/build_vlm_requests.py --split test --ids ids.json --parts 8 --out data/v9/q38_test/part.jsonl
"""
import argparse
import json
from pathlib import Path

from claude_label import encoded, system_prompt
from k12.core import label_space, make_folds, read_csv


def image_part(path):
    source = encoded(path)['source']
    return {'type': 'image_url', 'image_url': {'url': f"data:{source['media_type']};base64,{source['data']}"}}


def user_content(row, images, split):
    content = []
    if str(row.question_image_file).strip():
        content += [{'type': 'text', 'text': 'Question diagram:'}, image_part(images / 'questions' / row.question_image_file)]
    content += [{'type': 'text', 'text': "Student's handwritten working:"}, image_part(images / split / row.image_file),
                {'type': 'text', 'text': f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\n'
                                         f'Answer key: {row.correct_answer}'}]
    return content


def answer_schema(classes):
    code = {'type': 'string', 'enum': classes}
    fields = {'transcription': {'type': 'string'}, 'root_line': {'type': 'string'}, 'reasoning': {'type': 'string'},
              'first_code': code, 'second_code': code, 'third_code': code}
    return {'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/bundle/x/competition_data')
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument('--ids', help='JSON list of trace_ids, in the order to send them (required for test)')
    parser.add_argument('--parts', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0, help='development fold of make_folds; -1 = every training row (use with --ids)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    data = Path(args.data)
    taxonomy = read_csv(data / 'taxonomy_v1.csv')
    classes = label_space(taxonomy)
    if args.split == 'test':
        if not args.ids:
            raise SystemExit('--split test needs --ids')
        rows = read_csv(data / 'test.csv').set_index('trace_id').loc[json.loads(Path(args.ids).read_text())].reset_index()
    else:
        train = read_csv(data / 'train.csv')
        rows = train if args.fold < 0 else train.loc[make_folds(train) == args.fold].reset_index(drop=True)
        if args.ids:
            rows = rows.set_index('trace_id').loc[json.loads(Path(args.ids).read_text())].reset_index()
    system = system_prompt(taxonomy) + ('\nAnswer with one JSON object with exactly these keys: "transcription", "root_line", '
                                        '"reasoning", "first_code", "second_code", "third_code". Each code is a taxonomy '
                                        'code or NONE, and the three codes are different.\n')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for part in range(args.parts):
        chunk = rows.iloc[part::args.parts]
        target = out if args.parts == 1 else out.with_name(f'{out.stem}_{part + 1}{out.suffix}')
        with target.open('w') as handle:
            handle.write(json.dumps({'header': True, 'schema': answer_schema(classes), 'classes': classes, 'rows': len(chunk)}) + '\n')
            for row in chunk.itertuples(index=False):
                messages = [{'role': 'system', 'content': system},
                            {'role': 'user', 'content': user_content(row, data / 'images', args.split)}]
                handle.write(json.dumps({'trace_id': row.trace_id, 'messages': messages}) + '\n')
        print({'rows': len(chunk), 'out': str(target), 'mb': round(target.stat().st_size / 1e6, 1), 'system_chars': len(system)})


if __name__ == '__main__':
    main()
