"""Second-teacher labelling with an OpenAI vision model, same prompt and answer format as claude_label.py.

Training and unlabelled photos only; test photos are refused. The key is read from
~/.config/k12/openai_key and never printed. Usage is logged per photo.
"""
import argparse
import json
from pathlib import Path
import time

import openai

from claude_label import encoded, frame, system_prompt
from k12.core import label_space, read_csv

OUT = Path('runs/openai')


def client():
    return openai.OpenAI(api_key=(Path.home() / '.config/k12/openai_key').read_text().strip(), max_retries=3)


def response_format(classes):
    code = {'type': 'string', 'enum': classes}
    return {'type': 'json_schema', 'json_schema': {'name': 'diagnosis', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['transcription', 'root_line', 'reasoning', 'first_code', 'second_code', 'third_code'],
        'properties': {'transcription': {'type': 'string'}, 'root_line': {'type': 'string'},
                       'reasoning': {'type': 'string'}, 'first_code': code, 'second_code': code,
                       'third_code': code}}}}


def body(row, images, split, system, classes, model, effort):
    content = []
    if str(row.question_image_file).strip():
        content += [{'type': 'text', 'text': 'Question diagram:'},
                    image_part(encoded(images / 'questions' / row.question_image_file))]
    content += [{'type': 'text', 'text': "Student's handwritten working:"}, image_part(encoded(images / split / row.image_file)),
                {'type': 'text', 'text': f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\n'
                                         f'Answer key: {row.correct_answer}'}]
    return {'model': model, 'reasoning_effort': effort, 'max_completion_tokens': 16000,
            'response_format': response_format(classes),
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]}


def image_part(block):
    source = block['source']
    return {'type': 'image_url', 'image_url': {'url': f"data:{source['media_type']};base64,{source['data']}"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pilot'])
    parser.add_argument('--name', required=True)
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--effort', default='medium')
    parser.add_argument('--data', default='data/bundle/x/competition_data')
    parser.add_argument('--split', default='train', choices=['train', 'unlabelled', 'test'])
    parser.add_argument('--ids', help='JSON list of trace_ids')
    parser.add_argument('--pilot', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    taxonomy = read_csv(Path('data/raw/taxonomy_v1.csv'))
    classes = label_space(taxonomy)
    system = system_prompt(taxonomy)
    images = Path(args.data) / 'images'
    api = client()
    rows = frame(args).head(args.pilot)
    with (OUT / f'{args.name}.jsonl').open('a') as out:
        for row in rows.itertuples(index=False):
            start = time.monotonic()
            reply = api.chat.completions.create(**body(row, images, args.split, system, classes, args.model, args.effort))
            choice = reply.choices[0]
            answer = json.loads(choice.message.content) if choice.finish_reason == 'stop' else {'finish_reason': choice.finish_reason}
            usage = reply.usage.model_dump()
            out.write(json.dumps({'trace_id': row.trace_id, 'mode': 'pilot', 'model': args.model,
                                  'usage': usage, 'answer': answer}) + '\n')
            details = usage.get('prompt_tokens_details') or {}
            print(json.dumps({'trace_id': row.trace_id, 'seconds': round(time.monotonic() - start, 1),
                              'prompt_tokens': usage['prompt_tokens'], 'cached': details.get('cached_tokens'),
                              'completion_tokens': usage['completion_tokens'],
                              'codes': [answer.get(k) for k in ('first_code', 'second_code', 'third_code')]}), flush=True)


if __name__ == '__main__':
    main()
