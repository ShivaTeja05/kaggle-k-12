"""Label labelled or unlabelled student photos with Claude: transcript, reasoning, three ranked codes.

Training and unlabelled photos only; test photos are refused. Spend is tracked in
reports/claude_budget.json and every step refuses to run past the approved limit.
The key is read from ~/.config/k12/anthropic_key and never printed.
"""
import argparse
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zlib

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
import pandas as pd
from PIL import Image, ImageOps

from k12.core import label_space, make_folds, read_csv

MODEL = 'claude-opus-5-5'
# USD per million tokens (Anthropic pricing page, 11 Nov 2026); batch requests cost half.
PRICES = {'claude-opus-5-5': {'input_tokens': 4.0, 'output_tokens': 20.0, 'cache_read_input_tokens': .2,
                              'cache_creation_input_tokens': 5.0},
          'claude-fable-5-2': {'input_tokens': 8.0, 'output_tokens': 40.0, 'cache_read_input_tokens': .2,
                               'cache_creation_input_tokens': 10.0}}
RATES = PRICES[MODEL]
BUDGET = Path('reports/claude_budget.json')
KEY = Path.home() / '.config/k12/anthropic_key'
OUT = Path('runs/claude')


def client():
    key = KEY.read_text().strip()
    return anthropic.Anthropic(api_key=key, max_retries=4)


def cost(usage, batch):
    total = sum((getattr(usage, field, 0) or 0) * rate / 1e6 for field, rate in RATES.items())
    return total * (.5 if batch else 1.)


def budget():
    return json.loads(BUDGET.read_text())


def record(amount, note):
    state = budget()
    state['spent_usd'] = round(state['spent_usd'] + amount, 4)
    state['log'].append({'at': datetime.now(timezone.utc).isoformat(), 'usd': round(amount, 4), 'note': note})
    BUDGET.write_text(json.dumps(state, indent=2) + '\n')
    return state


def system_prompt(taxonomy):
    rows = '\n'.join(f'{r.id} | {r.name} | {r.description} | {r.nature} | grade band {r.grade_band} | {r.section}'
                     for r in taxonomy.itertuples(index=False))
    return f"""You are an experienced mathematics teacher diagnosing Indonesian K-12 students' handwritten working.

For each photo, identify the ROOT misconception: the major error that the rest of the working follows from. It is not necessarily the first error written on the page. A root is always a major error, one that changes the reasoning or the result. Answer NONE when the working has no major error: it is fully correct, or carries only minor slips such as imprecise notation that do not change the result. About a third of workings are NONE, and false alarms on correct work are as harmful as missed errors. A correct final answer does not prove the reasoning is correct, and a wrong final answer can come from a minor slip. Respect crossed-out work: only the student's final, uncrossed reasoning counts. Do not invent steps that are not written.

Procedure:
1. Transcribe the student's working line by line as "L1 | ...", "L2 | ...", preserving the student's own symbols and mistakes exactly; never correct them. Mark unreadable parts as [unclear].
2. Check each line against the question, the answer key and the previous line. Find the first line where a major error enters, and decide whether later errors are consequences of it.
3. Choose the code whose definition best matches that root error; use the descriptions, not just the names. Then give your second and third most likely codes, which may include NONE.

Taxonomy (code | name | description | nature | grade band | section). NONE means no major misconception.
{rows}
"""


def schema(classes, explain=False, text_only=False):
    if explain:
        return {'type': 'json_schema', 'schema': {
            'type': 'object', 'additionalProperties': False,
            'required': ['transcription', 'root_line', 'reasoning'],
            'properties': {'transcription': {'type': 'string'}, 'root_line': {'type': 'string'},
                           'reasoning': {'type': 'string'}}}}
    code = {'type': 'string', 'enum': classes}
    fields = {'transcription': {'type': 'string'}, 'root_line': {'type': 'string'}, 'reasoning': {'type': 'string'},
              'first_code': code, 'second_code': code, 'third_code': code}
    if text_only:  # the transcription is an input here, so it is not written again
        del fields['transcription']
    return {'type': 'json_schema', 'schema': {'type': 'object', 'additionalProperties': False,
                                              'required': list(fields), 'properties': fields}}


def encoded(path, maxside=1024):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGB')
    image.thumbnail((maxside, maxside))
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=90)
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                        'data': base64.standard_b64encode(buffer.getvalue()).decode()}}


def explanation_request(row, names):
    """Training rows only: give the annotators' transcription and root code and ask for the reasoning behind it."""
    code = row.root_code
    if code == 'NONE':
        verdict = 'NONE: the working has no major misconception.'
        task = ('give "none" as the root line, and explain why the working has no major error '
                '(name any minor slips and why they do not change the reasoning or the result).')
    else:
        verdict = f'{code} ({names[code]}).'
        task = ("give the root line, and explain, using the code's definition, why this is the root "
                'and why later errors are consequences of it.')
    return (f"\nAn expert annotator transcribed this working as:\n{row.transcription_steps}\n\n"
            f"The annotators' verified root diagnosis is {verdict} Treat it as correct. Transcribe the working in the "
            f"usual format (you may use the annotator's transcription), {task}")


def few_shot(row, pool, names, k):
    """Up to k annotated training examples (same question first, then same topic), never the row itself.

    pool must hold fold-fit training rows only, so development rows are never shown as examples.
    """
    pool = pool[pool.trace_id != row.trace_id]
    order = lambda part: part.assign(key=part.trace_id.map(lambda t: zlib.crc32(f'{row.trace_id}:{t}'.encode()))).sort_values('key')
    chosen = pd.concat([order(pool[pool.question_id == row.question_id]),
                        order(pool[(pool.question_id != row.question_id) & (pool.topic_name == row.topic_name)])]).head(k)
    if chosen.empty:
        return ''
    blocks = []
    for i, ex in enumerate(chosen.itertuples(index=False), 1):
        findings = json.loads(ex.misconceptions) if str(ex.misconceptions).strip() else []
        root = next((f for f in findings if f.get('root')), None)
        if ex.root_code == 'NONE':
            minor = ' '.join(f"{f['step']}: {f['text']}" for f in findings if f.get('severity') == 'minor')
            why = 'No major error.' + (f' Minor slips only: {minor}' if minor else '')
        else:
            why = f"{root['step']} | {root['span']}: {root['text']}" if root else ''
        blocks.append(f"Example {i}\nQuestion: {ex.question_text}\nAnswer key: {ex.correct_answer}\n"
                      f"Transcription:\n{ex.transcription_steps}\n"
                      f"Root code: {ex.root_code} ({names.get(ex.root_code, 'no major misconception')})\n"
                      f"Annotator's explanation: {why}")
    return ('Annotated examples of other students\' working (verified by expert annotators), for calibration:\n\n'
            + '\n\n'.join(blocks) + '\n\nNow diagnose the following student.')


def text_params(row, transcript, system, classes, effort):
    """No photo: the model reads only a transcription of the working, as a text-only student would.

    The question diagram is left out too, because our text models never see it.
    """
    text = ('No photo is available for this student. Below is a transcription of the handwritten working, produced by '
            'an automatic reader; it can contain reading mistakes such as wrong, missing or reordered lines. Skip step 1 '
            'of the procedure, treat the transcription as the working, and refer to its line labels.\n\n'
            f'Transcription:\n{transcript}\n\n'
            f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\n'
            f'Answer key: {row.correct_answer}')
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=16000,
        system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
        output_config={'effort': effort, 'format': schema(classes, text_only=True)},
        messages=[{'role': 'user', 'content': [{'type': 'text', 'text': text}]}])


def params(row, images, split, system, classes, effort, names=None, shots='', transcripts=None):
    if transcripts is not None:
        return text_params(row, transcripts[row.trace_id], system, classes, effort)
    content = [{'type': 'text', 'text': shots}] if shots else []
    if str(row.question_image_file).strip():
        content += [{'type': 'text', 'text': 'Question diagram:'},
                    encoded(images / 'questions' / row.question_image_file)]
    content += [{'type': 'text', 'text': "Student's handwritten working:"}, encoded(images / split / row.image_file),
                {'type': 'text', 'text': f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\n'
                                         f'Answer key: {row.correct_answer}'
                                         + (explanation_request(row, names) if names else '')}]
    return MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=16000,
        system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
        output_config={'effort': effort, 'format': schema(classes, explain=bool(names))},
        messages=[{'role': 'user', 'content': content}])


def parse(message):
    text = next((b.text for b in message.content if b.type == 'text'), '')
    return json.loads(text) if message.stop_reason == 'end_turn' else {'stop_reason': message.stop_reason}


def frame(args):
    if args.split == 'test':
        raise SystemExit('Test photos are never sent to the API')
    data = Path(args.data)
    rows = read_csv(data / ('train.csv' if args.split == 'train' else 'unlabelled.csv'))
    if args.ids:
        keep = json.loads(Path(args.ids).read_text())
        rows = rows.set_index('trace_id').loc[keep].reset_index()
    return rows


def configure(args):
    """Point the module-level model, prices, key and budget file at this run's choices."""
    global MODEL, RATES, BUDGET, KEY
    MODEL, RATES, BUDGET, KEY = args.model, PRICES[args.model], Path(args.budget_file), Path(args.key_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['pilot', 'submit', 'collect'])
    parser.add_argument('--name', required=True, help='run name, e.g. dev_opus')
    parser.add_argument('--data', default='data/bundle/x/competition_data')
    parser.add_argument('--split', default='train', choices=['train', 'unlabelled', 'test'])
    parser.add_argument('--ids', help='JSON list of trace_ids')
    parser.add_argument('--pilot', type=int, default=5)
    parser.add_argument('--effort', default='medium')
    parser.add_argument('--chunk', type=int, default=700, help='photos per batch (256 MB batch limit)')
    parser.add_argument('--limit', type=int, help='submit at most this many photos')
    parser.add_argument('--per-photo', type=float, help='USD per photo for the budget projection')
    parser.add_argument('--model', default=MODEL, choices=sorted(PRICES))
    parser.add_argument('--key-file', default=str(KEY))
    parser.add_argument('--budget-file', default=str(BUDGET))
    parser.add_argument('--explain', action='store_true',
                        help='training rows: explain the annotators\' root code instead of predicting one')
    parser.add_argument('--fewshot', type=int, default=0,
                        help='prepend up to this many annotated fold-fit training examples (same question, then topic)')
    parser.add_argument('--transcripts', help='JSON {trace_id: transcription}: send this text instead of the photo '
                                              '(training rows only; measures a text-only reader of our transcripts)')
    args = parser.parse_args()
    if args.transcripts and (args.explain or args.fewshot or args.split != 'train'):
        raise SystemExit('--transcripts is for --split train without --explain or --fewshot')
    if args.explain and args.fewshot:
        raise SystemExit('--explain and --fewshot do not combine')
    if args.explain and args.split != 'train':
        raise SystemExit('--explain needs --split train: only training rows carry a verified root code')
    configure(args)
    OUT.mkdir(parents=True, exist_ok=True)
    results = OUT / f'{args.name}.jsonl'
    taxonomy = read_csv(Path('data/raw/taxonomy_v1.csv'))
    classes = label_space(taxonomy)
    system = system_prompt(taxonomy)
    names = taxonomy.set_index('id').name.to_dict() if args.explain else None
    shot = lambda row: ''
    if args.fewshot:
        train = read_csv(Path('data/raw/train.csv'))
        pool = train[make_folds(train) != 0]
        code_names = taxonomy.set_index('id').name.to_dict()
        shot = lambda row: few_shot(row, pool, code_names, args.fewshot)
    images = Path(args.data) / 'images'
    transcripts = json.loads(Path(args.transcripts).read_text()) if args.transcripts else None
    api = client()
    state = budget()

    if args.command == 'pilot':
        rows = frame(args).head(args.pilot)
        spent = 0.
        with results.open('a') as out:
            for row in rows.itertuples(index=False):
                if state['spent_usd'] + spent >= state['limit_usd']:
                    raise SystemExit('Budget reached')
                message = api.messages.create(**params(row, images, args.split, system, classes, args.effort, names, shot(row), transcripts))
                usd = cost(message.usage, batch=False)
                spent += usd
                out.write(json.dumps({'trace_id': row.trace_id, 'mode': 'pilot', 'usd': usd,
                                      'usage': message.usage.to_dict(), 'answer': parse(message)}) + '\n')
                print(json.dumps({'trace_id': row.trace_id, 'usd': round(usd, 4),
                                  'output_tokens': message.usage.output_tokens}), flush=True)
        state = record(spent, f'{args.name} pilot, {len(rows)} photos, standard price')
        per_photo = spent / len(rows)
        print(json.dumps({'pilot_photos': len(rows), 'usd': round(spent, 4),
                          'per_photo_standard': round(per_photo, 4), 'per_photo_batch_estimate': round(per_photo / 2, 4),
                          'spent_total': state['spent_usd'], 'limit': state['limit_usd']}))

    elif args.command == 'submit':
        done = {json.loads(line)['trace_id'] for line in results.read_text().splitlines()} if results.exists() else set()
        pending = {r for b in state['batches'] if b['name'] == args.name and not b['collected'] for r in b.get('ids', [])}
        rows = frame(args)
        rows = rows[~rows.trace_id.isin(done | pending)]
        if args.limit:
            rows = rows.head(args.limit)
        # Observed batch cost per photo for this run (cache hits included), unless given explicitly.
        observed = [json.loads(line)['usd'] for line in (results.read_text().splitlines() if results.exists() else [])
                    if json.loads(line)['mode'] == 'batch']
        if args.per_photo is None and not observed:
            raise SystemExit('No observed batch cost yet; pass --per-photo')
        per_photo = args.per_photo or sum(observed) / len(observed)
        projected = per_photo * 1.3 * len(rows)
        if state['spent_usd'] + sum(b['projected_usd'] for b in state['batches'] if not b['collected']) + projected > state['limit_usd']:
            raise SystemExit(f'Projected ${projected:.2f} for {len(rows)} photos exceeds the remaining budget')
        for start in range(0, len(rows), args.chunk):
            part = rows.iloc[start:start + args.chunk]
            batch = api.messages.batches.create(requests=[
                Request(custom_id=row.trace_id, params=params(row, images, args.split, system, classes, args.effort, names, shot(row), transcripts))
                for row in part.itertuples(index=False)])
            state['batches'].append({'name': args.name, 'id': batch.id, 'rows': len(part), 'ids': part.trace_id.tolist(),
                                     'projected_usd': round(per_photo * 1.3 * len(part), 2), 'collected': False})
            BUDGET.write_text(json.dumps(state, indent=2) + '\n')
            print(json.dumps({'batch': batch.id, 'rows': len(part)}), flush=True)
        print(json.dumps({'submitted_rows': len(rows), 'projected_usd': round(projected, 2)}))

    else:
        open_batches = [b for b in state['batches'] if b['name'] == args.name and not b['collected']]
        ended = [b for b in open_batches if api.messages.batches.retrieve(b['id']).processing_status == 'ended']
        if not ended:
            print(json.dumps({'status': 'in_progress', 'open_batches': len(open_batches)}))
            return
        for entry in ended:
            collect_one(api, args, results, entry)
        print(json.dumps({'collected_batches': len(ended), 'still_open': len(open_batches) - len(ended),
                          'spent_total': budget()['spent_usd']}))


def collect_one(api, args, results, entry):
        spent, counts = 0., {}
        with results.open('a') as out:
            for result in api.messages.batches.results(entry['id']):
                counts[result.result.type] = counts.get(result.result.type, 0) + 1
                if result.result.type != 'succeeded':
                    continue
                message = result.result.message
                usd = cost(message.usage, batch=True)
                spent += usd
                out.write(json.dumps({'trace_id': result.custom_id, 'mode': 'batch', 'usd': usd,
                                      'usage': message.usage.to_dict(), 'answer': parse(message)}) + '\n')
        state = budget()
        for b in state['batches']:
            if b['id'] == entry['id']:
                b['collected'] = True
        BUDGET.write_text(json.dumps(state, indent=2) + '\n')
        state = record(spent, f"{args.name} batch {entry['id']}")
        print(json.dumps({'batch': entry['id'], 'counts': counts, 'usd': round(spent, 4),
                          'spent_total': state['spent_usd'], 'limit': state['limit_usd']}), flush=True)


if __name__ == '__main__':
    main()
