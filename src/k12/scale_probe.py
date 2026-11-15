"""Zero-shot scale probe: a large open reasoner reads machine transcripts and ranks three codes.

No training. Measures how much raw model scale helps diagnosis before paying to fine-tune.
Development rows only (fold 0); outputs rank-based scores in the usual bundle format.
"""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np

from .core import label_space, make_folds, map3, normalize, read_csv, save_bundle


def system_prompt(taxonomy):
    rows = '\n'.join(f'{r.id} | {r.name} | {r.description}' for r in taxonomy.itertuples(index=False))
    return f"""You are an experienced mathematics teacher diagnosing Indonesian K-12 students' written working.

Identify the ROOT misconception: the major error that the rest of the working follows from (not necessarily the first error written). Answer NONE when there is no major error: fully correct work, or only minor slips such as imprecise notation that do not change the result. About a third of workings are NONE. A correct final answer does not prove correct reasoning. Ignore crossed-out work. The transcript was produced by a machine reader and may contain small reading mistakes.

Check each line against the question, the answer key and the previous line; find where a major error enters; choose the code whose description best matches it.

Taxonomy (code | name | description). NONE means no major misconception.
{rows}

End your reply with one final line, exactly in this form, giving your three most likely codes from best to worst:
CODES: <code1> <code2> <code3>"""


def labelled_example(r):
    """A training row as an in-context example: gold transcript, root code and the annotator's reason."""
    reason = next((m.get('text', '') for m in json.loads(r.misconceptions or '[]') if m.get('root')), '')
    answer = f'{r.root_code}: {reason}' if reason and r.root_code != 'NONE' else r.root_code
    return (f'Question: {r.question_text}\nAnswer key: {r.correct_answer}\nStudent working:\n'
            f'{r.transcription_steps}\nRoot code: {answer}')


def pick_examples(row, pool, shots):
    """Same-question rows first, then same-topic rows; pool holds labelled rows outside the scored split."""
    same = pool[pool.question_id == row.question_id]
    topic = pool[(pool.topic_name == row.topic_name) & (pool.question_id != row.question_id)]
    return list(same.head(shots).itertuples(index=False)) + \
        list(topic.head(max(0, shots - len(same))).itertuples(index=False))


def parse(text, canonical):
    lines = [line for line in text.splitlines() if 'CODES:' in line.upper()]
    if not lines:
        return []
    found = []
    for token in re.findall(r'[A-Za-z]{2,4}-\d{2}|NONE', lines[-1].split(':', 1)[-1]):
        code = canonical.get(token.upper())
        if code and code not in found:
            found.append(code)
    return found[:3]


def main():
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--transcripts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tensor-parallel', type=int, default=2)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=3000)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev',
                        help='dev: fold-0 rows, examples from other folds; test: test.csv, examples from all training rows')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-model-len', type=int, default=12288)
    parser.add_argument('--gpu-memory', type=float, default=.92)
    parser.add_argument('--chunk', type=int, default=60, help='rows per generate call; results saved after each')
    parser.add_argument('--deadline', type=float, help='epoch seconds; no chunk starts if it would end later '
                        '(default: contents of <output>/../probe_deadline when present)')
    parser.add_argument('--shots', type=int, default=0,
                        help='labelled training examples (non-development folds) shown before each row')
    parser.add_argument('--rerun-unparsed', action='store_true',
                        help='regenerate only rows without parsed codes in <output>/responses.jsonl')
    args = parser.parse_args()
    deadline_file = Path(args.output).parent / 'probe_deadline'
    if args.deadline is None and deadline_file.exists():
        args.deadline = float(deadline_file.read_text().strip())
    data = Path(args.data)
    train, taxonomy = read_csv(data / 'train.csv'), read_csv(data / 'taxonomy_v1.csv')
    classes = label_space(taxonomy)
    canonical = {c.upper(): c for c in classes}
    folds = make_folds(train)
    validation = train.loc[folds == 0].reset_index(drop=True)
    pool = train.loc[folds != 0].reset_index(drop=True)
    if args.split == 'test':
        validation, pool = read_csv(data / 'test.csv'), train
    if args.limit:
        validation = validation.head(args.limit)
    transcripts = json.loads(Path(args.transcripts).read_text())
    system = system_prompt(taxonomy)
    def user_message(r):
        target = (f'Grade: {r.grade}\nTopic: {r.topic_name}\nQuestion: {r.question_text}\n'
                  f'Answer key: {r.correct_answer}\nStudent working (line by line):\n{transcripts[r.trace_id]}')
        examples = pick_examples(r, pool, args.shots) if args.shots else []
        if not examples:
            return target
        shown = '\n\n'.join(f'### Example {i}\n{labelled_example(e)}' for i, e in enumerate(examples, 1))
        return (f'Teacher-labelled examples from other students (exact human transcripts):\n\n{shown}\n\n'
                f'### Student to diagnose (machine transcript)\n{target}')

    conversations = [[{'role': 'system', 'content': system}, {'role': 'user', 'content': user_message(r)}]
                     for r in validation.itertuples(index=False)]
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory, enable_prefix_caching=True, seed=2026)
    sampling = SamplingParams(temperature=.6, top_p=.95, top_k=20, max_tokens=args.max_tokens, seed=2026)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    index = {c: i for i, c in enumerate(classes)}
    # One record per row; --rerun-unparsed keeps parsed rows from the earlier pass and regenerates the rest.
    records = [None] * len(validation)
    if args.rerun_unparsed:
        earlier = {}
        for line in (out / 'responses.jsonl').read_text().splitlines():
            record = json.loads(line)
            earlier[record['trace_id']] = record
        (out / 'responses_pass1.jsonl').write_text((out / 'responses.jsonl').read_text())
        (out / 'metrics_pass1.json').write_text((out / 'metrics.json').read_text())
        records = [earlier.get(t) if earlier.get(t, {}).get('codes') else None for t in validation.trace_id]
    # Question order keeps rows with identical few-shot examples together, so vLLM's prefix cache shares them.
    todo = sorted((row for row, record in enumerate(records) if record is None),
                  key=lambda row: (validation.question_id[row], row))
    print(f'generating {len(todo)} rows', flush=True)
    done, tokens, start = 0, 0, time.monotonic()
    for begin in range(0, len(todo), args.chunk):
        elapsed = time.monotonic() - start
        if args.deadline and done and time.time() + elapsed / done * args.chunk * 1.2 > args.deadline:
            print(f'deadline: stopping after {done} rows', flush=True)
            break
        rows = todo[begin:begin + args.chunk]
        outputs = llm.chat([conversations[row] for row in rows], sampling,
                           chat_template_kwargs={'enable_thinking': args.thinking})
        for row, o in zip(rows, outputs):
            text = o.outputs[0].text
            tokens += len(o.outputs[0].token_ids)
            records[row] = {'trace_id': validation.trace_id[row], 'codes': parse(text, canonical),
                            'finish': o.outputs[0].finish_reason, 'tokens': len(o.outputs[0].token_ids),
                            'max_tokens': args.max_tokens, 'tail': text[-300:]}
        done += len(rows)
        # Unscored rows keep a flat floor so the bundle stays aligned with all development rows.
        scores = np.full((len(validation), len(classes)), 1e-4)
        for row, record in enumerate(records):
            for share, code in zip((.6, .25, .15), (record or {}).get('codes', [])):
                scores[row, index[code]] = share
        probabilities = normalize(scores)
        have = [row for row, record in enumerate(records) if record is not None]
        save_bundle(out / ('test.npz' if args.split == 'test' else 'validation.npz'), validation.trace_id.tolist(),
                    classes, probabilities, kind='zero_shot_scale_probe', scored_rows=len(have))
        (out / 'responses.jsonl').write_text(''.join(json.dumps(records[row]) + '\n' for row in have))
        metrics = {'model': args.model, 'thinking': args.thinking, 'rows': len(validation), 'scored_rows': len(have),
                   'split': args.split, 'shots': args.shots,
                   'map3': map3(validation.root_code.iloc[have], probabilities[have], classes)
                   if 'root_code' in validation else None,
                   'none_first_share': float(np.mean([records[row]['codes'][:1] == ['NONE'] for row in have])),
                   'unparsed': sum(not records[row]['codes'] for row in have),
                   'truncated': sum(records[row].get('finish', 'length' if not records[row]['codes'] else 'stop')
                                    == 'length' for row in have),
                   'generated_rows': done, 'seconds': round(time.monotonic() - start), 'output_tokens': tokens,
                   'rerun_unparsed': args.rerun_unparsed}
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
