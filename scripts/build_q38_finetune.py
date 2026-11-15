"""Training and scoring files for fine-tuning Qwen3.8-27B on the teachers' labels (photo in, teacher-style answer out).

The answer the model is trained to write is the teachers' own work, in the JSON layout it already uses zero-shot:
  transcription  the teacher's line-by-line transcription
  root_line      the line the teacher marked as the root error ("none" when there is none)
  reasoning      the teacher's note on the root error (the reasoning comes BEFORE the codes, so the model still
                 reasons before it answers; training on bare codes damages a thinking model)
  first_code     the teacher's root code; second/third: what Qwen3.8 guessed zero-shot for that photo, minus the truth
Thinking is switched off for training and scoring: the JSON itself is the reasoning trace.

Prompt, shorter than the zero-shot one so training is affordable: all codes by name in the system message; in the user
message the definitions of the codes most used in the row's topic, and the codes teachers gave other students on the
same question (a hint). A training row never counts itself. A share of training rows is repeated without the question
hint, because dev and test contain unseen questions.

Files (OpenAI chat format, photos as data URLs, same encoding as the zero-shot runs):
  ft_train.jsonl          {"trace_id", "messages", "target", "label"}  fit rows (or all 1,500 with --full); "label" is the
                          teacher's root code, the target of the classification layer; "final_ok" is the teacher's flag
                          "the student's final answer is correct" (a second, yes/no output of that layer: on the 1,500
                          training rows a correct final answer means no major error 94% of the time, a wrong one means
                          an error 98% of the time, and Qwen3.8 untrained gets this wrong on 7% of photos)
  classes.json            all codes in the order the classification layer scores them
  ft_dev.jsonl            header + {"trace_id", "messages"}   the 301 dev rows, evidence from fit rows only
  ft_test_<k>.jsonl       header + requests for all 3,498 test rows in k parts, evidence from all training rows
Test photos go only to our own pod. Usage: python scripts/build_q38_finetune.py --out data/v11 [--full] [--test-parts 6]
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from build_ranker_v3 import KEYS, QWEN, clip, qwen_answers, root_note
from build_vlm_requests import answer_schema, image_part
from k12.core import label_space, make_folds, read_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/bundle/x/competition_data'
REFS = False  # --image-refs: write photo references ('file:images/train/x.jpg') instead of embedding the photos


def photo(path):
    return {'type': 'image_url', 'image_url': {'url': 'file:' + str(path.relative_to(DATA))}} if REFS else image_part(path)


LAYOUT = ('Answer with one JSON object with exactly these keys, in this order: "transcription", "root_line", "reasoning", '
          '"first_code", "second_code", "third_code". Each code is a taxonomy code or NONE, and the three codes are different.')


def system_text(taxonomy):
    names = '\n'.join(f'{r.id} | {r.name}' for r in taxonomy.itertuples(index=False))
    return ("You are an experienced mathematics teacher diagnosing Indonesian K-12 students' handwritten working.\n\n"
            'For each photo, identify the ROOT misconception: the major error that the rest of the working follows from. It is not '
            'necessarily the first error on the page. Answer NONE when the working has no major error (fully correct, or only minor slips).\n\n'
            'Procedure:\n1. Transcribe the working line by line as "L1 | ...", "L2 | ...", keeping the student\'s own symbols and mistakes.\n'
            '2. Find the line where the root error enters ("none" if there is none) and say in one or two sentences what went wrong there.\n'
            '3. Give the code for that root error, then your second and third most likely codes.\n\n'
            f'Taxonomy (code | name). NONE means no major misconception.\n{names}\n\n{LAYOUT}\n')


class Evidence:
    def __init__(self, pool, tax):
        self.pool, self.tax = pool, tax
        self.by_q = {q: g for q, g in pool.groupby('question_id')}
        self.by_topic = {t: g for t, g in pool.groupby('topic_name')}

    def look(self, row, blind):
        """(codes teachers gave other students on this question, codes most used in the topic) without the row itself."""
        same = self.by_q.get(row.question_id)
        counts = Counter() if blind or same is None else Counter(same.loc[same.trace_id != row.trace_id, 'root_code'])
        topic = self.by_topic.get(row.topic_name)
        used = Counter() if topic is None else Counter(topic.loc[(topic.trace_id != row.trace_id) &
                                                                 ((topic.question_id != row.question_id) | (not blind)), 'root_code'])
        return counts, used


def user_text(row, counts, used, tax):
    name = lambda c: tax[c][0] if c in tax else 'no major error'
    hint = (', '.join(f'{c} ({name(c)}) x{n}' for c, n in counts.most_common(8)) if counts
            else 'no other student on this question has been marked')
    listed = [c for c, _ in counts.most_common() if c != 'NONE'] + [c for c, _ in used.most_common() if c != 'NONE']
    shown = []
    for c in listed:
        if c in tax and c not in shown:
            shown.append(c)
    defs = '\n'.join(f'{c} | {tax[c][0]} | {clip(tax[c][1], 260)}' for c in shown[:12])
    return (f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\nAnswer key: {row.correct_answer}\n'
            f'Codes the teachers gave other students on this question: {hint}\n'
            + (f'Definitions of the codes most used on this question and topic:\n{defs}' if defs else 'No codes are on record for this topic.'))


def messages(row, split, system, text):
    content = []
    if str(row.question_image_file).strip() not in ('', 'nan'):
        content += [{'type': 'text', 'text': 'Question diagram:'}, photo(DATA / 'images/questions' / row.question_image_file)]
    content += [{'type': 'text', 'text': "Student's handwritten working:"}, photo(DATA / 'images' / split / row.image_file),
                {'type': 'text', 'text': text}]
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def target(row, qwen, counts, used, classes):
    raw = str(row.misconceptions).strip()
    findings = json.loads(raw) if raw not in ('', 'nan') else []
    root = next((f for f in findings if f.get('root')), None)
    if row.root_code == 'NONE':
        minor = ' '.join(f"{f.get('step', '')}: {f.get('text', '')}".strip(': ') for f in findings if f.get('severity') == 'minor')
        line, why = 'none', 'No major error.' + (f' Minor slips only: {clip(minor, 400)}' if minor else '')
    else:
        line = str(root.get('step', '')).strip() if root else ''
        why = clip(root_note(row), 600) or 'Major error (no teacher note).'
    order = [row.root_code] + [qwen[k] for k in KEYS if qwen and qwen.get(k)] + [c for c, _ in counts.most_common()] + \
            ['NONE'] + [c for c, _ in used.most_common()] + classes
    codes = []
    for c in order:
        if c in classes and c not in codes:
            codes.append(c)
    return json.dumps({'transcription': str(row.transcription_steps).strip(), 'root_line': line or 'unknown', 'reasoning': why,
                       'first_code': codes[0], 'second_code': codes[1], 'third_code': codes[2]}, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/v11'); ap.add_argument('--full', action='store_true', help='train on all 1,500 rows (no dev check possible)')
    ap.add_argument('--blind', type=float, default=.1); ap.add_argument('--test-parts', type=int, default=6)
    ap.add_argument('--seed', type=int, default=2026); ap.add_argument('--skip-test', action='store_true')
    ap.add_argument('--image-refs', action='store_true', help='small files: photos are referenced by path inside the competition folder '
                    '(for Kaggle, where that folder is mounted; use with q38_finetune --image-root)')
    a = ap.parse_args()
    global REFS
    REFS = a.image_refs
    taxonomy = read_csv(ROOT / 'data/raw/taxonomy_v1.csv'); classes = label_space(taxonomy)
    tax = {r.id: (r.name, r.description) for r in taxonomy.itertuples(index=False)}
    train = read_csv(DATA / 'train.csv'); test = read_csv(DATA / 'test.csv'); folds = make_folds(train)
    fit = train if a.full else train.loc[folds != 0].reset_index(drop=True); dev = train.loc[folds == 0].reset_index(drop=True)
    qwen = qwen_answers(QWEN['train']); system = system_text(taxonomy); rng = random.Random(a.seed)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    ev_fit, ev_all = Evidence(fit, tax), Evidence(train, tax)
    header = json.dumps({'header': True, 'schema': answer_schema(classes), 'classes': classes})
    (out / 'classes.json').write_text(json.dumps(classes))  # one output of the classification layer per code, in this order
    report = {'system_chars': len(system), 'train_on': 'all 1,500 rows' if a.full else 'fit rows (dev fold held out)'}

    lines, chars, hinted = [], [], []
    for row in fit.itertuples(index=False):
        for blind in ([False, True] if rng.random() < a.blind else [False]):
            counts, used = ev_fit.look(row, blind)
            text = user_text(row, counts, used, tax); goal = target(row, qwen.get(row.trace_id), counts, used, classes)
            lines.append(json.dumps({'trace_id': row.trace_id + (':blind' if blind else ''), 'messages': messages(row, 'train', system, text),
                                     'target': goal, 'label': row.root_code,
                                     'final_ok': str(row.final_answer_correct).strip().lower() == 'true'}, ensure_ascii=False))
            chars.append(len(text) + len(goal)); hinted.append(row.root_code in counts)
    order = list(range(len(lines))); rng.shuffle(order)  # fixed mixed order, so a run cut short has seen a fair sample
    (out / 'ft_train.jsonl').write_text('\n'.join(lines[i] for i in order) + '\n')
    report['train'] = {'rows': len(lines), 'blind_rows': sum(':blind' in json.loads(l)['trace_id'] for l in lines),
                       'mean_user_plus_target_chars': sum(chars) / len(chars), 'truth_among_question_hints': sum(hinted) / len(hinted),
                       'rows_with_qwen_alternatives': sum(t in qwen for t in fit.trace_id), 'mb': round((out / 'ft_train.jsonl').stat().st_size / 1e6, 1)}

    if not a.full:
        rows = [json.dumps({'trace_id': r.trace_id, 'messages': messages(r, 'train', system, user_text(r, *ev_fit.look(r, False), tax))},
                           ensure_ascii=False) for r in dev.itertuples(index=False)]
        (out / 'ft_dev.jsonl').write_text('\n'.join([header] + rows) + '\n')
        report['dev'] = {'rows': len(rows), 'truth_among_question_hints': sum(r.root_code in ev_fit.look(r, False)[0] for r in dev.itertuples(index=False)) / len(dev)}
    if not a.skip_test:
        rows = [json.dumps({'trace_id': r.trace_id, 'messages': messages(r, 'test', system, user_text(r, *ev_all.look(r, False), tax))},
                           ensure_ascii=False) for r in test.itertuples(index=False)]
        for old in out.glob('ft_test_*.jsonl'):
            old.unlink()
        for k in range(a.test_parts):
            (out / f'ft_test_{k + 1}.jsonl').write_text('\n'.join([header] + rows[k::a.test_parts]) + '\n')
        report['test'] = {'rows': len(rows), 'parts': a.test_parts}
    (out / 'ft_build_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
