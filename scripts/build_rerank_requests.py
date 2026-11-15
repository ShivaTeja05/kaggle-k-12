"""Second-opinion ("rerank") requests for an offline vision model that has already answered once.

Each request shows the model the photo again together with: its own first transcription and codes, the shortlist of
the submitted system (blend x ranker) with probabilities, and, optionally, teacher-labelled workings of other students
on the same question. The model verifies and ranks; it does not transcribe again.

Dev: candidates come from models trained on the fit rows only, and examples from the fit rows only, so no dev label
leaks. Test: candidates from the submitted test bundle, examples from all 1,500 training rows; for our own offline model
on our own pod only (never an API).

Usage:
  python scripts/build_rerank_requests.py --split dev --out data/v9/rerank/dev_ex.jsonl --examples 4
  python scripts/build_rerank_requests.py --split test --ids ids.json --parts 4 --out data/v9/rerank/test_ex.jsonl --examples 4
"""
import argparse
import json
from pathlib import Path

import numpy as np

from build_vlm_requests import image_part
from claude_label import system_prompt
import fuse_candidate as F
from fuse_candidate import fuse, geo, marginals
from k12.core import aligned_bundle, label_space, make_folds, read_csv

ROOT = F.ROOT
DATA = ROOT / 'data/bundle/x/competition_data'
RANKER_DEV = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz'
FIRST = {'dev': 'runs/deep/q38/pod_out/q38_dev_think.jsonl', 'test': 'runs/deep/q38/test_answers_merged.jsonl'}
KEYS = ('first_code', 'second_code', 'third_code')


def addendum(with_examples):
    items = ['(a) a transcription of the working made by a first reader; it can contain reading mistakes, so check it against the photo;',
             "(b) the first reader's three ranked codes;",
             "(c) a shortlist from a statistical classifier trained on the teachers' labels, with its confidence. It is reliable for "
             'errors the teachers have labelled before on this question and weak at errors that are new for the question;']
    if with_examples:
        items.append('(d) teacher-labelled workings of other students on this same question, when there are any. They show how the '
                     "teachers apply the codes here. This student's error may be different from all of them.")
    return ('\nSecond-opinion task. Besides the photo you are given:\n' + '\n'.join(items) + '\n'
            'Decide the root misconception yourself from the photo. Treat (b) and (c) as hypotheses to verify line by line against '
            'what this student actually wrote, not as answers: confirm the one whose definition matches the working, or choose '
            'another taxonomy code when none of them fits. Skip step 1 of the procedure: do not write a transcription.\n'
            'Answer with one JSON object with exactly these keys: "root_line", "reasoning" (two or three sentences), "first_code", '
            '"second_code", "third_code". Each code is a taxonomy code or NONE, and the three codes are different.\n')


def answer_schema(classes):
    code = {'type': 'string', 'enum': classes}
    fields = {'root_line': {'type': 'string'}, 'reasoning': {'type': 'string'}, 'first_code': code, 'second_code': code, 'third_code': code}
    return {'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


def examples(row, pool, names, k):
    """Up to k teacher-labelled workings on the same question: one per distinct code first (commonest first), then the rest."""
    same = pool[(pool.question_id == row.question_id) & (pool.trace_id != row.trace_id)]
    if same.empty or not k:
        return ''
    order = same.root_code.value_counts().index.tolist()
    first = [same[same.root_code == c].iloc[0] for c in order]
    rest = [r for _, r in same.iterrows() if r.trace_id not in {f.trace_id for f in first}]
    blocks = []
    for i, ex in enumerate((first + rest)[:k], 1):
        findings = json.loads(ex.misconceptions) if str(ex.misconceptions).strip() not in ('', 'nan') else []
        root = next((f for f in findings if f.get('root')), None)
        if ex.root_code == 'NONE':
            minor = ' '.join(f"{f['step']}: {f['text']}" for f in findings if f.get('severity') == 'minor')
            note = 'No major error.' + (f' Minor slips only: {minor}' if minor else '')
        else:
            note = f"{root['step']} | {root['span']}: {root['text']}" if root else ''
        blocks.append(f"Other student {i}\nTranscription:\n{ex.transcription_steps}\n"
                      f"Teacher's root code: {ex.root_code} ({names.get(ex.root_code, 'no major misconception')})\nTeacher's note: {note}")
    return ('(d) Teacher-labelled workings of other students on this same question '
            f'({len(same)} in total, {same.root_code.nunique()} different codes):\n\n' + '\n\n'.join(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['dev', 'test'], required=True)
    ap.add_argument('--ids', help='JSON list of trace_ids to include, in order (default: every row with a first answer)')
    ap.add_argument('--examples', type=int, default=0, help='teacher-labelled same-question workings to show (0 = none)')
    ap.add_argument('--top', type=int, default=8, help='length of the classifier shortlist')
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    taxonomy = read_csv(DATA / 'taxonomy_v1.csv'); classes = label_space(taxonomy); C = np.asarray(classes)
    names = dict(zip(taxonomy.id, taxonomy.name))
    train = read_csv(DATA / 'train.csv'); folds = make_folds(train)
    if a.split == 'dev':
        rows = train.loc[folds == 0].reset_index(drop=True); pool = train.loc[folds != 0].reset_index(drop=True)
        ids = rows.trace_id.tolist()
        P = fuse(geo(F.DEV, ids, classes), aligned_bundle(ROOT / RANKER_DEV, ids, classes), .5, .75, marginals(pool.root_code, classes), 30)
        images = 'train'
    else:
        rows = read_csv(DATA / 'test.csv'); pool = train; ids = rows.trace_id.tolist()
        P = aligned_bundle(ROOT / 'artifacts/submission-3way-pw14/test.npz', ids, classes)
        images = 'test'
    first = {}
    for line in (ROOT / FIRST[a.split]).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r['answer'].get('first_code'):
                first[r['trace_id']] = r['answer']
    keep = json.loads(Path(a.ids).read_text()) if a.ids else [t for t in ids if t in first]
    at = {t: i for i, t in enumerate(ids)}
    system = system_prompt(taxonomy) + addendum(bool(a.examples))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    stats = {'rows': len(keep), 'with_examples': 0}
    lines = []
    for trace in keep:
        row = rows.iloc[at[trace]]; ans = first[trace]; p = P[at[trace]]
        short = np.argsort(-p)[:a.top]
        text = (f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\nAnswer key: {row.correct_answer}\n\n'
                f"(a) First reader's transcription:\n{ans.get('transcription', '').strip()}\n\n"
                f"(b) First reader's codes, most likely first: " + ', '.join(f"{ans[k]} ({names.get(ans[k], 'no major misconception')})" for k in KEYS if ans.get(k)) + '\n\n'
                '(c) Classifier shortlist, most likely first:\n' + '\n'.join(f"{n}. {C[j]} ({names.get(C[j], 'no major misconception')}): {100 * p[j]:.0f}%" for n, j in enumerate(short, 1)))
        shown = examples(row, pool, names, a.examples)
        if shown:
            text += '\n\n' + shown; stats['with_examples'] += 1
        elif a.examples:
            text += '\n\n(d) There are no teacher-labelled workings of other students for this question.'
        content = []
        if str(row.question_image_file).strip() not in ('', 'nan'):
            content += [{'type': 'text', 'text': 'Question diagram:'}, image_part(DATA / 'images/questions' / row.question_image_file)]
        content += [{'type': 'text', 'text': "Student's handwritten working:"}, image_part(DATA / 'images' / images / row.image_file),
                    {'type': 'text', 'text': text}]
        lines.append(json.dumps({'trace_id': trace, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]}))
    header = json.dumps({'header': True, 'schema': answer_schema(classes), 'classes': classes, 'rows': len(lines)})
    for part in range(a.parts):
        target = out if a.parts == 1 else out.with_name(f'{out.stem}_{part + 1}{out.suffix}')
        target.write_text('\n'.join([header] + lines[part::a.parts]) + '\n')
        print({'out': str(target), 'rows': len(lines[part::a.parts]), 'mb': round(target.stat().st_size / 1e6, 1)})
    print(stats)


if __name__ == '__main__':
    main()
