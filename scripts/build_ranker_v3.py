"""Ranker v3 data: yes/no pairs (k12.pointwise_ranker format) built on what Qwen3.8-27B wrote about each photo.

What is new against ranker v1 (data/v7):
  context     Qwen3.8's zero-shot reading of the photo: transcription, root line, its description of the error and its
              three codes. Qwen never saw a label, so training rows and test rows carry the same kind of text.
  hints       the codes teachers used for other students on the same question, with counts (a hint, not a mask).
  candidates  our system's top five (dev and test only), Qwen's three, NONE, the question's codes, and the codes of the
              most similar teacher notes from other questions.
  examples    each candidate carries one teacher note: for another student on this question when there is one,
              otherwise the most similar teacher note with that code from another question.
A training row never sees itself: hints and examples come from the other fit rows. A share of training rows is repeated
"question-blind" (no hints, examples from other questions only) because dev and test contain unseen questions.

Dev rows use fit rows only as evidence. Test rows use all 1,500 training rows and exist only where Qwen3.8 answered.

Usage: python scripts/build_ranker_v3.py --out data/v10 [--blind 0.1] [--train-rows N]
"""
import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import fuse_candidate as F
from fuse_candidate import fuse, geo, marginals
from k12.core import aligned_bundle, label_space, make_folds, read_csv

ROOT = F.ROOT
RANKER_DEV = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz'
QWEN = {'train': ['runs/deep/q38/pod_out/q38_dev_think.jsonl', 'runs/deep/q38/train_pod/tr_*.jsonl'],
        'test': ['runs/deep/q38/test_answers_merged.jsonl']}
KEYS = ('first_code', 'second_code', 'third_code')
CUT = 1100  # characters of Qwen's transcription kept (the end of the working is kept, where the answer is)


def qwen_answers(patterns):
    out = {}
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / pattern))):
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get('answer', {}).get('first_code') and r['trace_id'] not in out:
                        out[r['trace_id']] = r['answer']
    return out


def root_note(row):
    """The teacher's note on the root error of a training row ('' for NONE rows or when it is missing)."""
    raw = str(row.misconceptions).strip()
    findings = json.loads(raw) if raw not in ('', 'nan') else []
    root = next((f for f in findings if f.get('root')), None)
    if row.root_code == 'NONE' or not root:
        return ''
    return f"{str(root.get('span', '')).strip()}: {str(root.get('text', '')).strip()}".strip(': ')


def clip(text, limit):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


class Evidence:
    """Teacher-labelled rows that may be shown: counts per question and notes searchable by similarity."""

    def __init__(self, pool, vectorizer):
        self.pool = pool.reset_index(drop=True)
        self.notes = [root_note(r) for r in self.pool.itertuples(index=False)]
        self.has = np.array([bool(n) for n in self.notes])
        self.N = vectorizer.transform(self.notes)
        self.vec = vectorizer
        self.q = self.pool.question_id.to_numpy(); self.code = self.pool.root_code.to_numpy(); self.trace = self.pool.trace_id.to_numpy()
        self.topic = self.pool.topic_name.to_numpy()

    def look(self, row, query, blind):
        """(question counts, same-question example per code, other-question example per code, nearest codes)."""
        other = self.trace != row.trace_id
        same_q = other & (self.q == row.question_id) & (not blind)
        sims = np.asarray((self.vec.transform([query]) @ self.N.T).todense()).ravel()
        counts = Counter(self.code[same_q])
        same, far = {}, {}
        for i in np.argsort(-sims):
            if not self.has[i] or not other[i]:
                continue
            target = same if same_q[i] else far
            if self.q[i] == row.question_id and not same_q[i]:
                continue  # blind: this question's rows are hidden altogether
            target.setdefault(self.code[i], i)
        near = []
        for i in np.argsort(-sims):
            if self.has[i] and other[i] and self.q[i] != row.question_id and self.code[i] not in near:
                near.append(self.code[i])
            if len(near) >= 6:
                break
        return counts, same, far, near


def context_text(row, ans, counts, names):
    name = lambda c: names.get(c, 'no major error')
    if counts:
        hint = ', '.join(f'{c} ({name(c)}) x{n}' for c, n in counts.most_common(8))
    else:
        hint = 'no other student on this question has been marked'
    tx = ' '.join(str(ans.get('transcription', '')).split('\r'))
    tx = tx if len(tx) <= CUT else '…' + tx[-CUT:]
    codes = ', '.join(f'{ans[k]} ({name(ans[k])})' for k in KEYS if ans.get(k))
    return (f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {clip(row.question_text, 500)}\nAnswer key: {clip(row.correct_answer, 120)}\n'
            f'Codes the teachers gave other students on this question: {hint}\n'
            f"Student working as read from the photo by a first reader:\n{tx.strip()}\n"
            f"First reader's root line: {clip(ans.get('root_line', ''), 160)}\n"
            f"First reader's finding: {clip(ans.get('reasoning', ''), 700)}\n"
            f"First reader's codes, most likely first: {codes}")


def candidate_text(code, tax, ev, counts, same, far):
    if code == 'NONE':
        text = 'NONE: No major error - the working is correct or has only minor slips.'
        return text + (f' The teachers found no major error for {counts["NONE"]} other student(s) on this question.' if counts.get('NONE') else '')
    text = f'{code}: {tax[code][0]} - {clip(tax[code][1], 330)}'
    if code in same:
        return text + f' Teacher\'s note for another student on this question: "{clip(ev.notes[same[code]], 230)}"'
    if code in far:
        return text + f' Teacher\'s note on a different question ({clip(ev.topic[far[code]], 40)}): "{clip(ev.notes[far[code]], 230)}"'
    return text


def build_row(row, ans, ev, tax, names, classes, first, cap, blind=False, truth=None, topic_codes=None):
    query = f"{ans.get('root_line', '')} {ans.get('reasoning', '')}"
    counts, same, far, near = ev.look(row, query, blind)
    order = list(first) + [ans[k] for k in KEYS if ans.get(k)] + ['NONE'] + [c for c, _ in counts.most_common()] + near
    order += (topic_codes or {}).get(row.topic_name, [])
    codes = []
    for c in ([truth] if truth else []) + order:
        if c in classes and c not in codes:
            codes.append(c)
    codes = codes[:cap]
    return dict(trace_id=row.trace_id + (':blind' if blind else ''), context=context_text(row, ans, counts, names),
                candidates=[dict(code=c, text=candidate_text(c, tax, ev, counts, same, far)) for c in codes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/v10')
    ap.add_argument('--blind', type=float, default=.1, help='share of training rows repeated without any information about their question')
    ap.add_argument('--cap', type=int, default=12, help='most candidates per row')
    ap.add_argument('--top', type=int, default=5, help="our system's top codes offered as candidates (dev and test)")
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--test-parts', type=int, default=6)
    ap.add_argument('--train-rows', type=int, help='use only the first N fit rows that have a Qwen answer (smaller, cheaper run)')
    a = ap.parse_args()
    taxonomy = read_csv(ROOT / 'data/raw/taxonomy_v1.csv'); classes = label_space(taxonomy); C = np.asarray(classes)
    tax = {r.id: (r.name, r.description) for r in taxonomy.itertuples(index=False)}; names = {k: v[0] for k, v in tax.items()}
    train = read_csv(ROOT / 'data/raw/train.csv'); test = read_csv(ROOT / 'data/raw/test.csv'); folds = make_folds(train)
    fit = train.loc[folds != 0].reset_index(drop=True); dev = train.loc[folds == 0].reset_index(drop=True)
    q_train, q_test = qwen_answers(QWEN['train']), qwen_answers(QWEN['test'])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit([n for n in map(root_note, fit.itertuples(index=False)) if n])
    ev_fit, ev_all = Evidence(fit, vec), Evidence(train, vec)
    rng = random.Random(a.seed)
    topic_codes = {t: [c for c, _ in Counter(g.root_code).most_common(8)] for t, g in fit.groupby('topic_name')}
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    report = {'qwen_answers': {'fit': sum(t in q_train for t in fit.trace_id), 'dev': sum(t in q_train for t in dev.trace_id),
                               'test': sum(t in q_test for t in test.trace_id)}}

    made, used = [], 0
    for row in fit.itertuples(index=False):
        if row.trace_id not in q_train or (a.train_rows and used >= a.train_rows):
            continue
        used += 1
        for blind in ([False, True] if rng.random() < a.blind else [False]):
            r = build_row(row, q_train[row.trace_id], ev_fit, tax, names, classes, [], a.cap, blind, row.root_code, topic_codes)
            r.update(source='gold', weight=1.0, soft={row.root_code: 1.0})
            made.append(r)
    (out / 'pw_train.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in made) + '\n')
    pair = [len(r['context']) + len(c['text']) for r in made for c in r['candidates']] or [0]
    seen_q = [r['soft'] and any('for another student on this question' in c['text'] or 'other student(s) on this question' in c['text']
                                for c in r['candidates'] if c['code'] in r['soft']) for r in made]
    report['train'] = dict(rows=len(made), blind_rows=sum(r['trace_id'].endswith(':blind') for r in made),
                           pairs=sum(len(r['candidates']) for r in made), mean_pair_chars=float(np.mean(pair)), max_pair_chars=int(max(pair)),
                           truth_has_same_question_evidence=float(np.mean(seen_q or [0])),
                           qwen_first_is_truth=float(np.mean([q_train[r['trace_id'].split(':')[0]]['first_code'] in r['soft'] for r in made] or [0])))

    ids = dev.trace_id.tolist()
    P = fuse(geo(F.DEV, ids, classes), aligned_bundle(ROOT / RANKER_DEV, ids, classes), .5, .75, marginals(fit.root_code, classes), 30)
    made = []
    for i, row in enumerate(dev.itertuples(index=False)):
        if row.trace_id in q_train:
            r = build_row(row, q_train[row.trace_id], ev_fit, tax, names, classes, C[np.argsort(-P[i])[:a.top]].tolist(), a.cap)
            r['truth'] = row.root_code; made.append(r)
    (out / 'pw_dev.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in made) + '\n')
    slices = json.loads((ROOT / 'data/v7/dev_slices.json').read_text())
    inside = np.array([r['truth'] in {c['code'] for c in r['candidates']} for r in made])
    hard = np.array([slices[r['trace_id']] in ('new_pair', 'unseen_q') for r in made])
    seen_q = np.array([any('for another student on this question' in c['text'] or 'other student(s) on this question' in c['text']
                           for c in r['candidates'] if c['code'] == r['truth']) for r in made])
    report['dev'] = dict(rows=len(made), pairs=sum(len(r['candidates']) for r in made), truth_in_candidates=float(inside.mean()),
                         truth_in_candidates_hard=float(inside[hard].mean()), hard_rows=int(hard.sum()),
                         truth_has_same_question_evidence=float(seen_q.mean()),
                         mean_candidates=float(np.mean([len(r['candidates']) for r in made])))

    ids = test.trace_id.tolist(); PT = aligned_bundle(ROOT / 'artifacts/submission-3way-pw14/test.npz', ids, classes)
    made = [build_row(row, q_test[row.trace_id], ev_all, tax, names, classes, C[np.argsort(-PT[i])[:a.top]].tolist(), a.cap)
            for i, row in enumerate(test.itertuples(index=False)) if row.trace_id in q_test]
    (out / 'pw_test.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in made) + '\n')
    report['test'] = dict(rows=len(made), pairs=sum(len(r['candidates']) for r in made))
    # the same rows in parts, least-confident rows of the submitted system first, so a run cut short keeps the rows that matter most
    at = {t: i for i, t in enumerate(ids)}
    made.sort(key=lambda r: PT[at[r['trace_id']]].max())
    for old in out.glob('pw_test_*.jsonl'):
        old.unlink()
    size = -(-len(made) // a.test_parts)
    for k in range(a.test_parts):
        (out / f'pw_test_{k + 1}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in made[k * size:(k + 1) * size]) + '\n')
    (out / 'build_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
