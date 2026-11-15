"""Worked examples keyed by error code, and the cards that show them to the reasoner (ARCHITECTURE_REFORM_2026.md 4.4).

The index is built from the rows it is given and from nothing else: fit rows for development work, all training
rows for the final test run. A row is never an example for itself, in any mode. Nothing here reads the label of
the row being diagnosed, so a card cannot say which code is the true one.

root_line, line_text and why are copies of the helpers in scripts/build_candidate_dataset.py (root_line extended
to ranges such as L1-L3); the originals stay where they are.
"""
import json
import re

import numpy as np

LINE = re.compile(r'^\s*L(\d+)\s*\|\s?(.*)$')
STEP = re.compile(r'^\s*L(\d+)(?:\s*[-–]\s*L?(\d+))?')
CONCLUSION = re.compile(r'\b(match|matches|root|major error|no error|NONE|minor slips?)\b', re.I)
TOKEN = re.compile(r'\w+')
NONE_NAME, NONE_DESCRIPTION = 'No major error', 'The working is correct or has only minor slips that do not change the result.'


def root_line(value):
    """(first, last) line numbers of a teacher step such as "L2" or "L1-L3"; None when it has no line label."""
    m = STEP.match(str(value or ''))
    if not m:
        return None
    first = int(m.group(1)); last = int(m.group(2)) if m.group(2) else first
    return (first, last) if last >= first else (first, first)


def line_text(transcription, line):
    """Text of the line labelled `line` ("L3") in a teacher transcript, or '' when there is no such line."""
    for raw in str(transcription).splitlines():
        m = LINE.match(raw)
        if m and f'L{m.group(1)}' == line:
            return m.group(2).strip()
    return ''


def why(text, words=60):
    """First sentences up to `words`, keeping the last concluding sentence (it names the error) when it fits."""
    sentences = [x for x in re.split(r'(?<=[.!?])\s+', ' '.join(str(text).split())) if x]
    n = lambda ix: len(' '.join(sentences[i] for i in ix).split())
    if not sentences or n(range(len(sentences))) <= words:
        return ' '.join(sentences)
    concl = next((i for i in range(len(sentences) - 1, -1, -1) if CONCLUSION.search(sentences[i])), len(sentences) - 1)
    keep = [concl] if n([concl]) <= words // 2 else []
    for i in range(len(sentences)):
        if i not in keep:
            if n(keep + [i]) > words:
                break
            keep.append(i)
    return ' '.join(sentences[i] for i in sorted(keep)) if keep else ' '.join(sentences[0].split()[:words])


def flat(text):
    return ' '.join(str(text).split())


def question_key(text):
    """Two question ids with the same wording are one question: an example from either would repeat the story."""
    return flat(text).lower()


class Tfidf:
    """Smoothed TF-IDF with unit-length rows (raw counts, idf = ln((1+N)/(1+df)) + 1); unknown words are ignored."""

    def __init__(self, documents):
        tokens = [TOKEN.findall(str(d).lower()) for d in documents]
        self.vocab = {w: i for i, w in enumerate(sorted({w for t in tokens for w in t}))}
        df = np.zeros(len(self.vocab))
        for t in tokens:
            for w in set(t):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + len(tokens)) / (1 + df)) + 1

    def vectors(self, documents):
        X = np.zeros((len(documents), len(self.vocab)))
        for row, d in enumerate(documents):
            for w in TOKEN.findall(str(d).lower()):
                if w in self.vocab:
                    X[row, self.vocab[w]] += 1
        X *= self.idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm > 0, norm, 1.)


def cosine(X, v):
    """Cosine of each row of X with v; 0 for an empty vector."""
    norm = np.linalg.norm(X, axis=1) * np.linalg.norm(v)
    return (X @ v) / np.where(norm > 0, norm, 1.)


def same_question(entry, question_id, question_text):
    """Whether an example sits on the row's own question (same id, or the same wording under another id)."""
    return entry['question_id'] == str(question_id) or question_key(entry['question_text']) == question_key(question_text)


class ExemplarIndex:
    """Error rows grouped by root code, and the correct (NONE) rows, of the frame given (train.csv columns)."""

    def __init__(self, frame):
        self.entries, self.nones, self.skipped = [], [], []
        for r in frame.itertuples(index=False):
            base = {'trace_id': str(r.trace_id), 'question_id': str(r.question_id), 'question_text': str(r.question_text),
                    'topic': str(r.topic_name), 'code': str(r.root_code), 'working': str(r.transcription_steps)}
            findings = json.loads(r.misconceptions) if str(r.misconceptions).strip() else []
            if r.root_code == 'NONE':
                self.nones.append(dict(base, clean=not findings))
                continue
            root = next((f for f in findings if f.get('root')), None)
            if root is None or not str(root.get('text', '')).strip():  # no annotated root: nothing to show
                self.skipped.append(base['trace_id'])
                continue
            self.entries.append(dict(base, step=str(root.get('step', '')).strip(), span=str(root.get('span', '')).strip(),
                                     explanation=flat(root['text'])))
        self.by_code = {}
        for j, e in enumerate(self.entries):
            self.by_code.setdefault(e['code'], []).append(j)
        self._phi = Tfidf([e['explanation'] for e in self.entries])
        self._A = self._phi.vectors([e['explanation'] for e in self.entries])
        questions = sorted({question_key(e['question_text']) for e in self.entries + self.nones})
        self._psi = Tfidf(questions)  # one document per question, so a question with many rows does not weigh more
        self._Q = self._psi.vectors([e['question_text'] for e in self.entries])
        self._query = {}

    def trace_ids(self):
        """Every row that can be shown as an example (for the check that no dev or test row is among them)."""
        return {e['trace_id'] for e in self.entries} | {e['trace_id'] for e in self.nones}

    def _question(self, text):
        key = question_key(text)
        if key not in self._query:
            self._query[key] = self._psi.vectors([text])[0]
        return self._query[key]

    def select(self, code, row_trace_id, question_id, question_text, mode='cross', mu=0.5):
        """The example of `code` to show a row, or None when there is none to show.

        'cross' (4.4.2): among examples on OTHER questions, the argmax of
        cos(explanation, mean explanation of the code) - mu * cos(example question, row question).
        'same' (4.4.3): the most typical example on the row's own question when one exists, otherwise as 'cross'.
        The row itself is left out of the pool and of the mean. Ties: the shorter working, then the trace_id.
        """
        if mode not in ('cross', 'same'):
            raise ValueError("mode is 'cross' or 'same'")
        pool = [j for j in self.by_code.get(code, []) if self.entries[j]['trace_id'] != str(row_trace_id)]
        if not pool:
            return None
        own = np.array([same_question(self.entries[j], question_id, question_text) for j in pool])
        typical = cosine(self._A[pool], self._A[pool].mean(0))
        if mode == 'same' and own.any():
            score, keep = typical, own
        elif (~own).any():
            score = typical - mu * cosine(self._Q[pool], self._question(question_text)) if mu else typical
            keep = ~own
        else:
            return None
        order = sorted((-round(float(score[n]), 9), len(self.entries[j]['working']), self.entries[j]['trace_id'], j)
                       for n, j in enumerate(pool) if keep[n])
        return self.entries[order[0][3]]

    def none_example(self, row_trace_id, question_id, topic, question_text=None):
        """One correct working from another question: same topic when there is one, an unannotated working when
        there is one, and then the working of middle length (a typical one), ties to the trace_id."""
        key = question_key(question_text) if question_text is not None else None
        pool = [e for e in self.nones if e['trace_id'] != str(row_trace_id) and e['question_id'] != str(question_id)
                and question_key(e['question_text']) != key]
        for test in (lambda e: e['topic'] == str(topic), lambda e: e['clean']):
            if any(test(e) for e in pool):
                pool = [e for e in pool if test(e)]
        pool.sort(key=lambda e: (len(e['working']), e['trace_id']))
        return pool[(len(pool) - 1) // 2] if pool else None


def one_line(working):
    return '   '.join(x.strip() for x in str(working).splitlines() if x.strip())


def root_text(entry):
    """"L2 | <span>", or the whole line(s) when the teacher marked no span; the label alone if the line is missing."""
    step, span = entry['step'], flat(entry['span'])
    if not span:
        lines = root_line(step)
        if lines:
            span = ' ; '.join(t for t in (line_text(entry['working'], f'L{n}') for n in range(lines[0], lines[1] + 1)) if t)
    return f'{step} | {span}' if span else (step or '-')


def render_card(code, name, description, entry, same_question=False, why_words=None):
    """Card text for one shortlisted code (4.4.4): the description alone when entry is None.

    The text describes the code and another student's working. It is the same whatever the true code of the row
    being diagnosed is; the caller must not add anything that depends on that label.
    """
    head = f'{code} | {flat(name)} | {flat(description)}'
    if entry is None:
        return head
    where = 'same question' if same_question else 'different question'
    lines = [head, f'Example from another student ({where}):', f"  Question: {flat(entry['question_text'])}",
             f"  Working:  {one_line(entry['working'])}"]
    if code == 'NONE':
        return '\n'.join(lines + ['  Root line: none', '  Why this is NONE: no major error'])
    reason = why(entry['explanation'], why_words) if why_words else flat(entry['explanation'])
    return '\n'.join(lines + [f'  Root line: {root_text(entry)}', f'  Why this is {code}: {reason}'])
