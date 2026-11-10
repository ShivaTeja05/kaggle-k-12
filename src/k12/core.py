import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_COLUMNS = ['question_id', 'grade', 'topic_name', 'question_text',
                 'question_image_file', 'correct_answer', 'image_file']
TRAIN_ONLY = {'root_code', 'transcription_steps', 'final_answer_correct', 'clean', 'misconceptions'}


def read_csv(path):
    # Literal NONE is a class. Never allow pandas to interpret labels as nulls.
    return pd.read_csv(path, keep_default_na=False, dtype={'trace_id': str, 'question_id': str})


def label_space(taxonomy):
    return sorted(set(taxonomy['id'].astype(str)) | {'NONE'})


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected a finite nonnegative probability matrix')
    if (p.sum(axis=1) <= 0).any():
        raise ValueError('Empty probability row')
    return p / p.sum(axis=1, keepdims=True)


def topk(p, classes, k=3):
    p = normalize(p)
    if p.shape[1] != len(classes) or len(set(classes)) != len(classes):
        raise ValueError('Class order does not match scores')
    return np.asarray(classes)[np.argsort(-p, axis=1, kind='stable')[:, :k]]


def row_scores(y, predictions):
    if len(y) != len(predictions):
        raise ValueError('Target/prediction length mismatch')
    scores = []
    for target, codes in zip(y, predictions):
        if isinstance(codes, str):
            codes = codes.split()
        hits = [1.0 / (i + 1) for i, c in enumerate(list(codes)[:3]) if c == target]
        scores.append(hits[0] if hits else 0.0)
    return np.asarray(scores)


def map3(y, p, classes):
    return float(row_scores(y, topk(p, classes)).mean())


def write_submission(test, p, classes, path):
    if len(test) != len(p) or not test.trace_id.is_unique:
        raise ValueError('Submission length or IDs invalid')
    out = pd.DataFrame({'trace_id': test.trace_id.tolist(),
                        'root_code': [' '.join(x) for x in topk(p, classes)]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def make_folds(train, n_splits=5, seed=2026):
    """Round-robin stratification with normalized-context duplicate grouping.

    Singletons are evaluated once, like any other row; their class will be absent
    from that fold's fitting set. This deliberately exposes the rare-class problem.
    Student IDs are unavailable, so this is not a student-group split.
    """
    rng = np.random.default_rng(seed)
    fold = np.full(len(train), -1, dtype=int)
    groups = {}
    for i, r in enumerate(train.itertuples(index=False)):
        text = ' '.join(str(r.transcription_steps).lower().split())
        key = str(r.question_id) + '|' + text
        groups.setdefault(key, []).append(i)
    sizes = np.zeros(n_splits, dtype=int)
    counts = {c: np.zeros(n_splits, dtype=int) for c in set(train.root_code)}
    items = list(groups.values())
    rng.shuffle(items)
    items.sort(key=lambda g: -len(g))
    for ix in items:
        labels = train.iloc[ix].root_code.tolist()
        costs = sum((counts[c] for c in labels), np.zeros(n_splits))
        candidates = np.flatnonzero(costs == costs.min())
        small = sizes[candidates]
        candidates = candidates[small == small.min()]
        f = int(rng.choice(candidates))
        fold[ix] = f
        sizes[f] += len(ix)
        for c in labels:
            counts[c][f] += 1
    return fold


def save_bundle(path, ids, classes, p, **meta):
    p = normalize(p)
    if len(set(ids)) != len(ids) or len(ids) != len(p):
        raise ValueError('Prediction IDs must be unique and match rows')
    np.savez_compressed(path, ids=np.asarray(ids, dtype=str),
                        classes=np.asarray(classes, dtype=str), p=p.astype('float32'),
                        meta=json.dumps(meta))


def aligned_bundle(path, ids, classes):
    with np.load(path, allow_pickle=False) as b:
        old_ids, old_classes = b['ids'].tolist(), b['classes'].tolist()
        if len(set(old_ids)) != len(old_ids) or set(old_ids) != set(ids):
            raise ValueError('Prediction ID sets differ')
        if len(set(old_classes)) != len(old_classes) or set(old_classes) != set(classes):
            raise ValueError('Prediction class sets differ')
        ii, cc = {x: i for i, x in enumerate(old_ids)}, {x: i for i, x in enumerate(old_classes)}
        return normalize(b['p'][[ii[x] for x in ids]][:, [cc[x] for x in classes]])


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
