"""Shared loaders for the shortcut-learning analysis (CPU only)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from k12.core import read_csv, label_space, normalize, topk, row_scores, make_folds, aligned_bundle
from common import blend, boot_ci  # runs/arch/label-structure/common.py

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
OUT = ROOT / 'runs/deep/shortcut'
MEMBERS = {'m32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 'm8': 'runs/pod/tx/text_tx_dev_e3.npz',
           'vl': 'runs/pod/dev-distill/validation_epoch_2.npz'}
TEST_MEMBERS = {'m32': 'runs/pod/text32-full/test.npz', 'm8': 'runs/pod/text-full/test.npz',
                'vl': 'runs/pod/full-distill/test.npz'}
W = {'m32': .35, 'm8': .30, 'vl': .35}


def load():
    tr = read_csv(ROOT / 'data/raw/train.csv'); tax = read_csv(ROOT / 'data/raw/taxonomy_v1.csv')
    classes = label_space(tax); folds = make_folds(tr)
    dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    return tr, tax, classes, dev, fit


def dev_preds(dev, classes):
    ids = dev.trace_id.tolist()
    P = {k: aligned_bundle(ROOT / p, ids, classes) for k, p in MEMBERS.items()}
    P['blend'] = blend(ids, classes)
    return P


def test_preds(test, classes):
    ids = test.trace_id.tolist()
    P = {k: aligned_bundle(ROOT / p, ids, classes) for k, p in TEST_MEMBERS.items()}
    lp = sum(W[k] * np.log(P[k]) for k in W)
    P['blend'] = normalize(np.exp(lp - lp.max(1, keepdims=True)))
    return P


def claude_dev(dev, classes):
    a = {}
    for line in open(ROOT / 'runs/claude/dev_opus.jsonl'):
        r = json.loads(line); x = r['answer']
        a[r['trace_id']] = [x.get('first_code'), x.get('second_code'), x.get('third_code')]
    ci = {c: i for i, c in enumerate(classes)}
    P = np.full((len(dev), len(classes)), 1e-4)
    codes = []
    for i, t in enumerate(dev.trace_id):
        cs = [c for c in a.get(t, []) if c in ci]
        codes.append(cs)
        for c, w in zip(cs, (.7, .2, .1)):
            P[i, ci[c]] += w
    return normalize(P), codes


def claude_unl():
    """Claude Opus labels on unlabelled photos, with question_id from unlabelled.csv."""
    unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
    rows = []
    for line in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
        r = json.loads(line); x = r.get('answer') or {}
        if r['trace_id'] in unl.index and x.get('first_code'):
            rows.append(dict(trace_id=r['trace_id'], question_id=unl.at[r['trace_id'], 'question_id'],
                             topic_name=unl.at[r['trace_id'], 'topic_name'], root_code=x['first_code'],
                             second=x.get('second_code'), third=x.get('third_code')))
    return pd.DataFrame(rows), unl


def slices(dev, fit):
    qc = fit.groupby('question_id').root_code.apply(set).to_dict()
    s = []
    for q, t in zip(dev.question_id, dev.root_code):
        if t == 'NONE':
            s.append('none')
        elif q not in qc:
            s.append('unseen_q')
        elif t in qc[q]:
            s.append('seen_pair')
        else:
            s.append('new_pair')
    return np.array(s), qc


def r(x, n=4):
    return None if x is None else round(float(x), n)
