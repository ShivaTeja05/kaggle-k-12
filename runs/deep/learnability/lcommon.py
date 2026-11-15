"""Shared loaders for the learnability lens (CPU only, read-only on project data)."""
import json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from common import ROOT, load, blend, boot_ci
from k12.core import read_csv, label_space, normalize, topk, row_scores, map3, make_folds, aligned_bundle

OUT = ROOT / 'runs/deep/learnability'
GENERIC = {'CAL', 'WOP', 'ANS', 'OTH', 'NOT', 'FRM', 'VIS', 'CHK', 'UNT', 'REL'}


def fam(c):
    return 'NONE' if c == 'NONE' else c.split('-')[0].upper()


def sinkhorn(P, target, lam, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(target / Q.mean(0))
    return normalize(P * np.exp(lam * b))


def claude_jsonl(path):
    out = {}
    for line in open(path):
        r = json.loads(line); a = r.get('answer') or {}
        codes = [a.get('first_code'), a.get('second_code'), a.get('third_code')]
        if codes[0]:
            out[r['trace_id']] = [c for c in codes if c]
    return out


def get_all():
    tr, tax, classes, folds, dev, fit = load()
    ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
    P = blend(ids, classes)
    ci = {c: i for i, c in enumerate(classes)}
    pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
    PS = sinkhorn(P, pi_fit, .5)
    comps = {'m32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 'm8': 'runs/pod/tx/text_tx_dev_e3.npz',
             'mvl': 'runs/pod/dev-distill/validation_epoch_2.npz',
             'scale_0': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
             'scale_25': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
             'scale_100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'}
    preds = {'blend': topk(P, classes), 'blend_sk': topk(PS, classes)}
    for k, p in comps.items():
        preds[k] = topk(aligned_bundle(ROOT / p, ids, classes), classes)
    cl = claude_jsonl(ROOT / 'runs/claude/dev_opus.jsonl')
    preds['claude'] = np.array([(cl[t] + ['NONE', 'NONE', 'NONE'])[:3] for t in ids])
    for name, f in [('claude_fewshot', 'dev_opus_fewshot.jsonl'), ('claude_fable', 'dev_fable52.jsonl')]:
        try:
            d = claude_jsonl(ROOT / 'runs/claude' / f)
            if all(t in d for t in ids):
                preds[name] = np.array([(d[t] + ['NONE'] * 3)[:3] for t in ids])
        except Exception:
            pass
    unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv')
    v4 = claude_jsonl(ROOT / 'runs/claude/unl_all_v4.jsonl')
    unl = unl[unl.trace_id.isin(v4)].copy()
    unl['c1'] = unl.trace_id.map(lambda t: v4[t][0]); unl['c3'] = unl.trace_id.map(lambda t: v4[t])
    test = read_csv(ROOT / 'data/raw/test.csv')
    return dict(tr=tr, tax=tax, classes=classes, folds=folds, dev=dev, fit=fit, ids=ids, y=y, P=P, PS=PS,
                preds=preds, unl=unl, test=test, pi_fit=pi_fit)


def score(y, pred):
    return row_scores(y, pred)
