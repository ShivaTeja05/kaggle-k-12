"""CPU check for the reason-then-classify design.

Question: if a student could write a Claude-like rationale, how much of the code decision is
recoverable from that rationale text alone, by a cheap model trained only on the 26k Claude
labels of unlabelled photos (no dev, no test)? And how much does a no-rationale input
(question + transcript) give with the same cheap model? Evaluated on the 301 dev rows, with
gold truth; the rationale used on dev is Claude Opus zero-shot reasoning (runs/claude/dev_opus.jsonl).
This bounds the 'rationale -> code' stage; it does NOT measure a student's ability to write it.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from k12.core import read_csv, label_space, make_folds, aligned_bundle, row_scores, topk, normalize, map3
from evaluate_claude import claude_scores

OUT = ROOT / 'runs/arch/design-distill'
tax = read_csv(ROOT / 'data/raw/taxonomy_v1.csv')
classes = label_space(tax)
ci = {c: i for i, c in enumerate(classes)}
train = read_csv(ROOT / 'data/raw/train.csv')
fold = make_folds(train)
dev = train[fold == 0].reset_index(drop=True)
fit = train[fold != 0].reset_index(drop=True)
ids = dev.trace_id.tolist()
y = dev.root_code.tolist()
unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')

CODE_RE = re.compile(r'\b(?:[A-Z][A-Za-z]{1,3}-\d{2}|NONE)\b')


def clean(t):
    return CODE_RE.sub(' ', t or '')


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        a = r.get('answer') or {}
        if a.get('first_code'):
            out[r['trace_id']] = a
    return out

U = load(ROOT / 'runs/claude/unl_all_v4.jsonl')
D = load(ROOT / 'runs/claude/dev_opus.jsonl')
uid = [t for t in U if t in unl.index and U[t]['first_code'] in ci]
print('unlabelled Claude rows usable', len(uid))


def soft(a):
    v = np.zeros(len(classes))
    for s, k in zip((.7, .2, .1), ('first_code', 'second_code', 'third_code')):
        c = a.get(k)
        if c in ci and v[ci[c]] == 0:
            v[ci[c]] = s
    return v / v.sum()


def last_sentence_dropped(t):
    s = re.split(r'(?<=[.!?])\s+', (t or '').strip())
    return ' '.join(s[:-1]) if len(s) > 1 else ''

fields = {
    'reasoning': lambda a, q: clean(a.get('reasoning')),
    'reasoning_no_conclusion': lambda a, q: clean(last_sentence_dropped(a.get('reasoning'))),
    'rootline_reasoning': lambda a, q: clean((a.get('root_line') or '') + ' || ' + (a.get('reasoning') or '')),
    'question_transcript': lambda a, q: clean(f"{q['topic_name']} || {q['question_text']} || {q['correct_answer']} || {a.get('transcription')}"),
}

# fit-row code sets per question -> coverage bins on dev
fit_codes = fit.groupby('question_id').root_code.apply(set).to_dict()
def bin_of(r):
    if r.root_code == 'NONE':
        return 'NONE'
    if r.question_id not in fit_codes:
        return 'q_unseen'
    return 'code_seen_for_q' if r.root_code in fit_codes[r.question_id] else 'code_unseen_for_q'
bins = np.array([bin_of(r) for r in dev.itertuples()])

# training targets: hard argmax for LR (sklearn multinomial) with sample replication of top-3 shares
Y = np.array([ci[U[t]['first_code']] for t in uid])
uq = unl.loc[uid]
dq = dev.set_index('trace_id')

ref_w = (.35, .30, .35)
bundles = [ROOT / 'runs/pod/text32/validation_tx_dev_epoch_1.npz', ROOT / 'runs/pod/tx/text_tx_dev_e3.npz',
           ROOT / 'runs/pod/dev-distill/validation_epoch_2.npz']
P = [aligned_bundle(b, ids, classes) for b in bundles]
blend = normalize(np.exp(sum(w * np.log(p + 1e-9) for w, p in zip(ref_w, P))))
s_blend = row_scores(y, topk(blend, classes))
pc, _ = claude_scores(ROOT / 'runs/claude/dev_opus.jsonl', ids, classes)
s_claude = row_scores(y, topk(pc, classes))
print('blend', s_blend.mean(), 'claude', s_claude.mean())

qid = dev.question_id.values
uq_ids = np.unique(qid)
rng = np.random.default_rng(0)
boots = [np.concatenate([np.flatnonzero(qid == q) for q in rng.choice(uq_ids, len(uq_ids))]) for _ in range(4000)]
def ci_delta(sa, sb):
    d = np.array([sa[b].mean() - sb[b].mean() for b in boots])
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]

# question-grouped 5-fold within dev, for cross-fitting blend weights
qf = {q: i % 5 for i, q in enumerate(rng.permutation(uq_ids))}
dfold = np.array([qf[q] for q in qid])

def per_bin(s):
    return {b: [int((bins == b).sum()), round(float(s[bins == b].mean()), 4)] for b in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen', 'NONE']}

res = {'blend': {'map3': round(float(s_blend.mean()), 4), 'bins': per_bin(s_blend)},
       'claude': {'map3': round(float(s_claude.mean()), 4), 'bins': per_bin(s_claude)}}
for name, f in fields.items():
    Xtr_txt = [f(U[t], uq.loc[t]) for t in uid]
    Xdv_txt = [f(D.get(t, {}), dq.loc[t]) for t in ids]
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
    Xtr = hstack([vw.fit_transform(Xtr_txt), vc.fit_transform(Xtr_txt)]).tocsr()
    Xdv = hstack([vw.transform(Xdv_txt), vc.transform(Xdv_txt)]).tocsr()
    lr = LogisticRegression(C=8.0, max_iter=300)
    lr.fit(Xtr, Y)
    pr = np.full((len(ids), len(classes)), 1e-6)
    pr[:, lr.classes_] += lr.predict_proba(Xdv)
    pr = normalize(pr)
    s = row_scores(y, topk(pr, classes))
    # agreement with Claude's own first code (how faithfully the rationale encodes the teacher's decision)
    agree = float(np.mean([classes[i] == D.get(t, {}).get('first_code') for i, t in zip(pr.argmax(1), ids)]))
    # blend + this stage, weight cross-fitted on question folds
    grid = np.linspace(0, 1, 21)
    cf = np.zeros(len(ids))
    picks = []
    for k in range(5):
        tr = dfold != k
        best = max(grid, key=lambda w: row_scores([y[i] for i in np.flatnonzero(tr)], topk(normalize(np.exp((1 - w) * np.log(blend[tr] + 1e-9) + w * np.log(pr[tr] + 1e-9))), classes)).mean())
        picks.append(float(best))
        te = ~tr
        cf[te] = row_scores([y[i] for i in np.flatnonzero(te)], topk(normalize(np.exp((1 - best) * np.log(blend[te] + 1e-9) + best * np.log(pr[te] + 1e-9))), classes))
    res[name] = {'map3': round(float(s.mean()), 4), 'top1_agree_with_claude_first': round(agree, 4),
                 'delta_vs_blend': round(float(s.mean() - s_blend.mean()), 4), 'ci_vs_blend': ci_delta(s, s_blend),
                 'bins': per_bin(s),
                 'blend_plus_stage_crossfit': round(float(cf.mean()), 4), 'blend_plus_ci': ci_delta(cf, s_blend),
                 'weights_picked': picks, 'blend_plus_bins': per_bin(cf)}
    print(name, json.dumps(res[name]))

json.dump(res, open(OUT / 'd1_rationale_to_code.json', 'w'), indent=1)
