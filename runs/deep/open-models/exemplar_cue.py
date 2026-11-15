"""Does the v5 candidate prompt leak a 'this code has a same-question example' cue that differs between train and the
unseen-pair dev rows? Measures P(truth candidate carries 'On this question, e.g.') vs P(a distractor carries it)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'scripts', 'runs/arch/label-structure']
from common import load
tr, tax, classes, folds, dev, fit = load()
qc = fit.groupby('question_id').root_code.apply(set).to_dict()
dq = dict(zip(dev.trace_id, dev.question_id))
def cand_lines(prompt):
    blk = prompt.split('Candidate error codes:\n', 1)[1].split('\nFind the root', 1)[0]
    out = {}
    for ln in blk.split('\n'):
        m = re.match(r'^([A-J])\. ([A-Z]+(?:-\d+)?):', ln)
        if m: out[m.group(2)] = 'On this question, e.g.' in ln
    return out
res = {}
t_has, d_has, n_ex = [], [], []
for l in open('data/v5/cand_train.jsonl'):
    r = json.loads(l); ans = r['target'].rsplit('ANSWER: ', 1)[1].strip()[:1]
    truth = r['codes'][r['letters'].index(ans)]; c = cand_lines(r['prompt'])
    if truth == 'NONE': continue
    t_has.append(c.get(truth, False)); d = [v for k, v in c.items() if k not in (truth, 'NONE')]; d_has.append(np.mean(d) if d else 0)
    n_ex.append(sum(c.values()))
res['train_error_rows'] = {'n': len(t_has), 'P_truth_has_same_q_example': round(float(np.mean(t_has)), 3),
    'P_distractor_has_example': round(float(np.mean(d_has)), 3), 'mean_candidates_with_example': round(float(np.mean(n_ex)), 2)}
seg = {}
for l in open('data/v5/cand_dev.jsonl'):
    r = json.loads(l); truth = r.get('truth'); q = dq[r['trace_id']]
    if truth is None or truth == 'NONE': continue
    c = cand_lines(r['prompts'][0])
    b = 'q_unseen' if q not in qc else ('pair_seen' if truth in qc[q] else 'pair_unseen')
    s = seg.setdefault(b, {'in_cands': [], 'truth_has_ex': [], 'distractor_has_ex': []})
    s['in_cands'].append(truth in c)
    if truth in c:
        s['truth_has_ex'].append(c[truth]); s['distractor_has_ex'].append(np.mean([v for k, v in c.items() if k not in (truth, 'NONE')]))
for b, s in seg.items():
    res['dev_' + b] = {'n': len(s['in_cands']), 'recall10': round(float(np.mean(s['in_cands'])), 3),
        'P_truth_has_same_q_example': round(float(np.mean(s['truth_has_ex'])), 3),
        'P_distractor_has_example': round(float(np.mean(s['distractor_has_ex'])), 3)}
Path('runs/deep/open-models/exemplar_cue.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
