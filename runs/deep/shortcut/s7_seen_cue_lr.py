"""(7) Strength of the 'candidate was seen for this question' cue in the candidate-student data vs dev.
P(seen | truth) / P(seen | distractor) and P(unseen | truth) / P(unseen | distractor), seen = code appears among fit gold +
Claude first codes for the question (own row excluded). A large train LR for 'unseen' (<<1) means the student learns to
veto codes never associated with the question - exactly the failure on new-pair dev rows."""
import json
from collections import Counter
import numpy as np
from sc_common import load, claude_unl, slices, OUT, ROOT, r
tr, tax, classes, dev, fit = load()
cl, unl = claude_unl(); S, qc = slices(dev, fit)
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
cl_ok = cl[~cl.trace_id.isin(set(dev.trace_id))]
qcount = {}
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount.setdefault(q, Counter())[c] += 1
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))
L = 'ABCDEFGHIJ'
def lr(items):
    t = np.mean([a for a, _ in items]); d = np.mean([x for _, b in items for x in b])
    return dict(p_seen_truth=r(t), p_seen_distractor=r(d), LR_seen=r(t / d), LR_unseen=r((1 - t) / (1 - d)))
res = {}
for src in ['claude', 'gold']:
    items = []
    for line in open(ROOT / 'data/v5/cand_train.jsonl'):
        x = json.loads(line)
        if x['source'] != src: continue
        truth = x['codes'][L.index(x['target'][-1])]
        if truth == 'NONE': continue
        cnt = Counter(qcount.get(qmap[x['trace_id']], Counter())); cnt[own.get(x['trace_id'])] -= 1
        items.append((cnt.get(truth, 0) > 0, [cnt.get(c, 0) > 0 for c in x['codes'] if c not in (truth, 'NONE')]))
    res['train_' + src + '_error_rows'] = lr(items)
dv = {}
for line, s, q in zip(open(ROOT / 'data/v5/cand_dev.jsonl'), S, dev.question_id):
    x = json.loads(line); truth = x['truth']; codes = x['codes_per_prompt'][0]
    if truth == 'NONE' or truth not in codes: continue
    cnt = qcount.get(q, Counter())
    it = (cnt.get(truth, 0) > 0, [cnt.get(c, 0) > 0 for c in codes if c not in (truth, 'NONE')])
    dv.setdefault('all_error', []).append(it); dv.setdefault(s, []).append(it)
res['dev'] = {k: lr(v) | {'n': len(v)} for k, v in dv.items()}
# dev error rows that are 'new' relative to fit + Claude (the definition the student's training data implies)
em = dev.root_code != 'NONE'
res['dev_error_rows_new_vs_fit_plus_claude'] = r(np.mean([qcount.get(q, Counter()).get(t, 0) == 0 for q, t in zip(dev.question_id[em], dev.root_code[em])]))
json.dump(res, open(OUT / 's7_seen_cue_lr.json', 'w'), indent=1); print(json.dumps(res, indent=1))
