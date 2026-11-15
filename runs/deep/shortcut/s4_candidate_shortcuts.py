"""(4b) Which shortcuts remain learnable in the candidate-student data (data/v5)?

Rules a student could learn without reading the working, scored on train rows and on dev rows:
- QFREQ: pick the candidate most frequent for this question in fit gold + Claude rows (leave-own-row-out).
- QSEEN: restrict to candidates ever seen for the question, then uniform.
- EXAMPLE: pick a candidate that carries an "On this question, e.g." example.
- POSITION/ORDER: dev order 1 is the blend's ranking (train order is shuffled, so not learnable; reported for reference).
Also the share of train rows whose truth is a 'new pair' for the question (absent from the other rows' codes),
i.e. how often training contradicts the question->code shortcut, vs dev.
"""
import json
from collections import Counter
import numpy as np
import pandas as pd
from sc_common import load, claude_unl, slices, OUT, ROOT, r

tr, tax, classes, dev, fit = load()
cl, unl = claude_unl()
S, qc = slices(dev, fit)
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
L = 'ABCDEFGHIJ'
train = [json.loads(x) for x in open(ROOT / 'data/v5/cand_train.jsonl')]
devc = [json.loads(x) for x in open(ROOT / 'data/v5/cand_dev.jsonl')]
dev_ids = set(dev.trace_id)
cl_ok = cl[~cl.trace_id.isin(dev_ids)]
# Same counts the builder uses: fit gold + all valid Claude first codes
qcount = {}
for q, c in list(zip(fit.question_id, fit.root_code)) + list(zip(cl_ok.question_id, cl_ok.root_code)):
    qcount.setdefault(q, Counter())[c] += 1
own = dict(zip(cl.trace_id, cl.root_code)) | dict(zip(fit.trace_id, fit.root_code))


def rules(codes, truth, q, tid, prompt=None, exclude_own=True):
    cnt = Counter(qcount.get(q, Counter()))
    if exclude_own and tid in own:
        cnt[own[tid]] -= 1
    freq = [cnt.get(c, 0) for c in codes]
    best = int(np.argmax(freq))
    seen = [c for c, f in zip(codes, freq) if f > 0]
    out = dict(qfreq_top1=codes[best] == truth,
               qseen_contains_truth=truth in seen, qseen_size=len(seen),
               qseen_uniform_acc=(truth in seen) / max(1, len(seen)) if seen else 1 / len(codes),
               truth_new_pair=cnt.get(truth, 0) <= 0, n_cands=len(codes))
    # MAP@3 of ranking candidates by question frequency (ties by list order)
    order = sorted(range(len(codes)), key=lambda i: -freq[i])[:3]
    hit = [k for k, i in enumerate(order) if codes[i] == truth]
    out['qfreq_map3'] = 1 / (hit[0] + 1) if hit else 0.
    if prompt is not None:
        block = prompt.split('\nCandidate error codes:\n', 1)[1].split('\n')
        ex = [i for i in range(len(codes)) if 'On this question, e.g.' in block[i]]
        out['example_rule_acc'] = (codes.index(truth) in ex) / len(ex) if ex else None
    return out


def summarise(rows):
    df = pd.DataFrame(rows)
    out = dict(n=len(df))
    for k in ['qfreq_top1', 'qfreq_map3', 'qseen_contains_truth', 'qseen_size', 'qseen_uniform_acc', 'truth_new_pair']:
        out[k] = r(df[k].mean())
    if 'example_rule_acc' in df:
        out['example_rule_acc'] = r(df.example_rule_acc.dropna().mean())
    out['chance_top1'] = r((1 / df.n_cands).mean())
    return out


res = {}
tr_rows = {'claude': [], 'gold': [], 'claude_error': [], 'gold_error': []}
for x in train:
    truth = x['codes'][L.index(x['target'][-1])]
    q = qmap[x['trace_id']]
    rr = rules(x['codes'], truth, q, x['trace_id'], x['prompt'])
    tr_rows[x['source']].append(rr)
    if truth != 'NONE':
        tr_rows[x['source'] + '_error'].append(rr)
res['train'] = {k: summarise(v) for k, v in tr_rows.items()}
dv = {s: [] for s in ['all', 'none', 'seen_pair', 'new_pair', 'unseen_q']}
blend_top1 = {s: [] for s in dv}
for x, s, q in zip(devc, S, dev.question_id):
    codes = x['codes_per_prompt'][0]; truth = x['truth']
    if truth not in codes:
        rr = dict(qfreq_top1=False, qfreq_map3=0., qseen_contains_truth=False, qseen_size=0, qseen_uniform_acc=0.,
                  truth_new_pair=True, n_cands=len(codes), example_rule_acc=0.)
    else:
        rr = rules(codes, truth, q, x['trace_id'], x['prompts'][0], exclude_own=False)
    for k in ('all', s):
        dv[k].append(rr); blend_top1[k].append(codes[0] == truth)
res['dev'] = {k: summarise(v) | dict(blend_order_top1=r(np.mean(blend_top1[k])),
                                     recall_at_10=r(np.mean([d['truth'] in d['codes_per_prompt'][0] for d, s in zip(devc, S) if k == 'all' or s == k])))
              for k, v in dv.items()}
# Rows per question in cand_train and the share of the student's training rows on dev questions
qs = pd.Series([qmap[x['trace_id']] for x in train])
res['cand_train_rows_per_q'] = dict(questions=int(qs.nunique()), mean=r(qs.value_counts().mean(), 2),
                                    rows_on_dev_questions_share=r(qs.isin(set(dev.question_id)).mean()),
                                    rows_on_test_questions_share=r(qs.isin(set(pd.read_csv(ROOT / 'data/raw/test.csv', dtype=str).question_id)).mean()))
# Does the dev candidate list on new-pair rows consist of the question's codes? (bias inherited from the blend)
fitcodes = qc
share = {s: [] for s in ['seen_pair', 'new_pair', 'none']}
for x, s, q in zip(devc, S, dev.question_id):
    if s in share:
        share[s].append(np.mean([c in fitcodes.get(q, set()) for c in x['codes_per_prompt'][0] if c != 'NONE']))
res['dev_candidates_share_fit_codes_of_q'] = {k: r(np.mean(v)) for k, v in share.items()}
tshare = []
for x in train:
    q = qmap[x['trace_id']]; truth = x['codes'][L.index(x['target'][-1])]
    cnt = qcount.get(q, Counter())
    tshare.append(np.mean([cnt.get(c, 0) > 0 for c in x['codes'] if c not in ('NONE', truth)]))
res['train_distractors_share_seen_for_q'] = r(np.mean(tshare))
json.dump(res, open(OUT / 's4_candidate_shortcuts.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
