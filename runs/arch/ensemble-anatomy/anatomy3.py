"""Part 3: question-memorisation split of the gap."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from evaluate_claude import claude_scores
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
train = read_csv('data/raw/train.csv'); classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
folds = make_folds(train); fit = train[folds != 0]; dev = train[folds == 0].reset_index(drop=True)
ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy(); N = len(y)
B = {'q32': 'runs/pod/text32/validation_tx_dev_epoch_1.npz', 'q8': 'runs/pod/tx/text_tx_dev_e3.npz',
     'vl': 'runs/pod/dev-distill/validation_epoch_2.npz', 'q8_s0': 'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz',
     'q8_s100': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', 'fs235': 'runs/pod/probe235/fewshot235/validation.npz'}
P = {k: aligned_bundle(v, ids, classes) for k, v in B.items()}
P['claude'] = claude_scores('runs/claude/dev_opus.jsonl', ids, classes)[0]
lp = sum(w*np.log(P[k]) for k, w in [('q32', .35), ('q8', .30), ('vl', .35)])
P['blend'] = normalize(np.exp(lp - lp.max(1, keepdims=True)))
S = {k: row_scores(y, topk(p, classes)) for k, p in P.items()}
qcodes = fit.groupby('question_id').root_code.apply(set).to_dict()
seen = np.array([q in qcodes for q in qid]); inq = np.array([y[i] in qcodes.get(qid[i], set()) for i in range(N)])
grp = np.where(~seen, 'unseen_question', np.where(inq, 'seen_truth_in_fit_qcodes', 'seen_truth_NOT_in_fit_qcodes'))
rows = []
for g in ['seen_truth_in_fit_qcodes', 'seen_truth_NOT_in_fit_qcodes', 'unseen_question']:
    m = grp == g
    d = {'group': g, 'n': int(m.sum())}
    for k in ['blend', 'claude', 'q32', 'q8', 'vl', 'q8_s0', 'q8_s100', 'fs235']:
        d[k] = round(float(S[k][m].mean()), 4)
    d['gap_map_contrib'] = round(float((S['claude'] - S['blend'])[m].sum() / N), 4)
    d['blend_lost_map'] = round(float((1 - S['blend'])[m].sum() / N), 4)
    # does each model's top1 land in the question's fit code set (memorisation)?
    for k in ['blend', 'claude', 'q8_s0']:
        t1 = topk(P[k], classes, 1)[:, 0]
        d[f'{k}_top1_in_fit_qcodes'] = round(float(np.mean([t1[i] in qcodes.get(qid[i], set()) for i in np.flatnonzero(m)])), 3) if g != 'unseen_question' else None
    d['none_rows'] = int((y[m] == 'NONE').sum())
    rows.append(d)
df = pd.DataFrame(rows); print(df.T.to_string())
Path('runs/arch/ensemble-anatomy/anatomy3.json').write_text(json.dumps(rows, indent=2) + '\n')
