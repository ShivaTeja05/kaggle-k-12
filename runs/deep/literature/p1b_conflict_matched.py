"""Size-matched comparison of 'bias-conflicting' share (label != leave-one-out modal code of its question)
between fit gold rows and Claude unlabelled first codes. Claude rows are subsampled per question to the
same count as fit gold rows for that question (20 draws), so LOO-mode noise is matched."""
import json, numpy as np, pandas as pd
from common import load, ROOT
tr, tax, classes, folds, dev, fit = load()
lab = pd.read_csv(ROOT/'data/claude_dataset/labels.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
unl = lab[lab.split == 'unlabelled'].drop_duplicates('trace_id', keep='last')
unl = unl[unl.first_code.isin(set(classes))]
def share(groups):
    s = n = 0
    for labs in groups:
        if len(labs) < 2: continue
        vc = pd.Series(labs).value_counts()
        for c in labs:
            loo = vc.copy(); loo[c] -= 1
            if loo.max() <= 0: continue
            s += int(c != loo.idxmax()); n += 1
    return s / max(n, 1), n
res = {}
for name, errs in [('all', False), ('errors_only', True)]:
    f = fit[fit.root_code != 'NONE'] if errs else fit
    u = unl[unl.first_code != 'NONE'] if errs else unl
    ug = {q: g.first_code.to_numpy() for q, g in u.groupby('question_id')}
    fg = {q: g.root_code.to_numpy() for q, g in f.groupby('question_id') if q in ug}
    gold = share(fg.values())
    rng = np.random.default_rng(0); draws = []
    for _ in range(20):
        groups = [rng.choice(ug[q], size=min(len(v), len(ug[q])), replace=False) for q, v in fg.items()]
        draws.append(share(groups)[0])
    res[name] = {'questions': len(fg), 'gold_conflict_share': round(gold[0], 3), 'gold_n': gold[1],
                 'claude_matched_conflict_share_mean': round(float(np.mean(draws)), 3),
                 'claude_matched_sd': round(float(np.std(draws)), 3),
                 'claude_rows_per_question_median': float(np.median([len(v) for v in ug.values()]))}
    # distinct codes per question (entropy proxy), matched size
    rng = np.random.default_rng(1)
    dg = np.mean([len(set(v)) for v in fg.values()])
    dc = np.mean([np.mean([len(set(rng.choice(ug[q], size=min(len(v), len(ug[q])), replace=False))) for _ in range(20)]) for q, v in fg.items()])
    res[name]['distinct_codes_per_q_gold'] = round(float(dg), 3); res[name]['distinct_codes_per_q_claude_matched'] = round(float(dc), 3)
json.dump(res, open(ROOT/'runs/deep/literature/p1b_conflict_matched.json', 'w'), indent=1); print(json.dumps(res, indent=1))
