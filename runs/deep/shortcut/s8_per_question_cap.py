"""(8) Design curve: capping Claude rows per question. For cap k, mix = fit gold + <=k Claude rows per question drawn from the
26,202 Claude labels (dev photos excluded; NONE limited to 1/3 of the picked rows like the distill sets). Report rows kept,
LOO new-pair share of error rows (how often training contradicts the question->code shortcut), the share of label entropy
question identity removes (LOO), and the up-weight new-pair rows would need to match the dev rate (37% of error rows are new
relative to fit+Claude)."""
import json
from collections import Counter
import numpy as np, pandas as pd
from sc_common import load, claude_unl, OUT, r
tr, tax, classes, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}; K = len(classes)
cl, unl = claude_unl(); cl = cl[~cl.trace_id.isin(set(dev.trace_id))]
def loo(df):
    y = df.root_code.map(ci).to_numpy(); g = np.bincount(y, minlength=K) + .5; g /= g.sum()
    tc = {t: np.bincount(y[i], minlength=K).astype(float) for t, i in df.groupby('topic_name').indices.items()}
    qq = {q: np.bincount(y[i], minlength=K).astype(float) for q, i in df.groupby('question_id').indices.items()}
    ll, newp, nerr = [], 0, 0
    for k, (q, t) in enumerate(zip(df.question_id, df.topic_name)):
        a = tc[t].copy(); a[y[k]] -= 1; b = qq[q].copy(); b[y[k]] -= 1
        p = (b + 2 * (a + 8 * g) / (a.sum() + 8)) / (b.sum() + 2); ll.append(-np.log(p[y[k]]))
        if classes[y[k]] != 'NONE': nerr += 1; newp += b[y[k]] == 0
    H = -(g * np.log(g)).sum()
    return newp / nerr, 1 - np.mean(ll) / H
res = {}
rng = np.random.default_rng(2026)
for cap in [0, 2, 4, 8, 16, 32, 10**6]:
    parts = []
    for q, g in cl.groupby('question_id'):
        e = g[g.root_code != 'NONE']; n = g[g.root_code == 'NONE']
        ke = min(len(e), cap); kn = min(len(n), max(0, min(cap - ke, ke // 2 + (1 if ke == 0 and cap > 0 else 0))))
        parts += [e.sample(n=ke, random_state=int(rng.integers(1e9))), n.sample(n=kn, random_state=int(rng.integers(1e9)))]
    c = pd.concat(parts) if parts else cl.iloc[:0]
    mix = pd.concat([fit[['question_id', 'topic_name', 'root_code']], c[['question_id', 'topic_name', 'root_code']]], ignore_index=True)
    newp, hq = loo(mix)
    w = (0.37 / 0.63) / (newp / (1 - newp))
    res[str(cap) if cap < 10**6 else 'none'] = dict(claude_rows=len(c), mix_rows=len(mix), loo_new_pair_share=r(newp),
                                                    share_H_removed_by_q=r(hq), upweight_needed_for_dev_rate=r(w, 2),
                                                    distinct_codes_claude=int(c.root_code.nunique()))
json.dump(res, open(OUT / 's8_per_question_cap.json', 'w'), indent=1); print(json.dumps(res, indent=1))
