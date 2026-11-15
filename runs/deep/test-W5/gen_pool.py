"""Extra donors: Claude-transcribed unlabelled photos that Claude labelled NONE (no dev/test rows involved).
At most 4 synthetic rows per question and 1 per donor.  Output: synthetic_pool.jsonl"""
import json, random
from collections import Counter, defaultdict
import pandas as pd
from inject_malrules import RULES, inject, OUT
ROOT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'
unl = pd.read_csv(ROOT + '/data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
rng = random.Random(7)
don = []
for l in open(ROOT + '/runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code') == 'NONE' and (a.get('transcription') or '').strip() and r['trace_id'] in unl.index:
        don.append((r['trace_id'], a['transcription']))
rng.shuffle(don)
perq, rows, per = Counter(), [], Counter()
for t, tx in don:
    u = unl.loc[t]
    if perq[u.question_id] >= 4:
        continue
    order = list(range(len(RULES))); rng.shuffle(order)
    for i in order:
        out = inject(tx, RULES[i], rng)
        if out:
            perq[u.question_id] += 1; per[out['rule']] += 1
            rows.append(dict(trace_id=f'synp_{t}', donor=t, question_id=u.question_id, topic_name=u.topic_name,
                             question_text=u.question_text, root_code=out['rule'], **{k: v for k, v in out.items() if k != 'rule'}, rule=out['rule']))
            break
with open(OUT + '/synthetic_pool.jsonl', 'w') as fh:
    for x in rows: fh.write(json.dumps(x, ensure_ascii=False) + '\n')
st = dict(pool_none_donors=len(don), rows=len(rows), questions=len(perq), by_code=per)
json.dump(st, open(OUT + '/synthetic_pool_stats.json', 'w'), indent=1); print(json.dumps(st, indent=1))
