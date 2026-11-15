"""CPU probe: add a (question, final written value) signature table built from Claude-labelled
unlabelled photos to existing fold-0 development probabilities. Dev final values come from the
machine transcriber (runs/pod/tx/tx_dev.json), as at test time. Aggregate metrics only."""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, 'src'); sys.path.insert(0, 'runs/research/q4-arxiv-2508')
from k12.core import read_csv
from answer_signature_probe_lib import final_value

def load(p):
    z = np.load(p, allow_pickle=True); return list(z['ids']), list(z['classes']), z['p']

def map3(P, gold, classes):
    top = np.argsort(-P, 1)[:, :3]; ci = {c: i for i, c in enumerate(classes)}
    return np.array([next((1/(k+1) for k in range(3) if top[r, k] == ci[g]), 0.) for r, g in enumerate(gold)])

train = read_csv('data/raw/train.csv').set_index('trace_id')
names = {'text32_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
         'text8_tx_e3': 'runs/pod/tx/text_tx_dev_e3.npz',
         'vision_distill': 'runs/pod/dev-distill/validation_epoch_2.npz'}
ids, classes, _ = load(names['text32_tx'])
probs = {}
for k, p in names.items():
    i2, c2, P = load(p); assert c2 == classes
    probs[k] = pd.DataFrame(P, index=i2).loc[ids].values
gold = train.loc[ids].root_code.values
tx = json.load(open('runs/pod/tx/tx_dev.json'))
fv = [final_value(tx.get(t)) for t in ids]
cl = pd.read_json('data/claude_dataset/labels.jsonl', lines=True)
cl = cl[cl.split == 'unlabelled'].drop_duplicates('trace_id'); cl['fv'] = cl.transcription.map(final_value)
by_k = {k: Counter(g.first_code) for k, g in cl.dropna(subset=['fv']).groupby(['question_id', 'fv'])}
by_q = {q: Counter(g.first_code) for q, g in cl.groupby('question_id')}
ci = {c: i for i, c in enumerate(classes)}
def dist(counter):
    v = np.zeros(len(classes))
    for c, n in counter.items():
        if c in ci: v[ci[c]] += n
    return v
S = np.ones((len(ids), len(classes))) / len(classes); covered = np.zeros(len(ids), bool)
qids = train.loc[ids].question_id.values
for r, (q, f) in enumerate(zip(qids, fv)):
    if f is None or (q, f) not in by_k: continue
    covered[r] = True
    qd = dist(by_q.get(q, Counter())); qd = (qd + 0.1) / (qd.sum() + 0.1 * len(classes))
    kd = dist(by_k[(q, f)]); S[r] = (kd + 2.0 * qd) / (kd.sum() + 2.0)
rng = np.random.default_rng(0); uq = np.unique(qids)
def boot(delta):
    out = []
    for _ in range(2000):
        pick = rng.choice(uq, len(uq)); m = np.concatenate([np.where(qids == q)[0] for q in pick])
        out.append(delta[m].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
res = {'rows': len(ids), 'dev_final_value_rate_machine_tx': float(np.mean([f is not None for f in fv])),
       'signature_coverage_dev': float(covered.mean()), 'blends': {}}
for wv in [0.5, 1.0]:
    logp = np.log(probs['text32_tx'] + 1e-9) + 0.5 * np.log(probs['text8_tx_e3'] + 1e-9) + wv * np.log(probs['vision_distill'] + 1e-9)
    base = map3(logp, gold, classes)
    entry = {'base_map3': float(base.mean())}
    for w in [0.1, 0.25, 0.5]:
        new = map3(logp + w * np.log(S), gold, classes); d = new - base
        entry[f'sig_w{w}'] = {'map3': float(new.mean()), 'delta': float(d.mean()), 'delta_95_question_boot': boot(d),
                              'improved': int((d > 0).sum()), 'regressed': int((d < 0).sum()),
                              'covered_error_rows_delta': float(d[covered & (gold != 'NONE')].mean())}
    res['blends'][f'text32+0.5*text8+{wv}*vision'] = entry
print(json.dumps(res, indent=1))
