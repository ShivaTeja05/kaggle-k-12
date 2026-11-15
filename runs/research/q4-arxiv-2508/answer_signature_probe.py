"""CPU probe: does (question_id, student's final written value) predict the root code?
Inspired loosely by 'match' ideas; aggregate metrics only, no student text printed."""
import json, re, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
from k12.core import make_folds, read_csv

def final_value(text):
    if not isinstance(text, str) or not text.strip(): return None
    lines = [l.split('|', 1)[-1] for l in text.strip().splitlines() if l.strip()]
    for line in reversed(lines):
        s = line.replace('−', '-').replace('–', '-').replace(' ', '').replace('$', '')
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', s)
        parts = re.split(r'=|≈', s)
        cand = parts[-1].strip(' .,')
        m = re.findall(r'-?\d+(?:[.,]\d+)?(?:/-?\d+(?:[.,]\d+)?)?', cand)
        if m: return m[-1].replace(',', '.')
    return None

def map3(gold, ranked):
    return np.mean([next((1/(k+1) for k, c in enumerate(r[:3]) if c == g), 0) for g, r in zip(gold, ranked)])

def rank(counter, fallback):
    out = [c for c, _ in counter.most_common()]
    for c in fallback:
        if c not in out: out.append(c)
    return out[:3]

train = read_csv('data/raw/train.csv')
train['fv'] = train.transcription_steps.map(final_value)
folds = make_folds(train, seed=2026)
glob = [c for c, _ in Counter(train.root_code).most_common()]
res = {'final_value_extracted_rate_gold': float(train.fv.notna().mean())}

# A) cross-fitted on the 1,500 labelled rows (gold transcriptions)
pri, sig, gold, cov = [], [], [], []
for f in sorted(set(folds)):
    fit, val = train[folds != f], train[folds == f]
    by_q = {q: Counter(g.root_code) for q, g in fit.groupby('question_id')}
    by_k = {k: Counter(g.root_code) for k, g in fit.dropna(subset=['fv']).groupby(['question_id', 'fv'])}
    for r in val.itertuples():
        qp = rank(by_q.get(r.question_id, Counter()), glob)
        k = (r.question_id, r.fv)
        hit = r.fv is not None and k in by_k
        cov.append(hit); gold.append(r.root_code); pri.append(qp)
        sig.append(rank(by_k[k] + Counter(), qp) if hit else qp)
cov = np.array(cov); gold = np.array(gold)
res['A_labelled_crossfit'] = {
    'rows': int(len(gold)), 'signature_coverage': float(cov.mean()),
    'map3_question_prior': float(map3(gold, pri)), 'map3_signature_then_prior': float(map3(gold, sig)),
    'covered_rows_map3_prior': float(map3(gold[cov], [p for p, c in zip(pri, cov) if c])),
    'covered_rows_map3_signature': float(map3(gold[cov], [s for s, c in zip(sig, cov) if c])),
    'covered_rows_top1_signature': float(np.mean([s[0] == g for s, g, c in zip(sig, gold, cov) if c])),
}

# B) Claude-labelled unlabelled pool as the table, evaluated on all 1,500 labelled rows
cl = pd.read_json('data/claude_dataset/labels.jsonl', lines=True)
cl = cl[cl.split == 'unlabelled'].drop_duplicates('trace_id')
cl['fv'] = cl.transcription.map(final_value)
res['claude_pool_rows'] = int(len(cl)); res['claude_pool_final_value_rate'] = float(cl.fv.notna().mean())
by_kc = {k: Counter(g.first_code) for k, g in cl.dropna(subset=['fv']).groupby(['question_id', 'fv'])}
by_qc = {q: Counter(g.first_code) for q, g in cl.groupby('question_id')}
pri, sig, cov, n = [], [], [], []
for r in train.itertuples():
    qp = rank(by_qc.get(r.question_id, Counter()), glob)
    k = (r.question_id, r.fv); hit = r.fv is not None and k in by_kc
    cov.append(hit); pri.append(qp); sig.append(rank(by_kc[k], qp) if hit else qp)
    n.append(sum(by_kc[k].values()) if hit else 0)
cov = np.array(cov); gold = train.root_code.values; n = np.array(n)
err = gold != 'NONE'
res['B_claude_table_on_labelled'] = {
    'signature_coverage': float(cov.mean()),
    'map3_claude_question_prior': float(map3(gold, pri)), 'map3_signature_then_prior': float(map3(gold, sig)),
    'covered_rows_map3_prior': float(map3(gold[cov], [p for p, c in zip(pri, cov) if c])),
    'covered_rows_map3_signature': float(map3(gold[cov], [s for s, c in zip(sig, cov) if c])),
    'covered_error_rows': int((cov & err).sum()),
    'covered_error_rows_map3_signature': float(map3(gold[cov & err], [s for s, c in zip(sig, cov & err) if c])),
    'covered_error_rows_map3_prior': float(map3(gold[cov & err], [p for p, c in zip(pri, cov & err) if c])),
    'covered_n_ge5_rows': int((cov & (n >= 5)).sum()),
    'covered_n_ge5_top1_signature': float(np.mean([s[0] == g for s, g, c in zip(sig, gold, cov & (n >= 5)) if c])),
}
print(json.dumps(res, indent=1))
