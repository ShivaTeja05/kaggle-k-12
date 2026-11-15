import json, re, collections, numpy as np, pandas as pd
from k12.core import read_csv, make_folds
R = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
T = [json.loads(x) for x in open(R+'data/v5/cand_train.jsonl')]; D = [json.loads(x) for x in open(R+'data/v5/cand_dev.jsonl')]
u = read_csv(R+'data/semisupervised/competition_data/unlabelled.csv'); tr = read_csv(R+'data/raw/train.csv'); te = read_csv(R+'data/raw/test.csv')
ids = [r['trace_id'] for r in T if '\nTopic: nan\n' in r['prompt']]
sub = u[u.trace_id.isin(ids)]
print('raw topic values for nan rows', sub.topic_name.map(repr).value_counts().head(5).to_dict(), 'question ids', sub.question_id.nunique())
raw = pd.read_csv(R+'data/semisupervised/competition_data/unlabelled.csv', dtype=str, keep_default_na=False)
print('raw text topic for those', raw[raw.trace_id.isin(ids)].topic_name.value_counts().head(5).to_dict())
print('unl dup trace ids', u.trace_id.duplicated().sum())
for name, df in (('train', tr), ('test', te), ('unl', u)):
    rr = pd.read_csv(R+('data/semisupervised/competition_data/unlabelled.csv' if name == 'unl' else f'data/raw/{name}.csv'), dtype=str, keep_default_na=False)
    print(name, 'literal nan/NaN/None/empty topic', int(rr.topic_name.str.strip().str.lower().isin(['nan', 'none', 'null', '']).sum()), 'answer', int(rr.correct_answer.str.strip().str.lower().isin(['nan', 'none', 'null', '']).sum()),
          'answer values', rr.correct_answer[rr.correct_answer.str.strip().str.lower().isin(['nan', 'none', 'null', ''])].map(repr).value_counts().to_dict())
# same question ids with topic present in other rows?
q = set(sub.question_id); other = u[u.question_id.isin(q) & ~u.trace_id.isin(ids)]
print('other unl rows of those questions topic', other.topic_name.map(repr).value_counts().head(3).to_dict())
print('train/test rows on those questions', int(tr.question_id.isin(q).sum()), int(te.question_id.isin(q).sum()))
# unseen-pair dev rows: truth has example?
folds = make_folds(tr); dev = tr[folds == 0]; fit = tr[folds != 0]
seen = collections.defaultdict(set)
for a, b in zip(fit.question_id, fit.root_code): seen[a].add(b)
EX = re.compile(r' On this question, e\.g\. "(.*)"$')
dd = {d['trace_id']: d for d in D}; c = collections.Counter()
for r in dev.itertuples():
    if r.root_code == 'NONE': continue
    kind = 'unseen_q' if r.question_id not in seen else ('new_code' if r.root_code not in seen[r.question_id] else 'seen_code')
    d = dd[r.trace_id]; codes = d['codes_per_prompt'][0]
    ls = d['prompts'][0].split('\nCandidate error codes:\n', 1)[1].split('\nFind the root')[0].split('\n')
    c[(kind, 'n')] += 1
    if r.root_code in codes:
        c[(kind, 'in_cands')] += 1; c[(kind, 'truth_has_example')] += bool(EX.search(ls[codes.index(r.root_code)]))
print({f'{a}/{b}': v for (a, b), v in sorted(c.items())})
