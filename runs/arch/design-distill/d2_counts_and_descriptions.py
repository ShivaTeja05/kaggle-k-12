"""(a) How much 'new code for a known question' signal the 26k Claude labels contain (curriculum sizing).
(b) Zero-shot rationale->code by similarity to taxonomy name+description (no training), on dev Claude reasoning.
(c) Reasoning length (generation budget for a reason-first student)."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from k12.core import read_csv, label_space, make_folds, row_scores, topk, normalize

OUT = ROOT / 'runs/arch/design-distill'
tax = read_csv(ROOT / 'data/raw/taxonomy_v1.csv')
classes = label_space(tax)
ci = {c: i for i, c in enumerate(classes)}
train = read_csv(ROOT / 'data/raw/train.csv')
fold = make_folds(train)
dev = train[fold == 0].reset_index(drop=True)
fit = train[fold != 0]
test = read_csv(ROOT / 'data/raw/test.csv')
unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
U = {}
for line in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); a = r.get('answer') or {}
    if a.get('first_code') in ci and r['trace_id'] in unl.index:
        U[r['trace_id']] = a
D = {}
for line in open(ROOT / 'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); D[r['trace_id']] = r.get('answer') or {}

res = {}
full_codes = train.groupby('question_id').root_code.apply(set).to_dict()
q_of = unl.loc[list(U)].question_id
first = np.array([U[t]['first_code'] for t in U])
qs = q_of.values
err = first != 'NONE'
new = np.array([(q in full_codes) and (c not in full_codes[q]) for q, c in zip(qs, first)])
res['claude_unl_rows'] = len(U)
res['claude_error_rows'] = int(err.sum())
res['claude_error_rows_code_new_for_question_vs_full_train'] = int((err & new).sum())
res['claude_error_rows_question_not_in_train'] = int((err & np.array([q not in full_codes for q in qs])).sum())
import collections
per_q = collections.defaultdict(set)
for q, c in zip(qs, first):
    if c != 'NONE':
        per_q[q].add(c)
res['questions_with_>=2_distinct_claude_error_codes'] = sum(len(v) >= 2 for v in per_q.values())
res['test_questions'] = int(test.question_id.nunique())
res['test_rows_whose_question_has_claude_unl_labels'] = float(np.mean([q in set(qs) for q in test.question_id]))
# contrastive pairs: same question, both error, different first code
cnt = collections.Counter((q, c) for q, c in zip(qs, first) if c != 'NONE')
pairs = 0
byq = collections.defaultdict(list)
for (q, c), n in cnt.items():
    byq[q].append(n)
for q, ns in byq.items():
    s = sum(ns); pairs += (s * s - sum(n * n for n in ns)) // 2
res['same_question_different_code_error_pairs'] = int(pairs)

# reasoning length (whitespace tokens; ~1.3 BPE tokens per word is an estimate)
lens = np.array([len((a.get('reasoning') or '').split()) for a in U.values()])
tl = np.array([len((a.get('transcription') or '').split()) for a in U.values()])
res['reasoning_words_median_p90'] = [int(np.median(lens)), int(np.percentile(lens, 90))]
res['transcription_words_median_p90'] = [int(np.median(tl)), int(np.percentile(tl, 90))]
res['root_line_nonempty_share_on_error_rows'] = float(np.mean([bool(U[t].get('root_line')) for t, e in zip(U, err) if e]))

# zero-shot description similarity
CODE_RE = re.compile(r'\b(?:[A-Z][A-Za-z]{1,3}-\d{2}|NONE)\b')
desc = {r.id: f"{r.name}. {r.description}" for r in tax.itertuples()}
desc['NONE'] = 'No error. The working is correct and matches the answer key. No major error.'
docs = [desc[c] for c in classes]
ids = dev.trace_id.tolist(); y = dev.root_code.tolist()
reas = [CODE_RE.sub(' ', D.get(t, {}).get('reasoning') or '') for t in ids]
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english').fit(docs + [CODE_RE.sub(' ', a.get('reasoning') or '') for a in list(U.values())[:20000]])
S = (vec.transform(reas) @ vec.transform(docs).T).toarray()
p = normalize(np.exp(S * 20))
s = row_scores(y, topk(p, classes))
fit_codes = fit.groupby('question_id').root_code.apply(set).to_dict()
def b(r):
    if r.root_code == 'NONE': return 'NONE'
    if r.question_id not in fit_codes: return 'q_unseen'
    return 'code_seen_for_q' if r.root_code in fit_codes[r.question_id] else 'code_unseen_for_q'
bins = np.array([b(r) for r in dev.itertuples()])
res['zero_shot_description_similarity_on_claude_reasoning'] = {'map3': round(float(s.mean()), 4),
    'bins': {k: round(float(s[bins == k].mean()), 4) for k in ['code_seen_for_q', 'code_unseen_for_q', 'q_unseen', 'NONE']}}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / 'd2_counts_and_descriptions.json', 'w'), indent=1)
