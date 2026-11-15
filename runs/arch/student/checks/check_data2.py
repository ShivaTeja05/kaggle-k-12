import json, re, collections, numpy as np, sys
sys.path.insert(0, 'runs/arch/label-structure')
from common import *
R = ROOT; L = 'ABCDEFGHIJ'
tr, tax, classes, folds, dev, fit = load(); te = read_csv(R/'data/raw/test.csv')
T = [json.loads(x) for x in open(R/'data/v5/cand_train.jsonl')]; D = [json.loads(x) for x in open(R/'data/v5/cand_dev.jsonl')]; X = [json.loads(x) for x in open(R/'data/v5/cand_test.jsonl')]
claude = {}
for line in open(R/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); claude[r['trace_id']] = r.get('answer') or {}
unl = read_csv(R/'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
norm = lambda s: ' '.join(str(s).split())
strip = lambda s: re.sub(r'L\d+ \| ', '', norm(s)).lower()
# dup transcripts
devtx = {strip(t): tid for t, tid in zip(dev.transcription_steps, dev.trace_id)}
txt = json.load(open(R/'runs/pod/tx/tx_test.json')); testtx = collections.defaultdict(list)
for k, v in txt.items(): testtx[strip(v)].append(k)
for r in T:
    if r['source'] != 'claude': continue
    s = strip(claude[r['trace_id']]['transcription'])
    if s in devtx: print('DEV-DUP', r['trace_id'], unl.at[r['trace_id'], 'question_id'], dev.set_index('trace_id').question_id[devtx[s]], repr(s[:120]), 'claude', claude[r['trace_id']]['first_code'], 'gold', dev.set_index('trace_id').root_code[devtx[s]])
    if s in testtx: print('TEST-DUP', r['trace_id'], unl.at[r['trace_id'], 'question_id'], [te.set_index('trace_id').question_id[t] for t in testtx[s]], repr(s[:120]))
# any column overlap that would indicate same photo (image hashes?)
print('unl cols', unl.columns.tolist())
# explain keys
ex = {}
for l in open(R/'runs/claude/train_explain_opus.jsonl'):
    j = json.loads(l); ex[j['trace_id']] = j
k = next(iter(ex)); print('explain keys', list(ex[k].keys()), list((ex[k].get('answer') or {}).keys()))
print('explain covers dev ids', len(set(ex) & set(dev.trace_id)), 'fit', len(set(ex) & set(fit.trace_id)))
# WHY naming other code
CODE = re.compile(r'\b([A-Z][A-Za-z]{1,2}-\d\d|NONE)\b')
fr = dict(zip(tr.trace_id, tr.root_code)); n = 0
for r in T:
    t = claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else fr[r['trace_id']]
    w = r['target'].split('\nWHY: ')[1].split('\nANSWER:')[0]; named = set(CODE.findall(w))
    if named and t not in named and n < 6:
        n += 1; print('WHY-OTHER', r['source'], r['trace_id'], 'truth', t, 'named', named, '|', w[:220])
# Claude NONE WHY content
cn = [r for r in T if r['source'] == 'claude' and claude[r['trace_id']]['first_code'] == 'NONE']
w = [r['target'].split('\nWHY: ')[1].split('\nANSWER:')[0].lower() for r in cn]
full = [norm(claude[r['trace_id']]['reasoning']).lower() for r in cn]
print('claude NONE rows', len(cn), 'WHY says no major error/NONE', np.mean([('no major error' in x or 'none' in x.split() or 'none.' in x) for x in w]),
      'full reasoning says so', np.mean([('no major error' in x or 'none' in x.replace('.', ' ').split()) for x in full]), 'WHY truncated', np.mean([len(a) < len(b) - 5 for a, b in zip(w, full)]))
ce = [r for r in T if r['source'] == 'claude' and claude[r['trace_id']]['first_code'] != 'NONE']
w2 = [r['target'].split('\nWHY: ')[1].split('\nANSWER:')[0] for r in ce]; full2 = [norm(claude[r['trace_id']]['reasoning']) for r in ce]
print('claude err WHY truncated', np.mean([len(a) < len(b) - 5 for a, b in zip(w2, full2)]))
# fallback first-sentence-word-cut (no sentence end)
print('WHY not ending at sentence', sum(not r['target'].split('\nANSWER:')[0].rstrip()[-1] in '.!?' for r in T))
# header fields
bad = collections.Counter()
for r in T + [dict(prompt=p, trace_id=d['trace_id']) for d in D + X for p in d['prompts']]:
    p = r['prompt']
    for f in ('Grade: nan', 'Topic: nan', 'Question: nan', 'Answer key: nan', 'Answer key: \n', 'Question: \n', 'Answer key: None'):
        if f in p: bad[f] += 1
    if not p.startswith('Grade: '): bad['nostart'] += 1
    if not p.endswith('choose the matching candidate.'): bad['noend'] += 1
    if p.split('Student working (line by line):\n', 1)[1].startswith('\n\nCandidate'): bad['empty_working'] += 1
print('header issues', dict(bad))
# claude header matches unlabelled meta
mis = 0
for r in T:
    if r['source'] == 'claude':
        m = unl.loc[r['trace_id']]
        if not r['prompt'].startswith(f"Grade: {m['grade']}\nTopic: {m['topic_name']}\nQuestion: {m['question_text']}\nAnswer key: {m['correct_answer']}\n"): mis += 1
print('claude header mismatches', mis)
# dev blend reproduction via common.blend + s7 sinkhorn
ids = dev.trace_id.tolist(); P = blend(ids, classes)
ci = {c: i for i, c in enumerate(classes)}
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
b = np.zeros(len(classes))
for _ in range(30): b += np.log(pi / normalize(P * np.exp(b)).mean(0))
Q = normalize(P * np.exp(.5 * b))
top = [[classes[j] for j in np.argsort(-q, kind='stable')[:10]] for q in Q]
for t in top:
    if 'NONE' not in t: t[-1] = 'NONE'
dd = {d['trace_id']: d for d in D}
print('dev order1 exact', np.mean([dd[i]['codes_per_prompt'][0] == t for i, t in zip(ids, top)]), 'NONE swapped', sum('NONE' not in [classes[j] for j in np.argsort(-q)[:10]] for q in Q))
print('dev recall@10 of raw blend', np.mean([y in [classes[j] for j in np.argsort(-p)[:10]] for y, p in zip(dev.root_code, P)]))
# truth rank in order1 for error rows missed: which codes?
miss = collections.Counter(d['truth'] for d in D if d['truth'] not in d['codes_per_prompt'][0]); print('dev missed truths', dict(miss))
# train negative difficulty: share of candidates that are random fill (not truth/NONE/claude alt/question code/sibling)
sec = dict(zip(tax.id, tax.section))
qc = collections.defaultdict(set)
for q, c in zip(fit.question_id, fit.root_code): qc[q].add(c)
for t, a in claude.items():
    if t in unl.index: qc[unl.at[t, 'question_id']].add(a.get('first_code'))
q_of = {**dict(zip(tr.trace_id, tr.question_id)), **unl.question_id.to_dict()}
rand = []
for r in T:
    t = claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else fr[r['trace_id']]
    alt = {claude[r['trace_id']].get('second_code'), claude[r['trace_id']].get('third_code')} if r['source'] == 'claude' else set()
    rand.append(sum(c not in {t, 'NONE'} | alt | qc[q_of[r['trace_id']]] and sec.get(c) != sec.get(t) for c in r['codes']))
print('train random-fill candidates per row: mean', np.mean(rand), 'by source', {s: float(np.mean([x for x, r in zip(rand, T) if r['source'] == s])) for s in ('claude', 'gold')})
drand = [sum(c not in {'NONE'} | qc[q_of[d['trace_id']]] for c in d['codes_per_prompt'][0]) for d in D]
print('dev candidates never seen on question (fit+claude): mean', np.mean(drand))
# section sizes
print('dev truth seen on its question in fit/claude', np.mean([d['truth'] in qc[q_of[d['trace_id']]] for d in D if d['truth'] != 'NONE']))
