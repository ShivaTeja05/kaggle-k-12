import json, re, collections, numpy as np
from k12.core import read_csv
R = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'; L = 'ABCDEFGHIJ'
T = [json.loads(x) for x in open(R+'data/v5/cand_train.jsonl')]; D = [json.loads(x) for x in open(R+'data/v5/cand_dev.jsonl')]; X = [json.loads(x) for x in open(R+'data/v5/cand_test.jsonl')]
claude = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(R+'runs/claude/unl_all_v4.jsonl')}
tr = read_csv(R+'data/raw/train.csv'); fr = dict(zip(tr.trace_id, tr.root_code))
for name, rows in (('train', [r['prompt'] for r in T]), ('dev', [p for d in D for p in d['prompts'][:1]]), ('test', [p for d in X for p in d['prompts'][:1]])):
    print(name, 'Topic: nan', sum('\nTopic: nan\n' in p for p in rows), 'Answer key: nan', sum('\nAnswer key: nan\n' in p for p in rows), 'of', len(rows))
print('train nan by source', collections.Counter((r['source'], '\nTopic: nan\n' in r['prompt'], '\nAnswer key: nan\n' in r['prompt']) for r in T))
for f in ('train', 'test'):
    d = read_csv(R+f'data/raw/{f}.csv'); print(f, 'raw topic missing', int(d.topic_name.isna().sum()), 'answer missing', int(d.correct_answer.isna().sum()))
u = read_csv(R+'data/semisupervised/competition_data/unlabelled.csv'); print('unl raw topic missing', int(u.topic_name.isna().sum()), 'answer missing', int(u.correct_answer.isna().sum()))
EX = re.compile(r' On this question, e\.g\. "(.*)"$')
def lines(p): return p.split('\nCandidate error codes:\n', 1)[1].split('\nFind the root major error')[0].split('\n')
def stats(rows, truthf, codesf, promptf):
    te = tn = de = dn = 0; short = collections.Counter(); shortcut = []
    for r in rows:
        t = truthf(r); codes = codesf(r); ls = lines(promptf(r))
        has = [bool(EX.search(l)) for l in ls]
        for c, h in zip(codes, has):
            if c == 'NONE': continue
            if c == t: te += h; tn += 1
            else: de += h; dn += 1
        for l in ls:
            m = EX.search(l)
            if m and len(m.group(1)) <= 3: short[m.group(1)] += 1
        if t != 'NONE':
            pool = [c for c, h in zip(codes, has) if h]
            shortcut.append(t in pool and len(pool) > 0 and 1 / len(pool))
    return dict(truth_has_example=round(te / max(tn, 1), 3), distractor_has_example=round(de / max(dn, 1), 3),
                n_tiny_spans=sum(short.values()), tiny=short.most_common(5), rand_pick_among_exemplified_acc=round(float(np.mean([s or 0 for s in shortcut])), 3))
tt = lambda r: claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else fr[r['trace_id']]
print('train claude', stats([r for r in T if r['source'] == 'claude'], tt, lambda r: r['codes'], lambda r: r['prompt']))
print('train gold', stats([r for r in T if r['source'] == 'gold'], tt, lambda r: r['codes'], lambda r: r['prompt']))
print('dev', stats(D, lambda r: r['truth'], lambda r: r['codes_per_prompt'][0], lambda r: r['prompts'][0]))
# WHY conclusion presence for claude error rows
ce = [r for r in T if r['source'] == 'claude' and tt(r) != 'NONE']
def norm(s): return ' '.join(str(s).split())
full_conc = [bool(re.search(r'\b(this matches|matches|so the root|root error)\b', norm(claude[r['trace_id']]['reasoning']).lower())) for r in ce]
why_conc = [bool(re.search(r'\b(this matches|matches|so the root|root error)\b', r['target'].split('\nWHY: ')[1].lower())) for r in ce]
print('claude err rows: full reasoning has "matches/root" conclusion', np.mean(full_conc), 'WHY keeps it', np.mean(why_conc))
print('claude reasoning words p50/p90', np.percentile([len(norm(claude[r['trace_id']]['reasoning']).split()) for r in T if r['source'] == 'claude'], [50, 90]))
g = [r for r in T if r['source'] == 'gold']
print('gold WHY truncated share', np.mean([len(r['target'].split('\nWHY: ')[1].split('\nANSWER:')[0]) < 0.9 * 1 + 0 or True for r in g]) and None)
ex = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(R+'runs/claude/train_explain_opus.jsonl')}
print('gold explain words p50/p90', np.percentile([len(norm(ex[r['trace_id']]['reasoning']).split()) for r in g], [50, 90]))
# gold NONE rows: does WHY say no error?
gn = [r for r in g if tt(r) == 'NONE']
print('gold NONE WHY mentions no/minor error', np.mean([bool(re.search(r'no (major )?error|correct|none|minor', r['target'].split('\nWHY: ')[1].lower())) for r in gn]))
# the one WHY not ending with punctuation
for r in T:
    w = r['target'].split('\nANSWER:')[0].rstrip()
    if w[-1] not in '.!?': print('NOPUNCT', r['source'], r['trace_id'], repr(w[-120:]))
# exemplar = truth row's own text? (train): example span of truth equals a line of own working
own = 0; tot = 0
for r in T:
    t = tt(r); ls = lines(r['prompt']); work = norm(r['prompt'].split('Student working (line by line):\n', 1)[1].split('\n\nCandidate')[0])
    for c, l in zip(r['codes'], ls):
        m = EX.search(l)
        if m and len(m.group(1)) >= 15:
            tot += 1; own += m.group(1) in work
print('train example spans (>=15ch) appearing verbatim in the row own working', own, 'of', tot)
own = tot = 0
for d in D:
    ls = lines(d['prompts'][0]); work = norm(d['prompts'][0].split('Student working (line by line):\n', 1)[1].split('\n\nCandidate')[0])
    for l in ls:
        m = EX.search(l)
        if m and len(m.group(1)) >= 15: tot += 1; own += m.group(1) in work
print('dev example spans (>=15ch) in own working', own, 'of', tot)
