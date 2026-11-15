import json, re, collections
from k12.core import read_csv
R = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
T = [json.loads(x) for x in open(R+'data/v5/cand_train.jsonl')]
tr = read_csv(R+'data/raw/train.csv'); fr = dict(zip(tr.trace_id, tr.root_code))
claude = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(R+'runs/claude/unl_all_v4.jsonl')}
CODE = re.compile(r'\b([A-Z][A-Za-z]{1,2}-\d\d|NONE)\b')
c = collections.Counter(); shown = 0
for r in T:
    t = claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else fr[r['trace_id']]
    w = r['target'].split('\nWHY: ')[1].split('\nANSWER:')[0]; lw = w.lower()
    noerr = bool(re.search(r'no major error|has no (major )?error|is correct throughout|working is (fully )?correct', lw))
    c[(r['source'], 'NONE' if t == 'NONE' else 'err', 'says_no_error' if noerr else '-')] += 1
    named = set(CODE.findall(w))
    if r['source'] == 'gold' and named and t not in named and shown < 4:
        shown += 1; print('GOLD-WHY-OTHER', r['trace_id'], 'truth', t, named, '|', w[:260])
    if r['source'] == 'gold' and t != 'NONE' and noerr and shown < 8:
        shown += 1; print('GOLD-ERR-SAYS-NOERR', r['trace_id'], t, '|', w[:260])
print(dict(c))
