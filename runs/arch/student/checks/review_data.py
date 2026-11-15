import json, re, numpy as np
from collections import Counter
from k12.core import read_csv, label_space, make_folds
L='ABCDEFGHIJ'
tr=read_csv('data/raw/train.csv'); tax=read_csv('data/raw/taxonomy_v1.csv'); sec=dict(zip(tax.id, tax.section)); sec['NONE']='NONE'
rows=[json.loads(x) for x in open('data/v5/cand_train.jsonl')]; dev=[json.loads(x) for x in open('data/v5/cand_dev.jsonl')]
test=[json.loads(x) for x in open('data/v5/cand_test.jsonl')]
# 1 duplicate codes in claude first/second/third
cl=[json.loads(x) for x in open('runs/claude/unl_all_v4.jsonl')]
dup=sum(1 for r in cl if (a:=r.get('answer') or {}) and a.get('first_code') and (a.get('first_code') in (a.get('second_code'),a.get('third_code')) or (a.get('second_code') and a.get('second_code')==a.get('third_code'))))
print('claude rows with duplicate ranked codes', dup, 'of', len(cl))
# soft max letter == answer letter?
bad=[r['trace_id'] for r in rows if max(r['soft'],key=r['soft'].get)!=r['target'][-1]]
print('train rows where argmax soft != answer letter', len(bad))
print('soft sizes', Counter(len(r['soft']) for r in rows if r['source']=='claude'))
# ROOT_LINE distribution
print('ROOT_LINE dash share claude', np.mean([re.search(r'ROOT_LINE: (\S+)',r['target']).group(1)=='-' for r in rows if r['source']=='claude' and r['codes'][L.index(r['target'][-1])]!='NONE']))
print('ROOT_LINE dash share gold err', np.mean([re.search(r'ROOT_LINE: (\S+)',r['target']).group(1)=='-' for r in rows if r['source']=='gold' and r['codes'][L.index(r['target'][-1])]!='NONE']))
# section shortcut: is truth's section the plurality non-NONE section among candidates
def plural(codes, truth):
    c=Counter(sec[x] for x in codes if x!='NONE'); 
    if truth=='NONE' or not c: return None
    top=max(c.values()); return c[sec[truth]]==top, sum(1 for x in codes if x!='NONE' and sec[x]==sec[truth])
for name, data in (('claude',[(r['codes'],r['codes'][L.index(r['target'][-1])]) for r in rows if r['source']=='claude']),
                   ('gold',[(r['codes'],r['codes'][L.index(r['target'][-1])]) for r in rows if r['source']=='gold']),
                   ('dev',[(d['codes_per_prompt'][0],d['truth']) for d in dev if d['truth'] in d['codes_per_prompt'][0]])):
    v=[plural(c,t) for c,t in data]; v=[x for x in v if x]
    print(name,'truth section is plurality', round(np.mean([a for a,_ in v]),3), 'mean same-section cands', round(np.mean([b for _,b in v]),2), 'n',len(v))
# example-presence shortcut
def has_ex(prompt, letter):
    m=re.search(r'^'+letter+r'\. .*$', prompt, re.M); return 'On this question, e.g.' in m.group(0)
for name, data in (('claude',[(r['prompt'],r['target'][-1],r['codes']) for r in rows if r['source']=='claude']),
                   ('gold',[(r['prompt'],r['target'][-1],r['codes']) for r in rows if r['source']=='gold']),
                   ('dev',[(d['prompts'][0],L[d['codes_per_prompt'][0].index(d['truth'])],d['codes_per_prompt'][0]) for d in dev if d['truth'] in d['codes_per_prompt'][0]])):
    te=[];oth=[];n=[]
    for p,a,c in data:
        if c[L.index(a)]=='NONE': continue
        te.append(has_ex(p,a)); ex=[has_ex(p,l) for l,x in zip(L,c) if x!='NONE' and l!=a]; oth.append(np.mean(ex)); n.append(sum(ex)+has_ex(p,a))
    print(name,'P(truth has example)',round(np.mean(te),3),'P(other has example)',round(np.mean(oth),3),'mean #examples',round(np.mean(n),2))
# NONE position in dev order 1
print('dev NONE position in order1', Counter(d['codes_per_prompt'][0].index('NONE') for d in dev).most_common(4))
print('test NONE position in order1', Counter(d['codes_per_prompt'][0].index('NONE') for d in test).most_common(4))
# transcript formats
def wk(p): return p.split('Student working (line by line):\n',1)[1].split('\n\nCandidate error codes:')[0]
for name, ps in (('claude',[r['prompt'] for r in rows if r['source']=='claude']),('gold',[r['prompt'] for r in rows if r['source']=='gold']),('dev',[d['prompts'][0] for d in dev]),('test',[d['prompts'][0] for d in test])):
    print(name,'share with L1| lines', round(np.mean([bool(re.search(r'^\s*L\d+\s*\|',wk(p),re.M)) for p in ps]),3), 'mean chars', int(np.mean([len(wk(p)) for p in ps])), 'share at 1500 cap', round(np.mean([len(wk(p))>=1500 for p in ps]),3))
# ROOT_LINE reference beyond the kept tail
miss=0;tot=0
for r in rows:
    m=re.search(r'ROOT_LINE: L(\d+)',r['target'])
    if m:
        tot+=1; miss+= not re.search(r'^\s*L'+m.group(1)+r'\s*\|',wk(r['prompt']),re.M)
print('targets whose ROOT_LINE is not present in the shown working', miss,'of',tot)
