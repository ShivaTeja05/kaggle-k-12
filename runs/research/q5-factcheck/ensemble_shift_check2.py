"""Correct only members trained with Claude pseudo rows (text32, VL distill) using the v1 mix (claude_unl.jsonl, 3,427 rows); text8 (real-only) left alone."""
import json, csv, random, collections
import numpy as np
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train={r['trace_id']:r for r in csv.DictReader(open(R+'data/raw/train.csv'))}
dev=set(json.load(open(R+'data/v3/dev_ids.json')))
def load(p):
    d=np.load(R+p,allow_pickle=True); return list(d['ids']), list(d['classes']), d['p']
ids,classes,_=load('runs/pod/text32/validation_tx_dev_epoch_1.npz'); ci={c:i for i,c in enumerate(classes)}
def aligned(p):
    i2,c2,P=load(p); ix=[i2.index(t) for t in ids]; cx=[c2.index(c) for c in classes]; return P[ix][:,cx]
gold=np.array([ci[train[t]['root_code']] for t in ids]); qid=[train[t]['question_id'] for t in ids]
def rr(L):
    top=np.argsort(-L,1)[:,:3]; return np.array([(1/(list(top[i]).index(g)+1)) if g in top[i] else 0 for i,g in enumerate(gold)])
def mix(path):
    m=np.full(len(classes),.5)
    for t,r in train.items():
        if t not in dev: m[ci[r['root_code']]]+=1
    if path:
        for l in open(R+path):
            r=json.loads(l)
            for c,s in r['soft'].items():
                if c in ci: m[ci[c]]+=r['weight']*s
    return m/m.sum()
pg=mix(None)
L=lambda P: np.log(np.clip(P,1e-9,1))
d32=aligned('runs/pod/text32/validation_tx_dev_epoch_1.npz'); dd=aligned('runs/pod/dev-distill/validation_epoch_2.npz'); P8=aligned('runs/pod/tx/text_tx_dev_e3.npz')
qs=sorted(set(qid)); random.Random(0).shuffle(qs); f=np.array([{q:i%5 for i,q in enumerate(qs)}[q] for q in qid])
d=collections.defaultdict(list)
for i,q in enumerate(qid): d[q].append(i)
keys=list(d)
T=[0,.25,.5,.75,1.0]
res={}
for mixname in ['data/v3/claude_unl.jsonl','data/v3/claude_unl_v4.jsonl']:
    s=np.log(pg/mix(mixname))
    blend=lambda t: .35*(L(d32)+t*s)+.3*L(P8)+.35*(L(dd)+t*s)
    grid={str(t):round(float(rr(blend(t)).mean()),4) for t in T}
    cf=np.zeros(len(ids)); ch=[]
    for k in range(5):
        m=f==k; b=max(T,key=lambda t: rr(blend(t))[~m].mean()); ch.append(b); cf[m]=rr(blend(b))[m]
    base=rr(blend(0)); rng=np.random.default_rng(0); o=[]
    for _ in range(2000):
        ss=np.concatenate([d[kk] for kk in rng.choice(keys,len(keys))]); o.append(cf[ss].mean()-base[ss].mean())
    mem={n:{str(t):round(float(rr(L(P)+t*s).mean()),4) for t in T} for n,P in [('text32',d32),('distill',dd)]}
    res[mixname]={'members':mem,'ensemble_grid':grid,'crossfit':round(float(cf.mean()),4),'chosen':ch,'delta_ci95':[round(float(np.percentile(o,2.5)),4),round(float(np.percentile(o,97.5)),4)]}
print(json.dumps(res,indent=1)); json.dump(res,open(R+'runs/research/q5-factcheck/ensemble_shift_check2.json','w'),indent=1)
