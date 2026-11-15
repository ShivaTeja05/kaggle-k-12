"""Fact-check: does tau label-shift correction survive in the actual 3-way dev ensemble (0.7182)?"""
import json, csv, random, collections, glob
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
pg=np.full(len(classes),.5)
for t,r in train.items():
    if t not in dev: pg[ci[r['root_code']]]+=1
pg/=pg.sum()
pm=np.full(len(classes),.5)
for t,r in train.items():
    if t not in dev: pm[ci[r['root_code']]]+=1
for l in open(R+'data/v3/claude_unl_v4.jsonl'):
    r=json.loads(l)
    for c,s in r['soft'].items():
        if c in ci: pm[ci[c]]+=r['weight']*s
pm/=pm.sum()
shift=np.log(pg/pm)
L=lambda P: np.log(np.clip(P,1e-9,1))
d32=aligned('runs/pod/text32/validation_tx_dev_epoch_1.npz'); dd=aligned('runs/pod/dev-distill/validation_epoch_2.npz')
cands={p.replace(R,''):aligned(p.replace(R,'')) for p in ['runs/pod/tx/text_tx_dev_e2.npz','runs/pod/tx/text_tx_dev_e3.npz','runs/pod/v3_e2.npz','runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz','runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz','runs/pod/dev-v3/validation_epoch_3.npz']}
out={}
for name,P8 in cands.items():
    base=.35*L(d32)+.3*L(P8)+.35*L(dd)
    out[name]={'member':round(float(rr(L(P8)).mean()),4),'blend':round(float(rr(base).mean()),4)}
print(json.dumps(out,indent=1))
# choose the text8 dev candidate matching 0.7182
name=min(out,key=lambda k:abs(out[k]['blend']-.7182)); P8=cands[name]
qs=sorted(set(qid)); random.Random(0).shuffle(qs); f=np.array([{q:i%5 for i,q in enumerate(qs)}[q] for q in qid])
T=[0,.25,.5,.75,1.0,1.25]
def blend(t): return .35*(L(d32)+t*shift)+.3*(L(P8)+t*shift)+.35*(L(dd)+t*shift)
grid={str(t):round(float(rr(blend(t)).mean()),4) for t in T}
cf=np.zeros(len(ids)); ch=[]
for k in range(5):
    m=f==k; b=max(T,key=lambda t: rr(blend(t))[~m].mean()); ch.append(b); cf[m]=rr(blend(b))[m]
d=collections.defaultdict(list)
for i,q in enumerate(qid): d[q].append(i)
keys=list(d); rng=np.random.default_rng(0); o=[]; base=rr(blend(0))
for _ in range(2000):
    s=np.concatenate([d[kk] for kk in rng.choice(keys,len(keys))]); o.append(cf[s].mean()-base[s].mean())
res={'text8_dev_used':name,'ensemble_base':round(float(base.mean()),4),'tau_grid':grid,'crossfit':round(float(cf.mean()),4),'chosen':ch,
 'delta_ci95':[round(float(np.percentile(o,2.5)),4),round(float(np.percentile(o,97.5)),4)],
 'fixed_tau_0.75':round(float(rr(blend(.75)).mean()),4)}
print(json.dumps(res,indent=1)); json.dump({'members':out,'ensemble':res},open(R+'runs/research/q5-factcheck/ensemble_shift_check.json','w'),indent=1)
