"""CPU probe: post-hoc label-shift correction P*(p_gold/p_trainmix)^tau for models trained with Claude pseudo rows.
Pseudo rows come from the natural unlabelled distribution; dev/test are stratified like train."""
import json, collections, csv, random
import numpy as np
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train={r['trace_id']:r for r in csv.DictReader(open(R+'data/raw/train.csv'))}
dev=set(json.load(open(R+'data/v3/dev_ids.json')))
def load(p):
    d=np.load(R+p,allow_pickle=True); return list(d['ids']), list(d['classes']), d['p']
ids,classes,_=load('runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'); ci={c:i for i,c in enumerate(classes)}
gold=np.array([ci[train[t]['root_code']] for t in ids]); qid=[train[t]['question_id'] for t in ids]
def rr(P):
    top=np.argsort(-P,1)[:,:3]; return np.array([(1/(list(top[i]).index(g)+1)) if g in top[i] else 0 for i,g in enumerate(gold)])
pg=np.full(len(classes),.5)
for t,r in train.items():
    if t not in dev: pg[ci[r['root_code']]]+=1
pg/=pg.sum()
def mix(path):
    m=pg*1199/ pg.sum()*0+0; m=np.full(len(classes),.5)
    for t,r in train.items():
        if t not in dev: m[ci[r['root_code']]]+=1
    for l in open(R+path):
        r=json.loads(l)
        c=r.get('root_code') or r.get('code') or (r.get('codes') or [None])[0]
        if c in ci: m[ci[c]]+=r.get('weight',1.)
    return m/m.sum()
import itertools
first=json.loads(open(R+'data/v3/claude_unl_v4.jsonl').readline()); print('pseudo row keys',list(first.keys())[:20])
def mix(path):
    m=np.full(len(classes),.5)
    for t,r in train.items():
        if t not in dev: m[ci[r['root_code']]]+=1
    for l in open(R+path):
        r=json.loads(l)
        for c,s in r['soft'].items():
            if c in ci: m[ci[c]]+=r['weight']*s
    return m/m.sum()
pm=mix('data/v3/claude_unl_v4.jsonl')
qs=sorted(set(qid)); random.Random(0).shuffle(qs); f=np.array([{q:i%5 for i,q in enumerate(qs)}[q] for q in qid])
T=[0,.25,.5,.75,1.0,1.25,1.5]
def boot(ra,rb):
    d=collections.defaultdict(list)
    for i,q in enumerate(qid): d[q].append(i)
    keys=list(d); rng=np.random.default_rng(0); o=[]
    for _ in range(2000):
        s=np.concatenate([d[k] for k in rng.choice(keys,len(keys))]); o.append(rb[s].mean()-ra[s].mean())
    return [round(float(np.percentile(o,2.5)),4),round(float(np.percentile(o,97.5)),4)]
out={'top_mix_vs_gold_ratio':sorted([(classes[i],round(float(pm[i]/pg[i]),2)) for i in range(len(classes))],key=lambda x:-abs(np.log(x[1])))[:10]}
for name,p in {'8B_scale100_tx':'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz','8B_scale100_gold':'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz','8B_scale25_tx':'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz','8B_scale0_tx(control,no pseudo)':'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz'}.items():
    i2,c2,P=load(p); assert i2==ids
    adj=lambda tau: np.log(np.clip(P,1e-9,1))+tau*np.log(pg/pm)
    grid={str(t):round(float(rr(adj(t)).mean()),4) for t in T}
    cf=np.zeros(len(ids)); chosen=[]
    for k in range(5):
        m=f==k; b=max(T,key=lambda t: rr(adj(t))[~m].mean()); chosen.append(b); cf[m]=rr(adj(b))[m]
    out[name]={'base':round(float(rr(np.log(P+1e-12)).mean()),4),'grid':grid,'question_crossfit':round(float(cf.mean()),4),'chosen_tau':chosen,'delta_ci95':boot(rr(np.log(P+1e-12)),cf)}
print(json.dumps(out,indent=1)); json.dump(out,open(R+'runs/research/q5-datasets/label_shift_correction_probe.json','w'),indent=1)
