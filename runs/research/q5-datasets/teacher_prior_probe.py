"""CPU probe: does a per-question code prior built from 26k Claude teacher labels on unlabelled photos
add anything on top of the trained text reasoners (dev fold 0, 301 rows)? Read-only on project data."""
import json, ast, collections, csv, math, random
import numpy as np
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train={r['trace_id']:r for r in csv.DictReader(open(R+'data/raw/train.csv'))}
dev=json.load(open(R+'data/v3/dev_ids.json'))
unl={r['trace_id']:r['question_id'] for r in csv.DictReader(open(R+'data/semisupervised/competition_data/unlabelled.csv'))}
def load(p):
    d=np.load(R+p,allow_pickle=True); return list(d['ids']), list(d['classes']), d['p']
ids,classes,_=load('runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'); ci={c:i for i,c in enumerate(classes)}
gold=np.array([ci[train[t]['root_code']] for t in ids]); qid=[train[t]['question_id'] for t in ids]
def map3(P):
    top=np.argsort(-P,1)[:,:3]; return float(np.mean([(1/(list(top[i]).index(g)+1)) if g in top[i] else 0 for i,g in enumerate(gold)]))
# teacher prior
tq=collections.defaultdict(lambda: np.zeros(len(classes)))
for l in open(R+'runs/claude/unl_all_v4.jsonl'):
    r=json.loads(l); a=r['answer']; a=a if isinstance(a,dict) else (json.loads(a) if a.strip().startswith('{"') else ast.literal_eval(a))
    q=unl.get(r['trace_id']);
    if q is None: continue
    for w,k in ((.7,'first_code'),(.2,'second_code'),(.1,'third_code')):
        if a.get(k) in ci: tq[q][ci[a[k]]]+=w
# train-only prior (non-dev rows)
trq=collections.defaultdict(lambda: np.zeros(len(classes))); glob=np.zeros(len(classes))
devs=set(dev)
for t,r in train.items():
    if t in devs: continue
    trq[r['question_id']][ci[r['root_code']]]+=1; glob[ci[r['root_code']]]+=1
glob=(glob+.5)/(glob+.5).sum()
def prior(table,alpha=2.):
    P=np.stack([(table[q]+alpha*glob)/(table[q].sum()+alpha) if q in table else glob for q in qid]); return P
Pt=prior(tq); Ptr=prior(trq); Pc=prior({q:tq[q]+3*trq[q] for q in set(tq)|set(trq)})
cov=np.mean([q in tq for q in qid]); n=np.mean([tq[q].sum() for q in qid if q in tq])
out={'dev_rows':len(ids),'dev_questions_covered_by_teacher_prior':round(float(cov),3),'mean_teacher_labels_per_covered_question':round(float(n),1),
     'prior_only_map3':{'teacher_unlabelled':map3(Pt),'train_only':map3(Ptr),'combined':map3(Pc)}}
models={'8B_scale100_tx':'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz','8B_scale0_tx':'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz','32B_tx':'runs/pod/text32/validation_tx_dev_epoch_1.npz'}
qs=sorted(set(qid)); random.Random(0).shuffle(qs); fold={q:i%5 for i,q in enumerate(qs)}; f=np.array([fold[q] for q in qid])
W=[0,.05,.1,.2,.3,.5,.75,1.0]
def blend(P,Q,w): return np.log(np.clip(P,1e-9,1))+w*np.log(np.clip(Q,1e-9,1))
def boot(a,b):
    d=collections.defaultdict(list)
    for i,q in enumerate(qid): d[q].append(i)
    keys=list(d); rng=np.random.default_rng(0); out=[]
    def rr(P):
        top=np.argsort(-P,1)[:,:3]; return np.array([(1/(list(top[i]).index(g)+1)) if g in top[i] else 0 for i,g in enumerate(gold)])
    ra,rb=rr(a),rr(b)
    for _ in range(2000):
        s=np.concatenate([d[k] for k in rng.choice(keys,len(keys))]); out.append(rb[s].mean()-ra[s].mean())
    return [round(float(np.percentile(out,2.5)),4),round(float(np.percentile(out,97.5)),4)]
out['blends']={}
for name,p in models.items():
    try: i2,c2,P=load(p)
    except Exception as e: continue
    assert i2==ids and list(c2)==classes
    for pname,Q in (('teacher_prior',Pt),('train_prior',Ptr)):
        grid={w:map3(blend(P,Q,w)) for w in W}
        cf=np.zeros_like(P)
        for k in range(5):
            m=f==k; best=max(W,key=lambda w: (lambda S: np.mean([(1/(list(np.argsort(-S[i])[:3]).index(gold[i])+1)) if gold[i] in np.argsort(-S[i])[:3] else 0 for i in np.where(~m)[0]]))(blend(P,Q,w)))
            cf[m]=blend(P,Q,best)[m]
        out['blends'][f'{name}+{pname}']={'base':map3(P),'grid':{str(w):round(v,4) for w,v in grid.items()},'question_crossfit':round(map3(cf),4),'crossfit_delta_ci95':boot(np.log(np.clip(P,1e-9,1)),cf)}
print(json.dumps(out,indent=1))
json.dump(out,open(R+'runs/research/q5-datasets/teacher_prior_probe.json','w'),indent=1)
