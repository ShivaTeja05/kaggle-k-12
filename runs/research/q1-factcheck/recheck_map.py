import csv, numpy as np, random
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
lab={}
with open(R+'data/raw/train.csv',newline='') as f:
    for r in csv.DictReader(f):
        c=(r['root_code'] or '').strip(); lab[r['trace_id']]=c if c else 'NONE'
def sc(fn):
    d=np.load(R+'runs/pod/text32/'+fn,allow_pickle=True); cl=[str(c) for c in d['classes']]
    return {str(t):(1/(1+[cl[j] for j in np.argsort(-p)[:3]].index(lab[str(t)])) if lab[str(t)] in [cl[j] for j in np.argsort(-p)[:3]] else 0.0) for t,p in zip(d['ids'],d['p'])}
g=sc('validation_gold_epoch_1.npz'); m=sc('validation_tx_dev_epoch_1.npz'); ids=sorted(g)
G=np.array([g[i] for i in ids]); M=np.array([m[i] for i in ids]); D=G-M
print('n',len(ids),'gold',G.mean().round(4),'machine',M.mean().round(4),'gap',D.mean().round(4))
print('better gold',(D>0).sum(),'better machine',(D<0).sum(), 'NONE rows',sum(lab[i]=='NONE' for i in ids))
rng=np.random.default_rng(0); bs=[D[rng.integers(0,len(D),len(D))].mean() for _ in range(5000)]
print('CI',np.percentile(bs,[2.5,97.5]).round(4))
nn=np.array([lab[i]!='NONE' for i in ids]); print('root rows gold',G[nn].mean().round(4),'machine',M[nn].mean().round(4),'NONE rows gold',G[~nn].mean().round(4),'machine',M[~nn].mean().round(4))
print('paired SE of gap',(D.std(ddof=1)/np.sqrt(len(D))).round(4))
print('share of total gap from NONE rows',((G[~nn]-M[~nn]).sum()/D.sum()).round(3))
