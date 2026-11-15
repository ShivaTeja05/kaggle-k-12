import csv, numpy as np
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
lab={r['trace_id']:((r['root_code'] or '').strip() or 'NONE') for r in csv.DictReader(open(R+'data/raw/train.csv',newline=''))}
for fn in ['runs/pod/text32/validation_tx_dev_epoch_1.npz','runs/pod/tx/text_tx_dev_e2.npz','runs/pod/tx/text_tx_dev_e3.npz']:
    d=np.load(R+fn,allow_pickle=True); cl=[str(c) for c in d['classes']]
    s=[]
    for t,p in zip(d['ids'],d['p']):
        top=[cl[j] for j in np.argsort(-p)[:3]]; y=lab[str(t)]; s.append(1/(top.index(y)+1) if y in top else 0)
    print(fn, len(s), round(float(np.mean(s)),4), d['meta'] if 'meta' in d.files else '')
