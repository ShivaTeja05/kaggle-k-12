import json, re, numpy as np
from collections import Counter
from k12.core import read_csv, make_folds
tr=read_csv('data/raw/train.csv'); f=make_folds(tr); dev=tr[f==0]
norm=lambda t: ' '.join(re.sub(r'(?m)^\s*L\d+\s*\|\s?','',str(t)).lower().split())
txd=json.loads(open('runs/pod/tx/tx_dev.json').read())
devkeys={}
for r in dev.itertuples():
    for t in (r.transcription_steps, txd.get(r.trace_id,'')):
        k=(r.question_id, norm(t))
        if len(k[1])>=40: devkeys[k]=r.trace_id
rows=[json.loads(x) for x in open('data/v5/cand_train.jsonl')]
unl=read_csv('data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
hits=[]
for r in rows:
    if r['source']!='claude': continue
    w=r['prompt'].split('Student working (line by line):\n',1)[1].split('\n\nCandidate error codes:')[0]
    k=(unl.at[r['trace_id'],'question_id'], norm(w))
    if k in devkeys: hits.append((r['trace_id'], devkeys[k]))
print('claude train rows whose normalized transcript equals a dev row transcript on same question (>=40 chars):', len(hits))
print('truth NONE share: claude', round(np.mean([r['codes'][ord(r['target'][-1])-65]=='NONE' for r in rows if r['source']=='claude']),3),
      'gold', round(np.mean([r['codes'][ord(r['target'][-1])-65]=='NONE' for r in rows if r['source']=='gold']),3), 'dev', round(float((dev.root_code=='NONE').mean()),3))
