import json, glob, os, sys
from tokenizers import Tokenizer
t=Tokenizer.from_file(sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/*/tokenizer.json')))[0])
enc=lambda s: t.encode(s, add_special_tokens=False).ids
seed=enc('\nROOT_LINE:')
rows=[json.loads(x) for x in open('data/v5/cand_train.jsonl')]
over=0; ns=[]; notprefix=0
for r in rows:
    k=r['target'].rindex('ANSWER:')+7
    head=enc('\n'+r['target'][:k])
    if head[:len(seed)]!=seed: notprefix+=1
    gen=len(head)-len(seed)  # tokens the model must generate to emit ANSWER:
    ns.append(gen); over+= gen>160
import numpy as np
print('seed ids',seed,'not prefix',notprefix)
print('generated tokens needed to reach ANSWER: p50/p95/p99/max', [int(np.percentile(ns,q)) for q in (50,95,99,100)], 'rows >160:', over, 'of', len(rows))
for src in ('claude','gold'):
    v=[n for n,r in zip(ns,rows) if r['source']==src]; print(src, 'rows>160', sum(x>160 for x in v), 'of', len(v))
# joint vs split tokenization of "ANSWER: C"
print(enc('ANSWER: C'), enc('ANSWER:'), enc(' C'))
print(enc('x.\nROOT_LINE: L1'), enc('x.')+enc('\nROOT_LINE: L1'))
