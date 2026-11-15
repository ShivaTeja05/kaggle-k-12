import json, argparse, torch, numpy as np
from k12.candidate_student import Student, read_jsonl, ANSWER, SEED
a=argparse.Namespace(model=None, device='cpu', load_4bit=False, max_length=900)
torch.manual_seed(0)
s=Student(a, rank=8)
# randomize lora B so the adapter is not a no-op
for n,p in s.model.named_parameters():
    if 'lora_B' in n: p.data.normal_(0, .5)
s.model.eval()
rows=read_jsonl('data/v5/cand_train.jsonl', 6)
items=[s.train_item(r) for r in rows]
# logits at the training answer position vs letter_probs on prompt+'\n'+target[:ANSWER:] (predict path)
with torch.no_grad():
    for it, r in zip(items, rows):
        ids=torch.tensor([it['ids']]); h=s.hidden(ids, torch.ones_like(ids))
        lt=s.lm_head(h[0, it['answer']]).float()[it['letters']].softmax(-1).numpy()
        k=r['target'].rindex(ANSWER)+len(ANSWER)
        seq=it['ids'][:it['start']]+s.encode('\n'+r['target'][:k])
        lp=s.letter_probs([seq],[r['letters']])[0]
        print('max abs diff train-pos vs predict-pos', float(np.abs(lt-lp).max()), 'next token is letter', it['ids'][it['answer']+1]==s.letter(r['target'][-1]))
    # predict-path prompt (prompt_ids + seed) is a prefix of the training ids?
    for it, r in zip(items, rows):
        p,_=s.prompt_ids(r['prompt'], 10**6); seed=s.encode(SEED)
        print('predict prefix == train prefix', it['ids'][:len(p)+len(seed)]==p+seed)
print('config.use_cache', s.model.config.use_cache)
s.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); s.model.enable_input_require_grads()
print('after gc enable: config.use_cache', s.model.config.use_cache, 'generation_config.use_cache', getattr(s.model.generation_config,'use_cache',None))
