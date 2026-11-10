"""Offline generative image diagnosis using every taxonomy code.

An experimental complement to the trained classifier, not a scored result.
All demonstrations come from the fitting fold, including their annotations.
"""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from .core import read_csv, label_space, make_folds, save_bundle, map3, write_submission
from .vision import find_data, find_model, messages, seed_all
from .baseline import HierarchicalPrior


def parse_codes(text, classes, fallback):
    # Retain canonical case: the official taxonomy includes Dec-01 and Dec-02.
    lookup={c.upper():c for c in classes}
    text=re.sub(r'<think>.*?</think>','',text,flags=re.S)
    # Evidence may discuss rejected alternatives; only the final decision is ranked.
    final=re.findall(r'(?im)^\s*FINAL\s*:\s*(.*)$',text)
    if final:text=final[-1]
    hits=[]
    for code in re.findall(r'(?<![A-Za-z0-9])(?:[A-Za-z]{2,4}-\d{2}|NONE)(?![A-Za-z0-9])',text,flags=re.I):
        canonical=lookup.get(code.upper())
        if canonical and canonical not in hits:hits.append(canonical)
    for code in fallback:
        if code not in hits:hits.append(code)
    return hits[:3]


def taxonomy_definitions(taxonomy):
    definitions='\n'.join(f"{r.id}: {r.name}. {r.description}" for r in taxonomy.itertuples(index=False))
    definitions += '\nNONE: no major misconception; fully correct or minor non-result-changing slips only.'
    return definitions


def labelled_examples(row,fitting):
    # At most two same-question examples from the fitting fold, with different labels.
    examples=fitting.loc[fitting.question_id==row.question_id].drop_duplicates('root_code').head(2)
    parts=[]
    for r in examples.itertuples(index=False):
        explanation=''
        try:
            findings=json.loads(r.misconceptions)
            roots=[x for x in findings if x.get('root')]
            if roots:explanation=str(roots[0].get('text',''))[:600]
        except (AttributeError,ValueError,TypeError):pass
        parts.append(f"Labelled training example (different student):\n{r.transcription_steps}\nRoot code: {r.root_code}"+
                     (f"\nTeacher's root explanation: {explanation}" if explanation else ''))
    return '\n\n'.join(parts)


def extra_prompt(row, fitting, taxonomy, evidence=False):
    definitions=taxonomy_definitions(taxonomy)
    demonstrations=labelled_examples(row,fitting)
    output_rule=("Write at most 80 words identifying the decisive visible step and root cause. "
                 "Then end with FINAL: CODE1 CODE2 CODE3. Do not rank a downstream consequence above its cause."
                 if evidence else "Output codes only.")
    return ("\nOfficial taxonomy:\n"+definitions+'\n\n'+demonstrations+
            "\n\nThe labelled examples are references, not this student's work. Diagnose the attached student's image. "
            "Do not repair their mistakes or infer a misconception just because steps are omitted. "
            "Inspect the student's active reasoning and distinguish its root cause from downstream errors. "
            "Return exactly three distinct valid codes, most likely first, separated by spaces. "
            "Use NONE when there is no major misconception. "+output_rule)


def run(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    seed_all(args.seed)
    data=Path(args.data) if args.data else find_data('/kaggle/input')
    if args.model:model_path=Path(args.model)
    else:
        candidates=[]
        for config in Path('/kaggle/input').rglob('config.json'):
            try:
                if json.loads(config.read_text()).get('model_type') in {'qwen3_vl','qwen3_5'}:candidates.append(config.parent)
            except (ValueError,OSError):pass
        if len(candidates)!=1:raise ValueError(f'Expected one supported VLM, found {candidates}')
        model_path=candidates[0]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');test=read_csv(data/'test.csv');tax=read_csv(data/'taxonomy_v1.csv');classes=label_space(tax)
    folds=make_folds(train,seed=args.split_seed)
    fitting=train.loc[folds!=args.fold] if not args.test else train
    target=train.loc[folds==args.fold] if not args.test else test
    target=target.reset_index(drop=True)
    if args.limit:target=target.sample(min(args.limit,len(target)),random_state=args.seed).reset_index(drop=True)
    prior=HierarchicalPrior(classes).fit(fitting).predict_proba(target)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,max_pixels=args.max_pixels)
    # Long taxonomy prompts exceed a single T4's memory with the math SDPA backend.
    # Balance weights over Kaggle's two free T4s, retaining image resolution.
    devices=torch.cuda.device_count()
    placement={'device_map':'balanced','max_memory':{i:'12GiB' for i in range(devices)}} if devices>1 else {'device_map':{'':0}}
    model=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,dtype=torch.float16,
                attn_implementation='sdpa',**placement).eval()
    print(json.dumps({'stage':'loaded','devices':devices,'device_map':getattr(model,'hf_device_map',{})}),flush=True)
    path=out/('test.jsonl' if args.test else 'validation.jsonl')
    # A manifest prevents accidentally resuming predictions from a changed experiment.
    from .core import fingerprint
    manifest={**vars(args),'train_sha256':fingerprint(data/'train.csv'),
              'model_config_sha256':fingerprint(model_path/'config.json'),'classes':classes,
              'target_ids':target.trace_id.tolist(),'fitting_ids':fitting.trace_id.tolist()}
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Output directory belongs to a different experiment')
    mp.write_text(json.dumps(manifest,indent=2))
    cached={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);cached[r['trace_id']]=r
    scores=[];ci={c:i for i,c in enumerate(classes)};start=time.monotonic()
    for i,row in enumerate(target.itertuples(index=False)):
        if row.trace_id not in cached:
            msg=messages(row,data,'test' if args.test else 'train')
            msg[0]['content'][-1]['text']+=extra_prompt(row,fitting,tax,args.evidence)
            inputs=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True,enable_thinking=False)
            inputs={k:v.cuda() for k,v in inputs.items() if k!='token_type_ids'}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                ids=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False,logits_to_keep=1)
            text=processor.decode(ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            fallback=np.asarray(classes)[np.argsort(-prior[i])].tolist()
            codes=parse_codes(text,classes,fallback)
            r={'trace_id':row.trace_id,'raw':text,'codes':codes,
               'valid_generated_codes':len(parse_codes(text,classes,[])),
               'hit_generation_limit':ids.shape[1]-inputs['input_ids'].shape[1]>=args.max_new_tokens}
            with path.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            cached[row.trace_id]=r
        # Rank-derived scores, not calibrated model probabilities. Do not mix blindly.
        p=np.ones(len(classes))*1e-8
        for rank,c in enumerate(cached[row.trace_id]['codes']):p[ci[c]]=1/(rank+1)
        scores.append(p)
        if i%10==0:print(json.dumps({'rows':i+1,'total':len(target),'seconds':round(time.monotonic()-start)}),flush=True)
    p=np.asarray(scores)
    save_bundle(out/('test.npz' if args.test else 'validation.npz'),target.trace_id.tolist(),classes,p,kind='rank_scores_not_calibrated')
    if args.test:write_submission(target,p,classes,out/'submission.csv')
    else:
        metric={'map3':map3(target.root_code,p,classes),'rows':len(target),'kind':'held_out_images','fold':args.fold,
                'rows_needing_prior_fill':sum(cached[t].get('valid_generated_codes',len(parse_codes(cached[t]['raw'],classes,[])))<3 for t in target.trace_id)}
        (out/'metrics.json').write_text(json.dumps(metric,indent=2));print(metric)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--model');p.add_argument('--output',default='/kaggle/working/taxonomy_ranker')
    p.add_argument('--test',action='store_true');p.add_argument('--fold',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--split-seed',type=int,default=2026);p.add_argument('--max-pixels',type=int,default=768*768)
    p.add_argument('--evidence',action='store_true');p.add_argument('--max-new-tokens',type=int,default=64)
    run(p.parse_args())
