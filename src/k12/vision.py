"""Offline Qwen3-VL LoRA classifier. Identical image/context inputs at train and inference.

Train-only correctness labels supervise a separate auxiliary head, never an input.
All validation predictions use images, never gold transcriptions or annotations.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn

from .core import (read_csv, label_space, make_folds, map3, save_bundle,
                   fingerprint, write_submission)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def find_data(root):
    hits = sorted(p.parent for p in Path(root).rglob('train.csv') if (p.parent/'taxonomy_v1.csv').exists())
    if len(hits) != 1:
        raise ValueError(f'Expected one competition data directory, got {hits}')
    return hits[0]


def find_model(root):
    hits = []
    for p in Path(root).rglob('config.json'):
        try:
            cfg=json.loads(p.read_text())
            if cfg.get('model_type') == 'qwen3_vl': hits.append(p.parent)
        except (ValueError,OSError): pass
    if len(hits) != 1:
        raise ValueError(f'Expected one Qwen3-VL model directory, got {hits}')
    return hits[0]


def image(data, split, filename, maxside=1024):
    # Filenames are resolved within the competition tree; no missing-image fallback.
    candidates = [data/'images'/split/str(filename), data/str(filename)]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None: raise FileNotFoundError(f'Missing {split} image: {filename}')
    with Image.open(path) as im:
        out=ImageOps.exif_transpose(im).convert('RGB')
    out.thumbnail((maxside,maxside));return out


def messages(row, data, split, task='classify', maxside=1024):
    content=[]
    if str(row.question_image_file).strip():
        content += [{'type':'text','text':'Question diagram:'},
                    {'type':'image','image':image(data,'questions',row.question_image_file,maxside)}]
    content += [{'type':'text','text':"Student's handwritten working:"},
                {'type':'image','image':image(data,split,row.image_file,maxside)}]
    instruction = ("Read the student's actual reasoning. Identify the root major error that the rest of the work "
        "follows from, which need not be the first written error. NONE means no major error; minor notation "
        "slips can still be NONE. A matching final answer alone does not establish correct reasoning. "
        "Respect crossed-out corrections and do not invent missing steps.\nROOT:")
    if task == 'transcribe':
        instruction = "Transcribe the visible student working verbatim, line by line in L1 | ... format. Preserve errors; do not solve or correct it."
    elif task == 'rank':
        instruction = "Inspect this student's active working and rank three alternative root diagnoses using the system's taxonomy and output format."
    content.append({'type':'text','text':f"Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\nAnswer key: {row.correct_answer}\n{instruction}"})
    return [{'role':'user','content':content}]


class VisionSystem:
    def __init__(self, model_path, classes, args, checkpoint=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model, PeftModel
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,
                                                    min_pixels=128*128,max_pixels=args.max_pixels)
        self.processor.tokenizer.padding_side='left'
        # Kaggle T4s need FP16; Ampere pods pass --dtype bfloat16.
        self.dtype=getattr(torch,getattr(args,'dtype','float16'));self.maxside=getattr(args,'maxside',1024)
        quantization=None
        if getattr(args,'load_4bit',False):
            # QLoRA: NF4 frozen weights so a 32B model fits a 48 GB card.
            from transformers import BitsAndBytesConfig
            quantization=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=self.dtype)
        self.base=AutoModelForImageTextToText.from_pretrained(model_path,local_files_only=True,
                    dtype=self.dtype,device_map={'':getattr(args,'device_index',0)},attn_implementation='sdpa',
                    quantization_config=quantization)
        # Kaggle's 2026 image has PEFT 0.21 but torchao 0.10. We load ordinary
        # FP16 weights, so disable only this unused, incompatible quantizer dispatch.
        import peft.import_utils as peft_imports
        try:
            peft_imports.is_torchao_available()
        except ImportError as exc:
            if 'torchao' not in str(exc):raise
            if getattr(self.base.config,'quantization_config',None) and not getattr(args,'load_4bit',False):raise
            import peft.tuners.lora.torchao as peft_torchao
            peft_imports.is_torchao_available=lambda:False
            peft_torchao.is_torchao_available=lambda:False
            print('Disabled incompatible optional torchao dispatcher for unquantized FP16 LoRA',flush=True)
        cfg=self.base.config.get_text_config()
        if checkpoint:
            self.model=PeftModel.from_pretrained(self.base,str(Path(checkpoint)/'adapter'),
                                                is_trainable=getattr(args,'resume_training',False))
        else:
            targets=[n for n,m in self.base.named_modules() if isinstance(m,nn.Linear)
                     and 'language_model' in n and n.rsplit('.',1)[-1] in
                     {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}]
            if not targets: raise ValueError('No language-model LoRA targets found')
            if getattr(args,'lora_vision',False):
                # Handwriting reading happens in the vision tower and its mergers.
                vision=[n for n,m in self.base.named_modules() if isinstance(m,nn.Linear)
                        and '.visual.' in f'.{n}.' and n.rsplit('.',1)[-1] in {'qkv','proj','linear_fc1','linear_fc2'}]
                if not vision: raise ValueError('No vision LoRA targets found')
                targets+=vision
            self.model=get_peft_model(self.base,LoraConfig(r=args.rank,lora_alpha=args.rank*2,
                         lora_dropout=.05,target_modules=targets))
        # Seed is set BEFORE constructing either head.
        self.head=nn.Linear(cfg.hidden_size,len(classes),device='cuda',dtype=torch.float32)
        self.correct_head=nn.Linear(cfg.hidden_size,1,device='cuda',dtype=torch.float32)
        if checkpoint:
            h=torch.load(Path(checkpoint)/'heads.pt',map_location='cuda',weights_only=True)
            self.head.load_state_dict(h['root']);self.correct_head.load_state_dict(h['correct'])
        for p in self.model.parameters():
            if p.requires_grad:p.data=p.data.float()
        self.classes=classes

    def encode(self,row,data,split,task='classify'):
        return self.processor.apply_chat_template(messages(row,data,split,task,self.maxside),tokenize=True,
                    add_generation_prompt=True,return_dict=True,return_tensors='pt')

    def hidden(self,inp):
        inp={k:v.cuda() for k,v in inp.items() if k!='token_type_ids'}
        # Calling the multimodal backbone avoids materializing all vocabulary logits.
        return self.model.get_base_model().model(**inp,use_cache=False).last_hidden_state

    @torch.inference_mode()
    def predict(self,df,data,split,path=None):
        self.model.eval();self.head.eval();self.correct_head.eval()
        p=[];correct=[];start=time.monotonic()
        for i,row in enumerate(df.itertuples(index=False)):
            with torch.autocast('cuda',dtype=torch.float16):
                h=self.hidden(self.encode(row,data,split))[:,-1].float()
            p.append(self.head(h).softmax(-1)[0].cpu().numpy())
            correct.append(float(self.correct_head(h).sigmoid().item()))
            if i%50==0:print(json.dumps({'stage':'predict','split':split,'rows':i+1,'total':len(df),'seconds':round(time.monotonic()-start)}),flush=True)
        p=np.asarray(p)
        if path:save_bundle(path,df.trace_id.tolist(),self.classes,p,kind='image_predictions',correctness=correct)
        return p

    def save(self,path,metadata):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path/'adapter',safe_serialization=True)
        self.processor.save_pretrained(path/'processor')
        torch.save({'root':self.head.state_dict(),'correct':self.correct_head.state_dict()},path/'heads.pt')
        (path/'metadata.json').write_text(json.dumps(metadata,indent=2))


def train(args):
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; use the private Kaggle runner')
    seed_all(args.seed)
    data=Path(args.data) if args.data else find_data('/kaggle/input')
    model_path=Path(args.model) if args.model else find_model('/kaggle/input')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=read_csv(data/'train.csv');test=read_csv(data/'test.csv');tax=read_csv(data/'taxonomy_v1.csv')
    classes=label_space(tax);ci={c:i for i,c in enumerate(classes)}
    folds=make_folds(df,seed=args.split_seed)
    pd.DataFrame({'trace_id':df.trace_id,'fold':folds}).to_csv(out/'folds.csv',index=False)
    fit=df.loc[folds!=args.fold].reset_index(drop=True);val=df.loc[folds==args.fold].reset_index(drop=True)
    if args.full_train:fit=df.copy()
    if args.smoke:fit=fit.head(8);val=val.head(4);test=test.head(4);args.epochs=1
    environment={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                 'cuda':torch.version.cuda,'args':vars(args),'model':str(model_path),
                 'model_config_sha256':fingerprint(model_path/'config.json'),
                 'train_sha256':fingerprint(data/'train.csv'),'classes':classes}
    import importlib.metadata
    environment['versions']={p:importlib.metadata.version(p) for p in ['transformers','peft','accelerate','numpy','pandas']}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    print(json.dumps(environment),flush=True)
    system=VisionSystem(model_path,classes,args,checkpoint=args.checkpoint)
    if args.checkpoint:
        stored=json.loads((Path(args.checkpoint)/'metadata.json').read_text())
        if stored['classes']!=classes:raise ValueError('Checkpoint label order mismatch')
    else:
        system.model.gradient_checkpointing_enable();system.model.enable_input_require_grads()
        system.model.config.use_cache=False
        params=[p for p in system.model.parameters() if p.requires_grad]
        heads=list(system.head.parameters())+list(system.correct_head.parameters())
        opt=torch.optim.AdamW([{'params':params,'lr':args.lr},{'params':heads,'lr':args.head_lr}],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda')
        updates=args.epochs*math.ceil(len(fit)/args.accumulation)
        step=0;start=time.monotonic();logs=[]
        for epoch in range(args.epochs):
            system.model.train();system.head.train();system.correct_head.train();opt.zero_grad(set_to_none=True)
            order=np.random.default_rng(args.seed+epoch).permutation(len(fit));losses=[]
            for k,i in enumerate(order):
                row=fit.iloc[i];inp=system.encode(row,data,'train')
                with torch.autocast('cuda',dtype=torch.float16):h=system.hidden(inp)[:,-1].float()
                ce=nn.functional.cross_entropy(system.head(h),torch.tensor([ci[row.root_code]],device='cuda'),label_smoothing=.03)
                target=float(str(row.final_answer_correct).lower() in {'true','1'})
                aux=nn.functional.binary_cross_entropy_with_logits(system.correct_head(h).view(-1),torch.tensor([target],device='cuda'))
                loss=ce+.2*aux
                # Correct normalization of the final partial accumulation window.
                window=min(args.accumulation,len(order)-(k//args.accumulation)*args.accumulation)
                scaler.scale(loss/window).backward();losses.append(float(loss.detach()))
                if (k+1)%args.accumulation==0 or k+1==len(order):
                    factor=min(1.,(step+1)/max(1,.05*updates))*.5*(1+math.cos(math.pi*step/max(1,updates)))
                    opt.param_groups[0]['lr']=args.lr*factor;opt.param_groups[1]['lr']=args.head_lr*factor
                    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params+heads,1.)
                    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);step+=1
                if (k+1)%25==0 or k+1==len(order):
                    log={'stage':'train','epoch':epoch+1,'rows':k+1,'total':len(fit),'loss':float(np.mean(losses[-25:])),
                         'seconds':round(time.monotonic()-start),'max_gpu_gb':round(torch.cuda.max_memory_allocated()/1e9,2)}
                    print(json.dumps(log),flush=True);logs.append(log)
                    (out/'training_log.json').write_text(json.dumps(logs,indent=2))
            system.save(out/f'epoch-{epoch+1}',{**environment,'epoch':epoch+1,'trained_ids':fit.trace_id.tolist()})
            if not args.full_train:
                pv=system.predict(val,data,'train',out/f'validation_epoch_{epoch+1}.npz')
                metric={'epoch':epoch+1,'map3':map3(val.root_code,pv,classes),'rows':len(val),'kind':'held_out_images',
                        'fold':args.fold,'warning':'One development fold; not OOF across the full training set.'}
                print(json.dumps(metric),flush=True)
                (out/f'metrics_epoch_{epoch+1}.json').write_text(json.dumps(metric,indent=2))
        system.model.gradient_checkpointing_disable()
        del opt,scaler;gc.collect();torch.cuda.empty_cache()
    if args.predict_test:
        p=system.predict(test,data,'test',out/'test.npz')
        write_submission(test,p,classes,out/'submission.csv')
    print('K12_RUN_COMPLETE',flush=True)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--data');p.add_argument('--model');p.add_argument('--output',default='/kaggle/working/k12')
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--split-seed',type=int,default=2026)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--max-pixels',type=int,default=768*768)
    p.add_argument('--lr',type=float,default=7e-5);p.add_argument('--head-lr',type=float,default=7e-4)
    p.add_argument('--accumulation',type=int,default=8);p.add_argument('--smoke',action='store_true')
    p.add_argument('--predict-test',action='store_true');p.add_argument('--full-train',action='store_true')
    p.add_argument('--checkpoint');return p


if __name__=='__main__':train(parser().parse_args())
