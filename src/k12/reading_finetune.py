"""Resume internal LoRA with response-only reading/root-explanation supervision."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F

from .core import read_csv,label_space,make_folds,fingerprint,map3,save_bundle,write_submission
from .vision import VisionSystem,find_model,find_data,seed_all,messages
from .baseline import HierarchicalPrior
from .holdout_calibration import calibrated


def teacher_response(row,index):
    # This text is supervision only. The classifier reads the prefix position.
    if index%2==0:
        detail='READING: '+str(row.transcription_steps)
    else:
        findings=json.loads(row.misconceptions)
        roots=[str(f.get('text','')) for f in findings if f.get('root')]
        detail='CAUSE: '+(' '.join(roots) if roots else 'No major misconception is annotated.')
    return 'ROOT: '+str(row.root_code)+'\n'+detail


def append_teacher(inputs,target_ids):
    length=inputs['input_ids'].shape[1]
    result=dict(inputs)
    result['input_ids']=torch.cat([inputs['input_ids'],target_ids],dim=1)
    for key in ['attention_mask','token_type_ids','mm_token_type_ids']:
        if key in result:
            extension=torch.ones_like(target_ids) if key=='attention_mask' else torch.zeros_like(target_ids)
            result[key]=torch.cat([result[key],extension],dim=1)
    return result,length


def shard_order(order,rank,world):
    if len(order)%world:order=np.concatenate([order,order[:world-len(order)%world]])
    return order[rank::world]


def rank_zero():return not dist.is_initialized() or dist.get_rank()==0


class JointTrainingModel(nn.Module):
    def __init__(self,system,reading_weight):
        super().__init__();self.backbone=system.model;self.head=system.head
        self.system=system;self.reading_weight=reading_weight

    def forward(self,joined,length,encoded,label):
        hidden=self.system.hidden(joined)
        ce=F.cross_entropy(self.head(hidden[:,length-1].float()),label,label_smoothing=.03)
        lm_head=self.backbone.get_base_model().lm_head
        response_hidden=hidden[:,length-1:length+encoded.shape[1]-1]
        reading=ce.new_zeros(())
        for offset in range(0,encoded.shape[1],32):
            logits=lm_head(response_hidden[:,offset:offset+32])
            reading=reading+F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                encoded[:,offset:offset+32].reshape(-1),reduction='sum')/encoded.shape[1]
        return ce+self.reading_weight*reading,ce.detach(),reading.detach()


@torch.inference_mode()
def predict(system,frame,data,split,path,batch_size=2):
    system.model.eval();system.head.eval();scores=[];start=time.monotonic()
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    local=frame.iloc[rank::world];rows=list(local.itertuples(index=False))
    for offset in range(0,len(rows),batch_size):
        group=rows[offset:offset+batch_size]
        inputs=system.processor.apply_chat_template([messages(r,data,split) for r in group],
            tokenize=True,add_generation_prompt=True,padding=True,return_dict=True,return_tensors='pt')
        with torch.autocast('cuda',dtype=torch.float16):
            hidden=system.hidden(inputs)[:,-1].float();p=system.head(hidden).softmax(-1)
        scores.append(p.float().cpu().numpy())
        n=offset+len(group)
        if rank_zero() and (n%50==0 or n==len(rows)):
            print(json.dumps({'stage':'predict','split':split,'rows':min(n*world,len(frame)),
                              'total':len(frame),'seconds':round(time.monotonic()-start)}),flush=True)
    p=np.concatenate(scores)
    elapsed=time.monotonic()-start
    if dist.is_initialized():
        received=[None]*world;dist.all_gather_object(received,(local.trace_id.tolist(),p,elapsed))
        mapping={identifier:prob for ids,probs,_ in received for identifier,prob in zip(ids,probs)}
        p=np.asarray([mapping[i] for i in frame.trace_id]);elapsed=max(x[2] for x in received)
    if rank_zero():save_bundle(path,frame.trace_id.tolist(),system.classes,p,kind='reading_supervised_image_predictions')
    return p,elapsed


def run(args):
    if not torch.cuda.is_available():raise RuntimeError('Use the private free-GPU runner')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local_rank=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local_rank);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=4))
    seed_all(args.seed+rank);started=time.monotonic()
    data=find_data('/kaggle/input');model_path=find_model('/kaggle/input')
    candidates=[p.parent for p in Path('/kaggle/input').rglob('metadata.json')
                if p.parent.name=='epoch-3' and (p.parent/'heads.pt').exists()]
    if len(candidates)!=1:raise ValueError('Attach the original trained checkpoint exactly once')
    checkpoint=candidates[0];metadata=json.loads((checkpoint/'metadata.json').read_text())
    train=read_csv(data/'train.csv');test=read_csv(data/'test.csv');classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    if metadata['classes']!=classes or metadata['train_sha256']!=fingerprint(data/'train.csv'):
        raise ValueError('Data/checkpoint mismatch')
    if metadata['model_config_sha256']!=fingerprint(model_path/'config.json'):raise ValueError('Model configuration changed')
    folds=make_folds(train,seed=metadata['args']['split_seed']);fold=metadata['args']['fold']
    fitting=train.loc[folds!=fold].reset_index(drop=True);validation=train.loc[folds==fold].reset_index(drop=True)
    if set(fitting.trace_id)!=set(metadata['trained_ids']):raise ValueError('Resume partition differs')
    if args.smoke:fitting=fitting.head(16 if world>1 else 8);validation=validation.head(4);test=test.head(4)
    options=argparse.Namespace(**{**metadata['args'],'resume_training':True,'device_index':local_rank})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    system=VisionSystem(model_path,classes,options,str(checkpoint))
    parameters=[p for p in system.model.parameters() if p.requires_grad]
    if not parameters:raise ValueError('Checkpoint adapters are frozen')
    ci={c:i for i,c in enumerate(classes)}
    trained_ids=sorted(set(metadata['trained_ids'])|set(fitting.trace_id))
    manifest={'args':vars(args),'classes':classes,'fold':fold,'trained_ids':trained_ids,
              'fine_tuned_ids':fitting.trace_id.tolist(),
              'validation_ids':validation.trace_id.tolist(),'train_sha256':fingerprint(data/'train.csv'),
              'parent_checkpoint_sha256':fingerprint(checkpoint/'metadata.json'),
              'model_config_sha256':fingerprint(model_path/'config.json'),'source_sha256':fingerprint(Path(__file__)),
              'vision_source_sha256':fingerprint(Path(__file__).with_name('vision.py')),
              'world_size':world,'padded_duplicate_training_rows':(-len(fitting))%world,
              'trainable_lora_parameters':sum(p.numel() for p in parameters)}
    if rank_zero():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Validate causal prefix isolation on the actual Qwen model before learning.
    system.model.eval();system.head.eval()
    sample=fitting.iloc[rank%len(fitting)];prefix=system.encode(sample,data,'train')
    target=system.processor.tokenizer(teacher_response(sample,1),add_special_tokens=False,
        truncation=True,max_length=args.teacher_tokens,return_tensors='pt')['input_ids']
    joined,length=append_teacher(prefix,target)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        plain=system.head(system.hidden(prefix)[:,-1].float()).softmax(-1)
        supervised=system.head(system.hidden(joined)[:,length-1].float()).softmax(-1)
    difference=float((plain-supervised).abs().max())
    if difference>.01:raise ValueError('Teacher response changed classification prefix: '+str(difference))
    if rank_zero():print(json.dumps({'stage':'supervision_contract','max_prefix_probability_difference':difference}),flush=True)
    del plain,supervised,joined,prefix,target;torch.cuda.empty_cache()
    optimizer=torch.optim.AdamW([{'params':parameters,'lr':args.lr},
        {'params':system.head.parameters(),'lr':args.head_lr}],weight_decay=.02)
    scaler=torch.amp.GradScaler('cuda')
    system.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    system.model.enable_input_require_grads()
    system.model.config.use_cache=False;system.model.train();system.head.train();optimizer.zero_grad(set_to_none=True)
    network=JointTrainingModel(system,args.reading_weight)
    if world>1:network=DistributedDataParallel(network,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
    order=shard_order(np.random.default_rng(args.seed+3).permutation(len(fitting)),rank,world)
    accumulation=max(1,8//world);losses=[];updates=math.ceil(len(order)/accumulation)
    for k,i in enumerate(order):
        row=fitting.iloc[i];prefix=system.encode(row,data,'train')
        encoded=system.processor.tokenizer(teacher_response(row,k),add_special_tokens=False,
            truncation=True,max_length=args.teacher_tokens,return_tensors='pt')['input_ids']
        joined,length=append_teacher(prefix,encoded)
        update=(k+1)%accumulation==0 or k+1==len(order)
        sync=network.no_sync() if world>1 and not update else nullcontext()
        with sync:
            with torch.autocast('cuda',dtype=torch.float16):
                loss,ce,reading=network(joined,length,encoded.cuda(),torch.tensor([ci[row.root_code]],device='cuda'))
            window=min(accumulation,len(order)-(k//accumulation)*accumulation)
            scaler.scale(loss/window).backward()
        losses.append([float(ce),float(reading)])
        if update:
            step=k//accumulation;scale=.5*(1+math.cos(math.pi*step/max(1,updates)))
            optimizer.param_groups[0]['lr']=args.lr*scale;optimizer.param_groups[1]['lr']=args.head_lr*scale
            scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(parameters+list(system.head.parameters()),1.)
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        if rank_zero() and ((k+1)%25==0 or k+1==len(order)):
            log={'stage':'train','rows':min((k+1)*world,len(fitting)),'total':len(fitting),'root_loss':float(np.mean(losses[-25:],axis=0)[0]),
                 'reading_loss':float(np.mean(losses[-25:],axis=0)[1]),'seconds':round(time.monotonic()-started),
                 'max_gpu_gb':round(torch.cuda.max_memory_allocated()/1e9,2)}
            print(json.dumps(log),flush=True)
    if rank_zero():system.save(out/'checkpoint',{**metadata,'epoch':metadata['epoch']+1,'trained_ids':trained_ids,'reading_recipe':manifest})
    if world>1:dist.barrier()
    system.model.gradient_checkpointing_disable()
    del optimizer,scaler,network;gc.collect();torch.cuda.empty_cache()
    probabilities,validation_seconds=predict(system,validation,data,'train',out/'validation.npz')
    fixed=calibrated(probabilities,HierarchicalPrior(classes).fit(fitting).predict_proba(validation),.25,2.)
    fixed_score=map3(validation.root_code,fixed,classes)
    metric={'rows':len(validation),'map3':map3(validation.root_code,probabilities,classes),
            'fixed_calibration_map3':fixed_score,'calibration_prior_weight':.25,'calibration_temperature':2.,
            'smoke_only':args.smoke,'fold':fold,'validation_seconds':validation_seconds}
    metric['selected_variant']='fixed_calibration' if fixed_score>metric['map3'] else 'raw'
    metric['selected_map3']=max(fixed_score,metric['map3'])
    if rank_zero():
        save_bundle(out/'validation_calibrated.npz',validation.trace_id.tolist(),classes,fixed,kind='fixed_calibrated_reading_development')
        (out/'metrics.json').write_text(json.dumps(metric,indent=2));print(json.dumps(metric),flush=True)
    if args.smoke:
        predict(system,test,data,'test',out/'smoke_test.npz');return
    if metric['selected_map3']<args.min_map3:
        if rank_zero():print(json.dumps({'stage':'quality_gate_stopped_test','map3':metric['selected_map3'],'required':args.min_map3}),flush=True)
        return
    estimate=validation_seconds/len(validation)*len(test)*1.25+120
    remaining=args.budget_seconds-(time.monotonic()-started)
    admitted=estimate<=remaining
    if world>1:
        decision=[admitted if rank_zero() else None];dist.broadcast_object_list(decision,src=0);admitted=decision[0]
    if not admitted:
        if rank_zero():print(json.dumps({'stage':'runtime_gate_stopped_test','estimated_seconds':estimate,'remaining_seconds':remaining}),flush=True)
        return
    probabilities,_=predict(system,test,data,'test',out/'test.npz')
    if rank_zero():
        write_submission(test,probabilities,classes,out/'submission_raw.csv')
        fixed=calibrated(probabilities,HierarchicalPrior(classes).fit(train).predict_proba(test),.25,2.)
        save_bundle(out/'test_calibrated.npz',test.trace_id.tolist(),classes,fixed,kind='fixed_calibrated_reading_test')
        write_submission(test,fixed if metric['selected_variant']=='fixed_calibration' else probabilities,classes,out/'submission.csv')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/kaggle/working/reading')
    parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--teacher-tokens',type=int,default=128);parser.add_argument('--reading-weight',type=float,default=.15)
    parser.add_argument('--lr',type=float,default=2e-5);parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--min-map3',type=float,default=.615);parser.add_argument('--budget-seconds',type=int,default=6000)
    try:run(parser.parse_args())
    finally:
        if dist.is_initialized():dist.destroy_process_group()
