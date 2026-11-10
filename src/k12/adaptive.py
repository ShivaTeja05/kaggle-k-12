"""Small, testable adaptations of the trained image classifier.

The optional GPU stage caches the frozen multimodal representation once.
The small classifiers then train on CPU without validation-label gradients.
This is an output adapter, not AdaLoRA or a new transformer block.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import read_csv, make_folds, label_space, map3, save_bundle, fingerprint, write_submission


class AdaptiveDiagnosis(nn.Module):
    def __init__(self, weight, bias, kind='residual', rank=32):
        super().__init__()
        if kind not in {'residual', 'linear', 'conservative'}:
            raise ValueError('Unknown adaptation')
        classes, width = weight.shape
        self.kind = kind
        self.rank = min(rank,8) if kind=='conservative' else rank
        self.root = nn.Linear(width, classes)
        with torch.no_grad():
            self.root.weight.copy_(weight)
            self.root.bias.copy_(bias)
        self.reference_weight = weight.detach().clone()
        self.reference_bias = bias.detach().clone()
        if kind != 'linear':
            self.root.requires_grad_(False)
            self.norm = nn.LayerNorm(width, elementwise_affine=False)
            self.down = nn.Linear(width, self.rank, bias=False)
            self.up = nn.Linear(self.rank, width, bias=False)
            self.dropout = nn.Dropout(.2 if kind=='conservative' else .1)
            nn.init.zeros_(self.up.weight)
        self.error = nn.Linear(width, 1)
        nn.init.zeros_(self.error.weight)
        nn.init.zeros_(self.error.bias)

    def forward(self, hidden):
        delta = torch.zeros_like(hidden)
        if self.kind != 'linear':
            delta = self.up(self.dropout(F.gelu(self.down(self.norm(hidden)))))
        adapted = hidden + delta
        logits=self.root(adapted)
        if self.kind=='conservative':
            base=F.linear(hidden,self.reference_weight,self.reference_bias)
            logits=base+.5*torch.tanh((logits-base)/.5)
        return logits, self.error(adapted).squeeze(-1), delta

    def anchoring_loss(self, delta, hidden):
        if self.kind != 'linear':
            return delta.square().mean() / hidden.square().mean().clamp_min(1e-6)
        return ((self.root.weight-self.reference_weight).square().mean()
                + (self.root.bias-self.reference_bias).square().mean())


def class_weights(labels, classes, beta=.99, cap=3.):
    """Tempered effective-number weights; empirical MAP@3 still decides promotion."""
    counts = torch.bincount(labels, minlength=classes).float()
    effective = (1-beta**counts.clamp_min(1))/(1-beta)
    weights = effective.rsqrt()
    weights /= weights[counts > 0].mean()
    weights = weights.clamp(max=cap)
    weights[counts == 0] = 0
    return weights


def fit_head(hidden, labels, weight, bias, none_index, kind, steps=160, seed=2026):
    """No validation inputs or labels are accepted by this training function."""
    torch.manual_seed(seed)
    model = AdaptiveDiagnosis(weight, bias, kind)
    weights = None if kind=='conservative' else class_weights(labels, weight.shape[0])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr={'residual':1e-3,'linear':1e-4,'conservative':5e-4}[kind],
                                  weight_decay=.02)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(min(steps,40) if kind=='conservative' else steps):
        ix = torch.randint(len(hidden), (min(64, len(hidden)),), generator=generator)
        x, y = hidden[ix], labels[ix]
        logits, error, delta = model(x)
        ce = F.cross_entropy(logits, y, weight=weights, label_smoothing=.02)
        # Major-error presence differs from final-answer correctness.
        aux = F.binary_cross_entropy_with_logits(error, (y != none_index).float())
        anchor_weight=1. if kind=='conservative' else .1
        loss = ce + .1*aux + anchor_weight*model.anchoring_loss(delta, x)
        if kind=='conservative':
            teacher=F.linear(x,weight,bias).softmax(-1)
            loss=loss+F.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    return model.eval()


def extract(args):
    """Cache fitting/validation features from the already trained checkpoint."""
    from .vision import VisionSystem, find_model, find_data, messages, seed_all
    if not torch.cuda.is_available():
        raise RuntimeError('Use the private free-GPU extraction notebook')
    seed_all(args.seed)
    data = find_data('/kaggle/input')
    candidates = [p for p in Path('/kaggle/input').rglob('metadata.json')
                  if p.parent.name == 'epoch-3' and (p.parent/'heads.pt').exists()]
    if len(candidates) != 1:
        raise ValueError('Attach exactly one completed k12-vision-development output')
    checkpoint = candidates[0].parent
    metadata = json.loads(candidates[0].read_text())
    train = read_csv(data/'train.csv')
    classes = label_space(read_csv(data/'taxonomy_v1.csv'))
    if metadata['classes'] != classes or metadata['train_sha256'] != fingerprint(data/'train.csv'):
        raise ValueError('Checkpoint data/classes differ')
    options = argparse.Namespace(**metadata['args'])
    folds = make_folds(train, seed=options.split_seed)
    fitting = train.loc[folds != options.fold].reset_index(drop=True)
    validation = train.loc[folds == options.fold].reset_index(drop=True)
    if set(metadata['trained_ids']) != set(fitting.trace_id):
        raise ValueError('Checkpoint was trained on a different partition')
    model_path = find_model('/kaggle/input')
    if fingerprint(model_path/'config.json') != metadata['model_config_sha256']:
        raise ValueError('Base model configuration changed')
    system = VisionSystem(model_path, classes, options, str(checkpoint))
    system.model.eval().requires_grad_(False)
    system.head.eval().requires_grad_(False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = {'checkpoint_metadata_sha256':fingerprint(candidates[0]),
                'base_config_sha256':fingerprint(find_model('/kaggle/input')/'config.json'),
                'train_sha256':fingerprint(data/'train.csv'),
                'source_sha256':fingerprint(Path(__file__)), 'classes':classes,
                'fold':options.fold, 'split_seed':options.split_seed,
                'checkpoint_trained_ids':metadata['trained_ids'],
                'fitting_ids':fitting.trace_id.tolist(), 'validation_ids':validation.trace_id.tolist(),
                'test_extracted':args.test_only}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(out/'base_head.npz',weight=system.head.weight.detach().cpu().numpy(),
                        bias=system.head.bias.detach().cpu().numpy(), classes=np.asarray(classes))
    partitions = [('validation',validation,'train'), ('fitting',fitting,'train')]
    if args.test_only:
        partitions = [('test',read_csv(data/'test.csv'),'test')]
    elif args.fit_only:
        partitions = [('fitting',fitting,'train')]
    started = time.monotonic()
    checked_batch = False
    for name, frame, split in partitions:
        features = []; logits = []; offset = 0; batch_size = args.batch_size
        rows = list(frame.itertuples(index=False))
        while offset < len(rows):
            chunk = rows[offset:offset+batch_size]
            conversations = [messages(r,data,split) for r in chunk]
            try:
                inputs = system.processor.apply_chat_template(conversations,tokenize=True,
                    add_generation_prompt=True,padding=True,return_dict=True,return_tensors='pt')
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                    hidden = system.hidden(inputs)[:,-1].float()
                    scores = system.head(hidden).float()
                if not checked_batch:
                    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                        single = system.head(system.hidden(system.encode(chunk[0],data,split))[:,-1].float()).float()
                    difference = float((scores[0].softmax(-1)-single[0].softmax(-1)).abs().max())
                    if difference > .01:
                        raise ValueError(f'Batched probabilities differ from single image: {difference}')
                    print(json.dumps({'stage':'batch_contract','max_probability_difference':difference}),flush=True)
                    checked_batch = True
                features.append(hidden.cpu().numpy()); logits.append(scores.cpu().numpy())
                offset += len(chunk)
                del inputs,hidden,scores
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:raise
                batch_size = max(1,batch_size//2)
                import gc
                gc.collect(); torch.cuda.empty_cache()
                print(json.dumps({'stage':'reduce_batch','batch_size':batch_size}),flush=True)
                continue
            if offset % 32 == 0 or offset == len(rows):
                print(json.dumps({'stage':'inference','partition':name,'rows':offset,'total':len(rows),
                                  'seconds':round(time.monotonic()-started)}),flush=True)
            if offset % 64 == 0:
                # Preserve completed work if a platform timeout interrupts a pass.
                np.savez_compressed(out/(name+'_partial_features.npz'),
                    ids=np.asarray(frame.trace_id.iloc[:offset],dtype=str),
                    hidden=np.concatenate(features).astype('float32'),
                    logits=np.concatenate(logits).astype('float32'))
        h = np.concatenate(features); raw = np.concatenate(logits)
        np.savez_compressed(out/(name+'_features.npz'),ids=np.asarray(frame.trace_id,dtype=str),
                            hidden=h.astype('float32'),logits=raw.astype('float32'))
        if name == 'validation':
            probabilities = torch.tensor(raw).softmax(-1).numpy()
            metric = map3(frame.root_code, probabilities, classes)
            reference = json.loads((checkpoint.parent/'metrics_epoch_3.json').read_text())['map3']
            if abs(metric-reference) > 1/len(validation):
                raise ValueError(f'Frozen feature baseline changed: {metric} vs {reference}')
            print(json.dumps({'stage':'frozen_baseline','map3':metric,'reference':reference}),flush=True)


def train_cached(args):
    cache = Path(args.features); out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((cache/'manifest.json').read_text())
    train = read_csv(Path(args.data)/'train.csv')
    if fingerprint(Path(args.data)/'train.csv') != manifest['train_sha256']:
        raise ValueError('Feature data fingerprint differs')
    classes = manifest['classes']; ci = {c:i for i,c in enumerate(classes)}
    folds = make_folds(train, seed=manifest['split_seed'])
    fitting_ids = set(train.loc[folds != manifest['fold'],'trace_id'])
    validation_ids = set(train.loc[folds == manifest['fold'],'trace_id'])
    if fitting_ids != set(manifest['checkpoint_trained_ids']) or fitting_ids & validation_ids:
        raise ValueError('Leaking or incompatible checkpoint partition')
    with np.load(cache/'fitting_features.npz',allow_pickle=False) as b:
        fit_ids = b['ids'].tolist(); hidden = torch.tensor(b['hidden'].astype('float32'))
    if set(fit_ids) != fitting_ids or len(set(fit_ids)) != len(fit_ids):
        raise ValueError('Fitting features do not match the fitting fold')
    labels = torch.tensor([ci[c] for c in train.set_index('trace_id').loc[fit_ids,'root_code']])
    with np.load(cache/'base_head.npz',allow_pickle=False) as b:
        if b['classes'].tolist() != classes:raise ValueError('Base head class mismatch')
        weight = torch.tensor(b['weight']); bias = torch.tensor(b['bias'])
    # Fit each fixed recipe before accessing validation in this run. The
    # conservative recipe is an exploratory follow-up to the first overfit run.
    torch.set_num_threads(2)
    models = {kind:fit_head(hidden,labels,weight,bias,ci['NONE'],kind,args.steps,args.seed)
              for kind in ['linear','residual','conservative']}
    with np.load(cache/'validation_features.npz',allow_pickle=False) as b:
        val_ids = b['ids'].tolist(); val = torch.tensor(b['hidden'].astype('float32'))
        base_logits = torch.tensor(b['logits'])
    if set(val_ids) != validation_ids or len(set(val_ids)) != len(val_ids):
        raise ValueError('Incomplete or incompatible validation features')
    y = train.set_index('trace_id').loc[val_ids,'root_code']
    cpu_baseline = F.linear(val,weight,bias).softmax(-1).numpy()
    results = {'baseline_map3':map3(y,cpu_baseline,classes),
               'gpu_baseline_map3':map3(y,base_logits.softmax(-1).numpy(),classes),'rows':len(y),
               'baseline_fitting_map3':map3([classes[i] for i in labels.tolist()],F.linear(hidden,weight,bias).softmax(-1).numpy(),classes),
               'fold':manifest['fold'],'steps':args.steps,'seed':args.seed,'results':{},
               'interpretation':'Development fold reused in previous epoch/prompt selection; no untouched audit claim.',
               'design':'Output bottleneck adapter on frozen trained Qwen features. Not AdaLoRA or transformer-layer fine-tuning.',
               'training_source_sha256':fingerprint(Path(__file__)),
               'feature_manifest_sha256':fingerprint(cache/'manifest.json')}
    for kind, model in models.items():
        with torch.inference_mode():
            probabilities = model(val)[0].softmax(-1).numpy()
            fitting_probabilities = model(hidden)[0].softmax(-1).numpy()
        score = map3(y,probabilities,classes)
        results['results'][kind] = {'map3':score,'delta_vs_baseline':score-results['baseline_map3'],
                                    'training_steps':min(args.steps,40) if kind=='conservative' else args.steps,
                                    'fitting_map3':map3([classes[i] for i in labels.tolist()],fitting_probabilities,classes),
                                    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        save_bundle(out/(kind+'_validation.npz'),val_ids,classes,probabilities,
                    kind='heldout_adaptation',feature_manifest_sha256=results['feature_manifest_sha256'])
        torch.save({'state_dict':deepcopy(model.state_dict()),'kind':kind,'rank':model.rank,
                    'classes':classes,'seed':args.seed,'steps':results['results'][kind]['training_steps'],
                    'fitting_ids':fit_ids,'feature_manifest_sha256':results['feature_manifest_sha256'],
                    'checkpoint_metadata_sha256':manifest['checkpoint_metadata_sha256'],
                    'base_config_sha256':manifest['base_config_sha256'],
                    'train_sha256':manifest['train_sha256']},out/(kind+'.pt'))
    results['adapter_better_than_linear'] = max(results['results'][k]['map3'] for k in ['residual','conservative']) > results['results']['linear']['map3']
    results['candidate_kind'] = max(results['results'],key=lambda k:results['results'][k]['map3'])
    results['promote_to_test'] = results['results'][results['candidate_kind']]['map3'] >= results['baseline_map3']+.01
    (out/'metrics.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


def predict_cached(args):
    """CPU inference after an independently budgeted test-feature extraction."""
    if not args.checkpoint:raise ValueError('Supply a trained adaptation checkpoint')
    cache=Path(args.features);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((cache/'manifest.json').read_text())
    stored=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    for key in ['checkpoint_metadata_sha256','base_config_sha256','train_sha256','classes']:
        if stored[key] != manifest[key]:raise ValueError('Test feature checkpoint differs: '+key)
    with np.load(cache/'base_head.npz',allow_pickle=False) as b:
        if b['classes'].tolist() != stored['classes']:raise ValueError('Test head label order differs')
        model=AdaptiveDiagnosis(torch.tensor(b['weight']),torch.tensor(b['bias']),stored['kind'],stored['rank'])
    model.load_state_dict(stored['state_dict']);model.eval()
    with np.load(cache/'test_features.npz',allow_pickle=False) as b:
        ids=b['ids'].tolist();hidden=torch.tensor(b['hidden'].astype('float32'))
    test=read_csv(Path(args.data)/'test.csv')
    if len(ids)!=len(test) or len(set(ids))!=len(ids) or set(ids)!=set(test.trace_id):
        raise ValueError('Test feature IDs are incomplete or duplicated')
    with torch.inference_mode():probabilities=model(hidden)[0].softmax(-1).numpy()
    save_bundle(out/'test.npz',ids,stored['classes'],probabilities,kind='adapted_test_predictions',
                adaptation_sha256=fingerprint(args.checkpoint))
    write_submission(test.set_index('trace_id').loc[ids].reset_index(),probabilities,stored['classes'],out/'submission.csv')
    print(json.dumps({'rows':len(ids),'kind':stored['kind'],'output':str(out/'submission.csv')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['extract','train','predict'])
    parser.add_argument('--features',default='runs/adaptive-features/adaptive_features')
    parser.add_argument('--output',default='artifacts/adaptive')
    parser.add_argument('--data',default='data/raw');parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--steps',type=int,default=160);parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--fit-only',action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.test_only and args.fit_only:raise ValueError('Choose a single extraction partition')
    if args.stage == 'extract':extract(args)
    elif args.stage == 'train':train_cached(args)
    else:predict_cached(args)
