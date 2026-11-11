"""Fresh LoRA classifier on a single large GPU, with optional teacher-reading supervision.

Built for a rented Ampere pod: BF16, higher image resolution and longer teacher
targets than the Kaggle T4 recipe. Inputs are identical at train and inference;
teacher transcriptions and explanations are response targets only.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .baseline import HierarchicalPrior
from .core import fingerprint, label_space, make_folds, map3, read_csv, save_bundle, write_submission
from .holdout_calibration import calibrated
from .reading_finetune import append_teacher, teacher_response
from .vision import VisionSystem, messages, seed_all


def prefetched(function, items, workers=6, ahead=16):
    """Yield function(item) in order while background threads prepare upcoming items.

    Image decoding and processor work run on CPU threads so the GPU is not left idle.
    """
    with ThreadPoolExecutor(workers) as pool:
        queue = deque(pool.submit(function, item) for item in items[:ahead])
        for item in items[ahead:]:
            yield queue.popleft().result()
            queue.append(pool.submit(function, item))
        while queue:
            yield queue.popleft().result()


def encode_row(system, row, data):
    return system.encode(row, data, 'unlabelled' if isinstance(row.get('soft'), dict) else 'train')


def target(row, ci, secondary_weight):
    """Root code, with a share for other annotated major errors so ranks 2-3 learn near-misses."""
    vector = torch.zeros(len(ci), device='cuda')
    others = []
    if secondary_weight > 0:
        findings = json.loads(str(row.misconceptions).replace('NaN', 'null'))
        others = sorted({f['category'] for f in findings if f.get('severity') == 'major' and not f.get('root')
                         and f.get('category') in ci} - {row.root_code})
    vector[ci[row.root_code]] = 1 - (secondary_weight if others else 0)
    for code in others:
        vector[ci[code]] += secondary_weight / len(others)
    return vector[None]


def oversampled(fit, max_copies):
    """Repeat rows of rare codes, up to max_copies, toward a minimum of max_copies examples per code."""
    counts = fit.root_code.map(fit.root_code.value_counts())
    copies = np.clip(np.ceil(max_copies / counts), 1, max_copies).astype(int)
    return fit.loc[fit.index.repeat(copies)].reset_index(drop=True)


def auxiliary(system, family_head, families, hidden, row):
    """Training-only targets: the root code's error family and whether any major error exists."""
    family = torch.tensor([families[row.root_code.split('-', 1)[0].upper()]], device='cuda')
    error = torch.tensor([float(row.root_code != 'NONE')], device='cuda')
    return (F.cross_entropy(family_head(hidden), family)
            + F.binary_cross_entropy_with_logits(system.correct_head(hidden).view(-1), error))


def loss_terms(system, row, data, ci, index, args, family_head=None, families=None, prefix=None):
    if prefix is None:
        prefix = encode_row(system, row, data)
    if isinstance(row.get('soft'), dict):
        # Pseudo-labelled photo: soft target only; it has no teacher transcription or annotations.
        label = torch.zeros(1, len(ci), device='cuda')
        for code, share in row.soft.items():
            label[0, ci[code]] = share
        hidden = system.hidden(prefix)[:, -1].float()
        return F.cross_entropy(system.head(hidden), label), torch.zeros(()), torch.zeros(())
    label = target(row, ci, args.secondary_weight)
    if args.reading_weight <= 0:
        hidden = system.hidden(prefix)[:, -1].float()
        aux = auxiliary(system, family_head, families, hidden, row) if family_head else torch.zeros(())
        return F.cross_entropy(system.head(hidden), label, label_smoothing=.03), torch.zeros(()), aux
    text = args.v3[row.trace_id]['target'] if getattr(args, 'v3', None) else teacher_response(row, index)
    encoded = system.processor.tokenizer(text, add_special_tokens=False, truncation=True,
                                         max_length=args.teacher_tokens, return_tensors='pt')['input_ids']
    joined, length = append_teacher(prefix, encoded)
    hidden = system.hidden(joined)
    # Causal attention: the classifier position never sees the appended teacher text.
    prefix_hidden = hidden[:, length - 1].float()
    ce = F.cross_entropy(system.head(prefix_hidden), label, label_smoothing=.03)
    aux = auxiliary(system, family_head, families, prefix_hidden, row) if family_head else ce.new_zeros(())
    lm_head = system.model.get_base_model().lm_head
    response = hidden[:, length - 1:length + encoded.shape[1] - 1]
    encoded = encoded.cuda()
    reading = ce.new_zeros(())
    for offset in range(0, encoded.shape[1], 64):
        logits = lm_head(response[:, offset:offset + 64]).float()
        reading = reading + F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                            encoded[:, offset:offset + 64].reshape(-1), reduction='sum')
    return ce, reading / encoded.shape[1], aux


@torch.inference_mode()
def predict(system, frame, data, split, path, batch_size):
    system.model.eval(); system.head.eval()
    scores, rows, start = [], list(frame.itertuples(index=False)), time.monotonic()
    groups = [rows[offset:offset + batch_size] for offset in range(0, len(rows), batch_size)]

    def batch(group):
        return system.processor.apply_chat_template(
            [messages(r, data, split, maxside=system.maxside) for r in group], tokenize=True,
            add_generation_prompt=True, padding=True, return_dict=True, return_tensors='pt')

    for offset, inputs in zip(range(0, len(rows), batch_size), prefetched(batch, groups, ahead=4)):
        group = rows[offset:offset + batch_size]
        with torch.autocast('cuda', dtype=system.dtype):
            hidden = system.hidden(inputs)[:, -1].float()
        scores.append(system.head(hidden).softmax(-1).cpu().numpy())
        done = offset + len(group)
        if done % 200 < batch_size or done == len(rows):
            print(json.dumps({'stage': 'predict', 'split': split, 'rows': done, 'total': len(rows),
                              'seconds': round(time.monotonic() - start)}), flush=True)
    p = np.concatenate(scores)
    save_bundle(path, frame.trace_id.tolist(), system.classes, p, kind='pod_image_predictions')
    return p


def run(args):
    seed_all(args.seed)
    # Dataset v3: evidence-first teacher targets and rare-code weights, keyed by trace_id.
    args.v3 = ({r['trace_id']: r for r in map(json.loads, Path(args.dataset).read_text().splitlines())}
               if args.dataset else None)
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, test = read_csv(data / 'train.csv'), read_csv(data / 'test.csv')
    classes = label_space(read_csv(data / 'taxonomy_v1.csv'))
    ci = {c: i for i, c in enumerate(classes)}
    folds = make_folds(train, seed=args.split_seed)
    fit = train.copy() if args.full_train else train.loc[folds != args.fold].reset_index(drop=True)
    validation = train.loc[folds == args.fold].reset_index(drop=True)
    if args.smoke:
        fit, validation, test, args.epochs = fit.head(8), validation.head(4), test.head(4), 1
    if args.rare_copies > 1:
        fit = oversampled(fit, args.rare_copies)
    if args.pseudo:
        pseudo = {r['trace_id']: r for r in map(json.loads, Path(args.pseudo).read_text().splitlines())}
        extra = read_csv(data / 'unlabelled.csv')
        extra = extra[extra.trace_id.isin(pseudo)].copy()
        extra['soft'] = extra.trace_id.map(lambda t: pseudo[t]['soft'])
        extra['pseudo_weight'] = extra.trace_id.map(lambda t: pseudo[t]['weight'])
        extra['root_code'] = extra.soft.map(lambda d: max(d, key=d.get))
        fit = pd.concat([fit, extra], ignore_index=True)
        print(json.dumps({'stage': 'pseudo_rows', 'added': len(extra)}), flush=True)
    import importlib.metadata
    environment = {'args': {k: v for k, v in vars(args).items() if k != 'v3'}, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                   'versions': {p: importlib.metadata.version(p) for p in ['transformers', 'peft', 'accelerate']},
                   'model_config_sha256': fingerprint(Path(args.model) / 'config.json'),
                   'train_sha256': fingerprint(data / 'train.csv'), 'classes': classes,
                   'fit_rows': len(fit), 'validation_rows': 0 if args.full_train else len(validation)}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment), flush=True)

    system = VisionSystem(args.model, classes, args)
    system.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    system.model.enable_input_require_grads(); system.model.config.use_cache = False
    params = [p for p in system.model.parameters() if p.requires_grad]
    heads = list(system.head.parameters())
    family_head, families = None, None
    if args.aux_weight > 0:
        families = {name: i for i, name in enumerate(sorted({c.split('-', 1)[0].upper() for c in classes}))}
        family_head = torch.nn.Linear(system.head.in_features, len(families), device='cuda')
        heads += list(family_head.parameters()) + list(system.correct_head.parameters())
    optimizer = torch.optim.AdamW([{'params': params, 'lr': args.lr}, {'params': heads, 'lr': args.head_lr}],
                                  weight_decay=.01)
    updates = args.epochs * math.ceil(len(fit) / args.accumulation)
    step, start, logs, metrics = 0, time.monotonic(), [], []
    for epoch in range(args.epochs):
        system.model.train(); system.head.train(); optimizer.zero_grad(set_to_none=True)
        order = np.random.default_rng(args.seed + epoch).permutation(len(fit))
        losses = []
        prefixes = prefetched(lambda i: encode_row(system, fit.iloc[i], data), list(order))
        for k, (i, prefix) in enumerate(zip(order, prefixes)):
            with torch.autocast('cuda', dtype=system.dtype):
                ce, reading, aux = loss_terms(system, fit.iloc[i], data, ci, k + epoch, args, family_head, families,
                                              prefix)
            window = min(args.accumulation, len(order) - (k // args.accumulation) * args.accumulation)
            row = fit.iloc[i]
            if isinstance(row.get('soft'), dict):
                weight = row.pseudo_weight
            else:
                weight = args.v3[row.trace_id]['weight'] if args.v3 else 1.
            ((weight * ce + args.reading_weight * reading + args.aux_weight * aux) / window).backward()
            losses.append([float(ce), float(reading)])
            if (k + 1) % args.accumulation == 0 or k + 1 == len(order):
                factor = min(1., (step + 1) / max(1, .05 * updates)) * .5 * (1 + math.cos(math.pi * step / max(1, updates)))
                optimizer.param_groups[0]['lr'] = args.lr * factor
                optimizer.param_groups[1]['lr'] = args.head_lr * factor
                torch.nn.utils.clip_grad_norm_(params + heads, 1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if (k + 1) % 50 == 0 or k + 1 == len(order):
                mean = np.mean(losses[-50:], axis=0)
                log = {'stage': 'train', 'epoch': epoch + 1, 'rows': k + 1, 'total': len(fit),
                       'root_loss': float(mean[0]), 'reading_loss': float(mean[1]),
                       'seconds': round(time.monotonic() - start),
                       'max_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1)}
                print(json.dumps(log), flush=True); logs.append(log)
                (out / 'training_log.json').write_text(json.dumps(logs, indent=2))
        system.save(out / f'epoch-{epoch + 1}', {**environment, 'epoch': epoch + 1, 'trained_ids': fit.trace_id.tolist()})
        if not args.full_train:
            p = predict(system, validation, data, 'train', out / f'validation_epoch_{epoch + 1}.npz', args.batch_size)
            prior = HierarchicalPrior(classes).fit(fit).predict_proba(validation)
            metric = {'epoch': epoch + 1, 'rows': len(validation), 'fold': args.fold,
                      'map3': map3(validation.root_code, p, classes),
                      'fixed_calibration_map3': map3(validation.root_code, calibrated(p, prior, .25, 2.), classes),
                      'seconds': round(time.monotonic() - start)}
            metrics.append(metric); print(json.dumps(metric), flush=True)
            (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    system.model.gradient_checkpointing_disable()
    del optimizer; gc.collect(); torch.cuda.empty_cache()
    if args.predict_test:
        p = predict(system, test, data, 'test', out / 'test.npz', args.batch_size)
        write_submission(test, p, classes, out / 'submission_raw.csv')
        prior = HierarchicalPrior(classes).fit(train).predict_proba(test)
        write_submission(test, calibrated(p, prior, .25, 2.), classes, out / 'submission_calibrated.csv')
    print('K12_POD_RUN_COMPLETE', flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=2026); p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--fold', type=int, default=0); p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--rank', type=int, default=32); p.add_argument('--max-pixels', type=int, default=1280 * 1280)
    p.add_argument('--maxside', type=int, default=1280); p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--lr', type=float, default=5e-5); p.add_argument('--head-lr', type=float, default=5e-4)
    p.add_argument('--accumulation', type=int, default=8); p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--reading-weight', type=float, default=0.); p.add_argument('--teacher-tokens', type=int, default=384)
    p.add_argument('--full-train', action='store_true'); p.add_argument('--predict-test', action='store_true')
    p.add_argument('--aux-weight', type=float, default=0.); p.add_argument('--dataset', help='train_v3.jsonl')
    p.add_argument('--pseudo', help='pseudo_v3.jsonl; photos read from images/unlabelled')
    p.add_argument('--secondary-weight', type=float, default=0.); p.add_argument('--rare-copies', type=int, default=1)
    p.add_argument('--load-4bit', action='store_true'); p.add_argument('--lora-vision', action='store_true')
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
