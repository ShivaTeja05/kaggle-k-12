"""Fine-tune Qwen3-VL to transcribe a student's working, then transcribe photos.

Stage one of read-then-diagnose. Training targets are teacher transcriptions of the
fitting rows; the prompt carries only the photo, question context and answer key.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import label_space, make_folds, read_csv
from .pod_train import prefetched
from .reading_finetune import append_teacher
from .vision import VisionSystem, messages, seed_all


def train(args):
    seed_all(args.seed)
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame = read_csv(data / 'train.csv')
    classes = label_space(read_csv(data / 'taxonomy_v1.csv'))
    folds = make_folds(frame, seed=args.split_seed)
    fit = frame if args.full_train else frame.loc[folds != args.fold].reset_index(drop=True)
    if args.smoke:
        fit, args.epochs = fit.head(8), 1
    system = VisionSystem(args.model, classes, args)
    system.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    system.model.enable_input_require_grads(); system.model.config.use_cache = False
    params = [p for p in system.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    lm_head = system.model.get_base_model().lm_head
    updates = args.epochs * math.ceil(len(fit) / args.accumulation)
    step, start = 0, time.monotonic()

    def encode(i):
        row = fit.iloc[i]
        prefix = system.encode(row, data, 'train', task='transcribe')
        target = system.processor.tokenizer(str(row.transcription_steps) + system.processor.tokenizer.eos_token,
                                            add_special_tokens=False, truncation=True, max_length=args.max_tokens,
                                            return_tensors='pt')['input_ids']
        return append_teacher(prefix, target) + (target,)

    for epoch in range(args.epochs):
        system.model.train()
        order = list(np.random.default_rng(args.seed + epoch).permutation(len(fit)))
        losses = []
        for k, (joined, length, target) in enumerate(prefetched(encode, order)):
            with torch.autocast('cuda', dtype=system.dtype):
                hidden = system.hidden(joined)
            response = hidden[:, length - 1:length + target.shape[1] - 1]
            target = target.cuda()
            loss = hidden.new_zeros((), dtype=torch.float32)
            for offset in range(0, target.shape[1], 64):
                logits = lm_head(response[:, offset:offset + 64]).float()
                loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              target[:, offset:offset + 64].reshape(-1), reduction='sum')
            loss = loss / target.shape[1]
            window = min(args.accumulation, len(order) - (k // args.accumulation) * args.accumulation)
            (loss / window).backward(); losses.append(float(loss))
            if (k + 1) % args.accumulation == 0 or k + 1 == len(order):
                factor = min(1., (step + 1) / max(1, .05 * updates)) * .5 * (1 + math.cos(math.pi * step / max(1, updates)))
                optimizer.param_groups[0]['lr'] = args.lr * factor
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if (k + 1) % 50 == 0 or k + 1 == len(order):
                print(json.dumps({'stage': 'train', 'epoch': epoch + 1, 'rows': k + 1, 'total': len(order),
                                  'loss': float(np.mean(losses[-50:])), 'seconds': round(time.monotonic() - start),
                                  'max_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)
    system.save(out / 'checkpoint', {'args': vars(args), 'classes': classes, 'trained_ids': fit.trace_id.tolist(),
                                     'task': 'transcribe'})
    print('K12_TRANSCRIBER_TRAINED', flush=True)


@torch.inference_mode()
def transcribe(args):
    metadata = json.loads((Path(args.checkpoint) / 'metadata.json').read_text())
    options = argparse.Namespace(**metadata['args'])
    data = Path(args.data)
    classes = label_space(read_csv(data / 'taxonomy_v1.csv'))
    system = VisionSystem(args.model, classes, options, checkpoint=args.checkpoint)
    system.model.eval()
    frame = read_csv(args.csv)
    if args.ids:
        keep = set(json.loads(Path(args.ids).read_text()))
        frame = frame[frame.trace_id.isin(keep)]
    if args.limit:
        frame = frame.head(args.limit)
    out = Path(args.output)
    done = json.loads(out.read_text()) if out.exists() else {}
    rows = [r for r in frame.itertuples(index=False) if r.trace_id not in done]
    groups = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    tokenizer = system.processor.tokenizer

    def batch(group):
        return system.processor.apply_chat_template(
            [messages(r, data, args.split, task='transcribe', maxside=system.maxside) for r in group],
            tokenize=True, add_generation_prompt=True, padding=True, return_dict=True, return_tensors='pt')

    start = time.monotonic()
    for group, inputs in zip(groups, prefetched(batch, groups, ahead=3)):
        inputs = {k: v.cuda() for k, v in inputs.items() if k != 'token_type_ids'}
        with torch.autocast('cuda', dtype=system.dtype):
            generated = system.model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False,
                                              repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        done.update({r.trace_id: t.strip() for r, t in zip(group, texts)})
        out.write_text(json.dumps(done, ensure_ascii=False))
        print(json.dumps({'stage': 'transcribe', 'rows': len(done), 'total': len(frame),
                          'seconds': round(time.monotonic() - start)}), flush=True)
    print('K12_TRANSCRIBE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    t = commands.add_parser('train')
    t.add_argument('--data', required=True); t.add_argument('--model', required=True); t.add_argument('--output', required=True)
    t.add_argument('--seed', type=int, default=2026); t.add_argument('--split-seed', type=int, default=2026)
    t.add_argument('--fold', type=int, default=0); t.add_argument('--epochs', type=int, default=2)
    t.add_argument('--rank', type=int, default=32); t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--accumulation', type=int, default=8); t.add_argument('--max-tokens', type=int, default=512)
    t.add_argument('--max-pixels', type=int, default=1280 * 1280); t.add_argument('--maxside', type=int, default=1280)
    t.add_argument('--dtype', default='bfloat16'); t.add_argument('--full-train', action='store_true')
    t.add_argument('--smoke', action='store_true')
    g = commands.add_parser('transcribe')
    g.add_argument('--checkpoint', required=True); g.add_argument('--model', required=True)
    g.add_argument('--data', required=True); g.add_argument('--csv', required=True)
    g.add_argument('--split', required=True); g.add_argument('--output', required=True)
    g.add_argument('--ids', help='JSON list of trace_ids to keep'); g.add_argument('--limit', type=int)
    g.add_argument('--batch-size', type=int, default=8); g.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    {'train': train, 'transcribe': transcribe}[args.command](args)
