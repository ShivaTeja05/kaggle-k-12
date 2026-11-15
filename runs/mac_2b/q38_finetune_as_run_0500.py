"""LoRA fine-tune of Qwen3.8-27B (photo in, teacher-style JSON answer out) with a classification layer on top,
and merging the adapter for vLLM.

Standalone on purpose (torch, transformers, peft, pillow, numpy only) so it runs on a pod without the rest of the project.

    train    rows from scripts/build_q38_finetune.py ({"trace_id", "messages", "target", "label"}). Two losses in one pass:
             (1) the written answer: cross-entropy on the target text only; thinking is off (the prompt ends with an empty
                 think block, exactly as vLLM renders it with enable_thinking false);
             (2) the classification layer: a new LayerNorm + Linear on the model's state at the last prompt token (after it
                 has read the photo and the question, before it writes anything), trained on the teacher's root code.
             --max-minutes makes the run finish on time: the learning-rate schedule follows whichever is further along, the
             planned steps or the clock, so a slow card ends with a complete (shorter) schedule, not a cut-off one.
    head     score request files with the classification layer: one forward pass per photo, no text generated; writes a
             probability bundle (ids, classes, p) in the project's .npz format.
    features the model's untrained state at the last prompt token for every row, several layers, no training: cheap input for
             small classifiers fitted on a laptop (a first check of how much the 27B model already separates the codes).
    merge    base + adapter -> one model folder that `vllm serve` can load (the head file is copied next to it).
    probe    print the modules LoRA would attach to and the token count of a few rows (no training).
"""
import argparse
import base64
import io
import json
import math
from pathlib import Path
import random
import shutil
import time

EMPTY_THINK = '<think>\n\n</think>\n\n'
IMAGE_ROOT = None  # folder that 'file:' photo references are relative to (set from --image-root)


def read_rows(path, limit=None):
    rows, header = [], {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                if r.get('header'):
                    header = r
                elif 'messages' in r:
                    rows.append(r)
                    if limit and len(rows) >= limit:
                        break
    return rows, header


def images_of(messages):
    from PIL import Image
    found = []
    for m in messages:
        if isinstance(m['content'], list):
            for part in m['content']:
                if part.get('type') == 'image_url':
                    url = part['image_url']['url']
                    if url.startswith('file:'):  # same preparation as the data-URL files: upright, RGB, at most 1,024 px, JPEG quality 90
                        from PIL import ImageOps
                        with Image.open(Path(IMAGE_ROOT) / url[5:]) as raw:
                            image = ImageOps.exif_transpose(raw).convert('RGB')
                        image.thumbnail((1024, 1024)); buffer = io.BytesIO(); image.save(buffer, format='JPEG', quality=90)
                        found.append(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))
                    else:
                        found.append(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB'))
    return found


def prompt_text(processor, messages):
    """The chat prompt as vLLM renders it with thinking off: it must end with the empty think block."""
    import copy
    # a copy, because apply_chat_template rewrites the photo parts of the messages it is given
    text = processor.apply_chat_template(copy.deepcopy(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if text.endswith(EMPTY_THINK):
        return text
    if text.endswith('<think>\n'):  # the template ignored enable_thinking: close the block ourselves
        return text + '\n</think>\n\n'
    raise ValueError(f'unexpected end of prompt: {text[-80:]!r}')


def encode(processor, row, max_target=None):
    """(model inputs, prompt length). With max_target the row's target text is appended after the prompt."""
    import torch
    photos = images_of(row['messages']); text = prompt_text(processor, row['messages'])
    if text.count('<|image_pad|>') != len(photos):
        raise ValueError(f"{row.get('trace_id')}: {text.count('<|image_pad|>')} photo slots in the prompt but {len(photos)} photos")
    inputs = processor(text=[text], images=photos or None, return_tensors='pt')
    if photos and 'pixel_values' not in inputs:
        raise ValueError(f"{row.get('trace_id')}: the processor returned no photo data")
    out, start = dict(inputs), inputs['input_ids'].shape[1]
    if max_target:
        goal = processor.tokenizer(row['target'] + '<|im_end|>', add_special_tokens=False, truncation=True, max_length=max_target,
                                   return_tensors='pt')['input_ids']
        out['input_ids'] = torch.cat([inputs['input_ids'], goal], dim=1)
        for key in ('attention_mask', 'token_type_ids', 'mm_token_type_ids'):
            if key in out:
                pad = torch.ones_like(goal) if key == 'attention_mask' else torch.zeros_like(goal)
                out[key] = torch.cat([out[key], pad], dim=1)
    return out, start


def lora_targets(model):
    """Every linear layer of the language model (not the vision tower, not the output head), by full name."""
    import torch
    head = model.get_output_embeddings()
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and m is not head and not any(k in n.lower() for k in ('visual', 'vision', 'merger', 'lm_head', 'mtp'))]
    leaves = {}
    for n in names:
        leaves[n.rsplit('.', 1)[-1]] = leaves.get(n.rsplit('.', 1)[-1], 0) + 1
    return names, leaves


def load(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    global IMAGE_ROOT
    IMAGE_ROOT = getattr(args, 'image_root', None)
    processor = AutoProcessor.from_pretrained(args.model)
    extra = {}
    if args.device == 'auto':  # spread over every GPU (Kaggle 2x T4), leaving room for activations
        limits = args.gpu_memory.split(',')  # one value for every GPU, or one per GPU ("10GiB,8GiB")
        extra = dict(device_map='auto', max_memory={i: limits[min(i, len(limits) - 1)] for i in range(torch.cuda.device_count())})
    else:
        extra = dict(device_map={'': args.device})
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=amp_dtype(args) or torch.float32, **extra)
    return processor, model


def on_gpu(args):
    return args.device == 'auto' or args.device.startswith('cuda')


def amp_dtype(args):
    """Half precision used on a GPU: bfloat16 by default, float16 where the card has no bfloat16 (Tesla T4). None off GPU."""
    import torch
    if not on_gpu(args):
        return None
    return getattr(torch, args.dtype) if args.dtype else torch.bfloat16


def entry(model):
    """The device that inputs go to (the embedding layer's; with several GPUs the last layers sit elsewhere)."""
    return model.get_input_embeddings().weight.device


def peft_ready():
    """Kaggle's image has a PEFT whose optional torchao check raises; we never use torchao, so switch that check off."""
    import peft.import_utils as checks
    try:
        checks.is_torchao_available()
    except ImportError as exc:
        if 'torchao' not in str(exc):
            raise
        import peft.tuners.lora.torchao as lora_torchao
        checks.is_torchao_available = lambda: False; lora_torchao.is_torchao_available = lambda: False


def make_head(width, classes, device):
    """The classification layer: normalise the model's state, then one linear map to a score per code, plus one extra
    output: is the student's final answer correct (yes/no)? Always float32."""
    import torch
    return torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, classes + 1)).to(device=device, dtype=torch.float32)


def forward(model, inputs, keys, args, at=None):
    """(logits, last-layer state, input ids) for one row.

    With --answer-logits-only the vocabulary scores are computed only where the loss needs them: logits[0] then starts at
    position at - 1 (the last prompt token), and is None when `at` is not given (scoring needs no vocabulary scores at all).
    Same numbers, but about 2 GB less memory per 1,000 prompt tokens with gradients, which small cards need."""
    import torch
    batch = {key: inputs[key].to(entry(model)) for key in keys}
    half = amp_dtype(args)
    with torch.autocast('cuda', dtype=half or torch.bfloat16, enabled=half is not None):
        if getattr(args, 'answer_logits_only', False):
            import inspect
            whole = model.get_base_model() if hasattr(model, 'get_base_model') else model
            accepted = set(inspect.signature(whole.model.forward).parameters)
            state = whole.model(**{k: v for k, v in batch.items() if k in accepted}).last_hidden_state
            logits = None if at is None else whole.lm_head(state[:, at - 1:].to(whole.lm_head.weight.device))
            return logits, state, batch['input_ids']
        out = model(**batch, output_hidden_states=True)
    return out.logits, out.hidden_states[-1], batch['input_ids']


def accepted_keys(base, inputs):
    import inspect
    accepted = set(inspect.signature(base.forward).parameters)
    return [key for key in inputs if key in accepted]


def train(args):
    import torch
    from torch.nn import functional as F
    from peft import LoraConfig, get_peft_model
    random.seed(args.seed); torch.manual_seed(args.seed); cuda = on_gpu(args); peft_ready()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows, _ = read_rows(args.train, args.limit)
    use_head = args.head_weight > 0 and all('label' in r for r in rows)
    classes = json.loads(Path(args.classes).read_text()) if args.classes else sorted({r.get('label', 'NONE') for r in rows})
    index = {c: i for i, c in enumerate(classes)}
    processor, base = load(args)
    width = base.config.get_text_config().hidden_size
    names, leaves = lora_targets(base)
    print(json.dumps({'stage': 'targets', 'modules': len(names), 'by_leaf_name': leaves}), flush=True)
    if not names:
        raise ValueError('no LoRA targets found')
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=.05, target_modules=names))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        checkpointing = True
    except Exception as exc:  # the run still works without it, with more memory
        checkpointing = f'off: {exc}'
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': params, 'base_lr': args.lr}]
    head = None
    if use_head:
        head = make_head(width, len(classes), entry(base))
        groups.append({'params': list(head.parameters()), 'base_lr': args.head_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype(args) == torch.float16)  # float16 needs loss scaling; bfloat16 does not
    every = [p for g in groups for p in g['params']]
    planned = args.epochs * math.ceil(len(rows) / args.accumulation)
    budget = args.max_minutes * 60 if args.max_minutes else None
    print(json.dumps({'stage': 'train_start', 'rows': len(rows), 'epochs': args.epochs, 'planned_updates': planned, 'max_minutes': args.max_minutes,
                      'trainable': sum(p.numel() for p in params), 'gradient_checkpointing': checkpointing,
                      'classification_layer': bool(use_head), 'classes': len(classes), 'head_weight': args.head_weight}), flush=True)

    def save():
        model.save_pretrained(out / 'adapter')
        if head is not None:
            torch.save({'state': head.state_dict(), 'classes': classes, 'width': width}, out / 'adapter' / 'head.pt')

    model.train()
    step, seen, tokens, text_losses, head_losses, hits, failed, pending = 0, 0, 0, [], [], [], 0, 0
    start, saved, done, keys = time.monotonic(), time.monotonic(), False, None
    for epoch in range(args.epochs):
        order = list(range(len(rows))); random.Random(args.seed + epoch).shuffle(order)
        for k, i in enumerate(order):
            inputs, at = encode(processor, rows[i], args.max_target)
            if keys is None:
                keys = accepted_keys(base, inputs)
                print(json.dumps({'stage': 'inputs', 'keys': list(inputs), 'used': keys, 'tokens': int(inputs['input_ids'].shape[1]),
                                  'target_tokens': int(inputs['input_ids'].shape[1] - at)}), flush=True)
            try:
                logits, state, ids = forward(model, inputs, keys, args, at)
                answer = logits[0, :-1] if args.answer_logits_only else logits[0, at - 1:-1]
                loss = F.cross_entropy(answer.float(), ids[0, at:].to(logits.device))
                text_loss, extra, hit = float(loss.detach()), None, None
                if head is not None:
                    both = head(state[0, at - 1].float().unsqueeze(0).to(entry(base))); scores = both[:, :-1]
                    want = torch.tensor([index[rows[i]['label']]], device=scores.device)
                    extra = F.cross_entropy(scores, want, label_smoothing=.05)
                    if 'final_ok' in rows[i]:  # second output: the teacher's "final answer is correct" flag
                        extra = extra + .5 * F.binary_cross_entropy_with_logits(both[:, -1], torch.tensor([float(rows[i]['final_ok'])], device=both.device))
                    hit = float(scores.argmax(1).item() == want.item())
                    loss = loss + args.head_weight * extra.to(loss.device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss at row {seen}')
                scaler.scale(loss / args.accumulation).backward()
                text_losses.append(text_loss)
                if extra is not None:
                    head_losses.append(float(extra.detach())); hits.append(hit)
                seen += 1; pending += 1; tokens += int(ids.shape[1]); problem = None
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
                failed += 1
                if failed > args.skip_failed_rows:
                    raise
                problem = f'{type(exc).__name__}: {str(exc)[:200]}'
            logits = state = loss = extra = both = answer = None
            if problem:  # --skip-failed-rows: drop this row and the gradients gathered since the last update, then carry on
                optimizer.zero_grad(set_to_none=True); pending = 0
                if cuda:
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                print(json.dumps({'stage': 'row_skipped', 'trace_id': rows[i].get('trace_id'), 'tokens': int(inputs['input_ids'].shape[1]),
                                  'failed_so_far': failed, 'allowed': args.skip_failed_rows, 'error': problem}), flush=True)
            if (k + 1) % args.accumulation == 0 or k + 1 == len(order):
                elapsed = time.monotonic() - start
                progress = max(step / max(1, planned), elapsed / budget if budget else 0.)
                done = progress >= 1.
                factor = min(1., (step + 1) / max(1, .05 * planned)) * .5 * (1 + math.cos(math.pi * min(1., progress)))
                for g in optimizer.param_groups:
                    g['lr'] = g['base_lr'] * factor
                if pending:  # nothing to apply when every row since the last update was skipped
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(every, 1.)
                    scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); step += 1; pending = 0
                if step % args.log_every == 0 or done:
                    w = args.accumulation * args.log_every
                    mean = lambda x: round(sum(x[-w:]) / len(x[-w:]), 4) if x else None
                    print(json.dumps({'stage': 'train', 'epoch': epoch + 1, 'update': step, 'of': planned, 'rows': seen, 'text_loss': mean(text_losses),
                                      'head_loss': mean(head_losses), 'head_top1_running': mean(hits), 'lr_factor': round(factor, 4),
                                      'seconds': round(elapsed, 1), 'rows_per_s': round(seen / elapsed, 3), 'tokens_per_s': round(tokens / elapsed),
                                      'progress': round(progress, 3),
                                      'max_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1) if cuda else 0,
                                      'max_gpu_gb_each': [round(torch.cuda.max_memory_allocated(d) / 1e9, 1) for d in range(torch.cuda.device_count())] if cuda else [],
                                      'rows_skipped': failed}), flush=True)
                if args.save_minutes and time.monotonic() - saved > args.save_minutes * 60:
                    save(); saved = time.monotonic()
                if done:
                    break
        if done:
            break
    save()
    (out / 'train_report.json').write_text(json.dumps({'rows_seen': seen, 'rows_skipped': failed, 'updates': step, 'planned_updates': planned,
                                                       'seconds': round(time.monotonic() - start, 1), 'final_text_loss': text_losses[-1] if text_losses else None,
                                                       'classification_layer': bool(use_head), 'args': vars(args)}, indent=1))
    print('K12_Q38_FINETUNE_COMPLETE', flush=True)


def head_scores(args):
    """Probabilities from the classification layer for every request row: one forward pass each, nothing generated."""
    import numpy as np
    import torch
    processor, base = load(args)
    model = base
    if args.adapter:
        from peft import PeftModel
        peft_ready(); model = PeftModel.from_pretrained(base, args.adapter)
    saved = torch.load(args.head, map_location=entry(base), weights_only=True)
    head = make_head(saved['width'], len(saved['classes']), entry(base)); head.load_state_dict(saved['state']); head.eval(); model.eval()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    ids, probs, finals, start, keys = [], [], [], time.monotonic(), None

    def write(note):
        np.savez_compressed(out, ids=np.asarray(ids, dtype=str), classes=np.asarray(saved['classes'], dtype=str),
                            p=np.asarray(probs, dtype='float32'), final_ok=np.asarray(finals, dtype='float32'), meta=json.dumps(note))

    for path in args.requests:
        for row in read_rows(path, args.limit)[0]:
            if row['trace_id'] in ids or (args.deadline and time.time() >= args.deadline):
                continue
            inputs, at = encode(processor, row)
            keys = keys or accepted_keys(base, inputs)
            with torch.inference_mode():
                _, state, _ = forward(model, inputs, keys, args)
                both = head(state[0, at - 1].float().unsqueeze(0).to(entry(base)))
                p = torch.softmax(both[:, :-1], dim=1)[0]
            ids.append(row['trace_id']); probs.append(p.cpu().numpy()); finals.append(float(torch.sigmoid(both[0, -1])))
            if len(ids) % args.log_every == 0:
                print(json.dumps({'stage': 'head', 'rows': len(ids), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
            if len(ids) % 200 == 0:  # keep what is done if the run is cut short
                write({'kind': 'q38_finetune_head', 'partial': True})
    write({'kind': 'q38_finetune_head', 'rows': len(ids)})
    print(json.dumps({'stage': 'head_done', 'rows': len(ids), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
    print('K12_Q38_HEAD_COMPLETE', flush=True)


def features(args):
    """The model's state at the last prompt token for every request row, from several layers, with no training and no text
    generated. Small classifiers can then be fitted on these on a laptop (a "probe"), with cross-validation and many seeds."""
    import numpy as np
    import torch
    processor, base = load(args)
    model = base
    if args.adapter:
        from peft import PeftModel
        peft_ready(); model = PeftModel.from_pretrained(base, args.adapter)
    model.eval(); half = amp_dtype(args)
    layers = [int(x) for x in args.layers.split(',')]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    ids, feats, start, keys = [], [], time.monotonic(), None

    def write():
        np.savez_compressed(out, ids=np.asarray(ids, dtype=str), layers=np.asarray(layers), x=np.asarray(feats, dtype='float16'))

    for path in args.requests:
        for row in read_rows(path, args.limit)[0]:
            if row['trace_id'] in ids or (args.deadline and time.time() >= args.deadline):
                continue
            inputs, at = encode(processor, row)
            keys = keys or accepted_keys(base, inputs)
            batch = {key: inputs[key].to(entry(model)) for key in keys}
            with torch.inference_mode(), torch.autocast('cuda', dtype=half or torch.bfloat16, enabled=half is not None):
                hidden = model(**batch, output_hidden_states=True).hidden_states
            feats.append(torch.stack([hidden[l][0, at - 1].float() for l in layers]).cpu().numpy()); ids.append(row['trace_id'])
            if len(ids) % args.log_every == 0:
                print(json.dumps({'stage': 'features', 'rows': len(ids), 'seconds': round(time.monotonic() - start, 1),
                                  'layers_available': len(hidden)}), flush=True)
            if len(ids) % 500 == 0:
                write()
    write()
    print(json.dumps({'stage': 'features_done', 'rows': len(ids), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
    print('K12_Q38_FEATURES_COMPLETE', flush=True)


def merge(args):
    from peft import PeftModel
    processor, base = load(args); peft_ready()
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size='5GB')
    processor.save_pretrained(out)
    for path in Path(args.model).iterdir():  # chat template, preprocessor and generation settings exactly as the base ships them
        if path.is_file() and not path.name.endswith('.safetensors') and path.name not in ('model.safetensors.index.json', 'config.json'):
            shutil.copy2(path, out / path.name)
    if (Path(args.adapter) / 'head.pt').exists():
        shutil.copy2(Path(args.adapter) / 'head.pt', out / 'head.pt')
    print(json.dumps({'stage': 'merged', 'files': sorted(p.name for p in out.iterdir())[:40]}), flush=True)
    print('K12_Q38_MERGE_COMPLETE', flush=True)


def probe(args):
    processor, base = load(args)
    names, leaves = lora_targets(base)
    print(json.dumps({'modules': len(names), 'by_leaf_name': leaves, 'first': names[:6], 'width': base.config.get_text_config().hidden_size}))
    for row in read_rows(args.train, 4)[0]:
        inputs, at = encode(processor, row, args.max_target)
        print(json.dumps({'trace_id': row['trace_id'], 'tokens': int(inputs['input_ids'].shape[1]), 'target_tokens': int(inputs['input_ids'].shape[1] - at)}))


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='command', required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--model', required=True, help='folder with Qwen3.8-27B (bf16), or a merged folder')
    shared.add_argument('--max-target', type=int, default=900, help='most tokens of the answer that are trained on')
    shared.add_argument('--device', default='cuda:0', help="cuda:0, 'auto' (all GPUs), mps (Apple) or cpu")
    shared.add_argument('--dtype', choices=['bfloat16', 'float16'], help='GPU half precision; float16 for cards without bfloat16 (T4)')
    shared.add_argument('--gpu-memory', default='13GiB', help="per-GPU limit for the weights with --device auto; one value, or one per GPU separated by commas")
    shared.add_argument('--image-root', help="folder that 'file:' photo references in the request files are relative to")
    shared.add_argument('--log-every', type=int, default=5); shared.add_argument('--limit', type=int)
    shared.add_argument('--answer-logits-only', action='store_true',
                        help='compute vocabulary scores only at the answer positions (same loss, much less GPU memory; none at all when scoring)')
    t = sub.add_parser('train', parents=[shared])
    t.add_argument('--train', required=True); t.add_argument('--output', required=True)
    t.add_argument('--classes', help='JSON list of all codes, so the classification layer has one output per code')
    t.add_argument('--epochs', type=int, default=2); t.add_argument('--accumulation', type=int, default=8)
    t.add_argument('--lr', type=float, default=1e-4); t.add_argument('--rank', type=int, default=16)
    t.add_argument('--head-weight', type=float, default=.5, help='weight of the classification loss next to the text loss (0 = no classification layer)')
    t.add_argument('--head-lr', type=float, default=1e-3)
    t.add_argument('--max-minutes', type=float, help='finish the schedule within this many minutes of training')
    t.add_argument('--save-minutes', type=float, default=45, help='save the adapter this often while training (0 = only at the end)')
    t.add_argument('--seed', type=int, default=2026)
    t.add_argument('--skip-failed-rows', type=int, default=0,
                   help='carry on past this many rows that run out of GPU memory or give a non-finite loss (0 = stop at the first)')
    h = sub.add_parser('head', parents=[shared])
    h.add_argument('--requests', nargs='+', required=True); h.add_argument('--head', required=True); h.add_argument('--output', required=True)
    h.add_argument('--adapter', help='LoRA adapter folder when --model is the unmerged base')
    h.add_argument('--deadline', type=float, help='unix time after which no new row is scored')
    f = sub.add_parser('features', parents=[shared])
    f.add_argument('--requests', nargs='+', required=True); f.add_argument('--output', required=True)
    f.add_argument('--layers', default='-1,-9,-17,-25,-33', help='hidden-state layers to save (negative = from the top)')
    f.add_argument('--adapter'); f.add_argument('--deadline', type=float)
    m = sub.add_parser('merge', parents=[shared]); m.add_argument('--adapter', required=True); m.add_argument('--output', required=True)
    p = sub.add_parser('probe', parents=[shared]); p.add_argument('--train', required=True)
    args = ap.parse_args(argv)
    {'train': train, 'head': head_scores, 'features': features, 'merge': merge, 'probe': probe}[args.command](args)


if __name__ == '__main__':
    main()
