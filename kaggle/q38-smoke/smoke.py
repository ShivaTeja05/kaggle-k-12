"""Kaggle 2x T4 smoke test: can Qwen3.8-27B be fine-tuned here as a no-thinking photo classifier?

Loads the model in 4-bit (NF4, vision tower left in fp16) across both T4s, adds LoRA to every linear layer of the
language model, trains on 8 fit rows with the same prompt layout and last-token 76-way readout as k12.vision, then
times inference on 8 development rows. Reports seconds per row, tokens per row, memory per GPU and whether the
losses are finite. Offline, private, no test data, no submission. Everything is printed as JSON lines and written to
/kaggle/working/q38_smoke.json, also when a step fails.
"""
import gc, importlib, importlib.metadata, json, os, sys, time, traceback
from pathlib import Path

T0 = time.time()
BUDGET = 23 * 60  # seconds of wall clock after which no new step starts (the job timeout is 30 minutes)
R = {'stages': []}
OUT = Path('/kaggle/working/q38_smoke.json')


def say(stage, **kw):
    row = {'stage': stage, 'seconds': round(time.time() - T0, 1), **kw}
    R['stages'].append(row); print(json.dumps(row, default=str), flush=True)
    OUT.write_text(json.dumps(R, indent=1, default=str))


def left():
    return BUDGET - (time.time() - T0)


def main():
    pk = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors', 'torchao']:
        try: pk[name] = importlib.metadata.version(name)
        except Exception: pk[name] = 'MISSING'
    say('environment', packages=pk, python=sys.version.split()[0])
    import torch
    from torch import nn
    gpus = [{'i': i, 'name': torch.cuda.get_device_name(i), 'gb': round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
            for i in range(torch.cuda.device_count())]
    say('gpus', gpus=gpus, ram_gb=round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1),
        disk=os.popen('df -h /kaggle/working /tmp 2>/dev/null | tail -2').read())
    if pk['bitsandbytes'] == 'MISSING':
        say('STOP', reason='bitsandbytes is not in this image; a wheel must be attached before 4-bit loading can be tested'); return
    import numpy as np, pandas as pd
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    root = Path('/kaggle/input')
    model_dir = next(p.parent for p in root.rglob('config.json') if json.loads(p.read_text()).get('model_type') == 'qwen3_5')
    data = next(p.parent for p in root.rglob('train.csv') if (p.parent / 'taxonomy_v1.csv').exists())
    cfg = json.loads((model_dir / 'config.json').read_text())
    names = list(json.loads((model_dir / 'model.safetensors.index.json').read_text())['weight_map'])
    vision_keys = sorted({part for n in names for part in n.split('.')[:3] if part in ('visual', 'vision_tower', 'vision_model', 'vision')})
    text = cfg.get('text_config', {})
    lt = text.get('layer_types')
    say('model_files', model_dir=str(model_dir), architectures=cfg.get('architectures'), transformers_version=cfg.get('transformers_version'),
        gb=round(sum(p.stat().st_size for p in model_dir.iterdir() if p.is_file()) / 1e9, 1), vision_keys=vision_keys,
        hidden=text.get('hidden_size'), layers=text.get('num_hidden_layers'), layer_types=({x: lt.count(x) for x in set(lt)} if isinstance(lt, list) else lt),
        top_prefixes=sorted({'.'.join(n.split('.')[:2]) for n in names})[:12], sha256sums=(model_dir / 'SHA256SUMS').exists())

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    processor.tokenizer.padding_side = 'left'

    def image(split, filename, maxside=1024):
        path = next(p for p in [data / 'images' / split / str(filename), data / str(filename)] if p.is_file())
        with Image.open(path) as im: out = ImageOps.exif_transpose(im).convert('RGB')
        out.thumbnail((maxside, maxside)); return out

    def messages(row, split):  # same layout as k12.vision.messages(task='classify')
        content = []
        if str(row.question_image_file).strip():
            content += [{'type': 'text', 'text': 'Question diagram:'}, {'type': 'image', 'image': image('questions', row.question_image_file)}]
        content += [{'type': 'text', 'text': "Student's handwritten working:"}, {'type': 'image', 'image': image(split, row.image_file)}]
        instruction = ("Read the student's actual reasoning. Identify the root major error that the rest of the work "
                       "follows from, which need not be the first written error. NONE means no major error; minor notation "
                       "slips can still be NONE. A matching final answer alone does not establish correct reasoning. "
                       "Respect crossed-out corrections and do not invent missing steps.\nROOT:")
        content.append({'type': 'text', 'text': f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\nAnswer key: {row.correct_answer}\n{instruction}'})
        return [{'role': 'user', 'content': content}]

    thinking_kw = {'enable_thinking': False}

    def encode(row, split):
        nonlocal thinking_kw
        try:
            return processor.apply_chat_template(messages(row, split), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', **thinking_kw)
        except TypeError:
            thinking_kw = {}
            return processor.apply_chat_template(messages(row, split), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')

    frame = pd.read_csv(data / 'train.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
    tax = pd.read_csv(data / 'taxonomy_v1.csv', keep_default_na=False)
    classes = sorted(set(tax['id'].astype(str)) | {'NONE'}); ci = {c: i for i, c in enumerate(classes)}
    fit, dev = frame.iloc[:8], frame.iloc[100:108]  # speed test only: which rows does not matter, no score is reported
    enc0 = encode(fit.iloc[0], 'train')
    say('prompt', processor=type(processor).__name__, tokens=int(enc0['input_ids'].shape[1]), keys=sorted(enc0.keys()), thinking_flag_accepted=bool(thinking_kw),
        tail=processor.tokenizer.decode(enc0['input_ids'][0, -14:]), classes=len(classes))

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                               llm_int8_skip_modules=(vision_keys or None))
    t = time.time()
    model = AutoModelForImageTextToText.from_pretrained(str(model_dir), local_files_only=True, dtype=torch.float16, quantization_config=quant,
                                                        device_map='auto', max_memory={0: '12GiB', 1: '13GiB'}, low_cpu_mem_usage=True)
    dm = getattr(model, 'hf_device_map', {})
    where = {}
    for k, v in dm.items(): where[str(v)] = where.get(str(v), 0) + 1
    say('loaded', load_seconds=round(time.time() - t, 1), model_class=type(model).__name__, device_map_counts=where,
        device_map_edges=[(k, v) for k, v in list(dm.items())[:4] + list(dm.items())[-4:]],
        gpu_gb_after_load=[round(torch.cuda.memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())],
        offloaded=any(str(v) in ('cpu', 'disk') for v in dm.values()))

    kinds = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            part = 'vision' if any(('.' + k + '.') in ('.' + n + '.') for k in vision_keys) else ('language' if 'language_model' in n or 'layers.' in n else 'other')
            key = (part, n.rsplit('.', 1)[-1], type(m).__name__)
            kinds[key] = kinds.get(key, 0) + 1
    say('linear_layers', kinds=sorted([list(k) + [v] for k, v in kinds.items()], key=lambda x: (x[0], -x[3])))
    leaf = sorted({k[1] for k in kinds if k[0] == 'language' and k[1] not in ('lm_head',)})
    targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.rsplit('.', 1)[-1] in leaf and 'lm_head' not in n
               and not any(('.' + k + '.') in ('.' + n + '.') for k in vision_keys) and ('language_model' in n or 'layers.' in n)]

    import peft.import_utils as peft_imports
    try: peft_imports.is_torchao_available()
    except ImportError as exc:  # same workaround as k12.vision for the image's old torchao
        import peft.tuners.lora.torchao as peft_torchao
        peft_imports.is_torchao_available = lambda: False; peft_torchao.is_torchao_available = lambda: False
        say('torchao_dispatch_disabled', error=str(exc)[:160])
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=targets))
    for p in model.parameters():
        if p.requires_grad: p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say('lora', target_leaf_names=leaf, adapted_modules=len(targets), trainable_millions=round(trainable / 1e6, 1))

    base = model.get_base_model()
    backbone = getattr(base, 'model', None)
    last_device = None
    say('backbone', has_inner_model=backbone is not None, inner_class=type(backbone).__name__ if backbone is not None else None)

    def hidden(inp):
        inp = {k: v.to('cuda:0') for k, v in inp.items() if k != 'token_type_ids'}
        if backbone is not None:  # as k12.vision: the multimodal backbone, without materialising vocabulary logits
            return backbone(**inp, use_cache=False).last_hidden_state
        return base(**inp, use_cache=False, output_hidden_states=True).hidden_states[-1]

    hidden_size = text.get('hidden_size') or model.get_base_model().config.get_text_config().hidden_size
    # inference timing first: it needs no gradients and always fits
    model.eval(); times, finite = [], True
    with torch.inference_mode():
        for row in dev.head(5).itertuples(index=False):
            enc = encode(row, 'train'); t = time.time()
            with torch.autocast('cuda', dtype=torch.float16):
                h = hidden(enc)[:, -1].float()
            torch.cuda.synchronize(); times.append(time.time() - t); finite &= bool(torch.isfinite(h).all()); last_device = h.device
            if left() < 0: break
    say('inference', rows=len(times), seconds_per_row=[round(x, 2) for x in times], median_after_first=round(float(np.median(times[1:] or times)), 2),
        hidden_finite=finite, hidden_device=str(last_device), hidden_abs_max=float(h.abs().max()),
        gpu_gb_peak=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
    if left() > 420:
        try:  # a batch of 4 photos, to see whether batching helps on these cards
            batch = processor.apply_chat_template([messages(r, 'train') for r in dev.head(4).itertuples(index=False)], tokenize=True, add_generation_prompt=True,
                                                  padding=True, return_dict=True, return_tensors='pt', **thinking_kw)
            t = time.time()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16): hb = hidden(batch)[:, -1].float()
            torch.cuda.synchronize(); say('inference_batch4', seconds_per_row=round((time.time() - t) / 4, 2), finite=bool(torch.isfinite(hb).all()),
                                           gpu_gb_peak=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
            del batch, hb
        except Exception as exc:
            say('inference_batch4_failed', error=repr(exc)[:300])
    gc.collect(); torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(i)

    head = nn.Linear(hidden_size, len(classes), device=last_device, dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    by_device = {}
    for p in params: by_device.setdefault(str(p.device), []).append(p)
    opt = torch.optim.AdamW([{'params': params, 'lr': 5e-5}, {'params': head.parameters(), 'lr': 5e-4}], weight_decay=.01)
    model.train(); rows = []
    for k, row in enumerate(fit.itertuples(index=False)):
        if left() < 60: break
        enc = encode(row, 'train'); t = time.time()
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                h = hidden(enc)[:, -1].float()
            loss = nn.functional.cross_entropy(head(h), torch.tensor([ci[row.root_code]], device=h.device), label_smoothing=.03)
            (loss / 4).backward(); t_fb = time.time() - t
            gnorm = sum(float(torch.stack([(p.grad.float() ** 2).sum() for p in group if p.grad is not None]).sum()) for group in by_device.values()) ** .5
            if (k + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            rows.append({'row': k, 'tokens': int(enc['input_ids'].shape[1]), 'seconds': round(time.time() - t, 2), 'forward_backward': round(t_fb, 2),
                         'loss': float(loss), 'loss_finite': bool(torch.isfinite(loss)), 'grad_norm': gnorm})
            say('train_row', **rows[-1], gpu_gb_peak=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])
        except torch.cuda.OutOfMemoryError as exc:
            say('train_out_of_memory', row=k, tokens=int(enc['input_ids'].shape[1]), error=str(exc)[:300]); opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
    if rows:
        sec = [r['forward_backward'] for r in rows[1:]] or [rows[0]['forward_backward']]
        med = float(np.median(sec))
        say('SUMMARY', train_rows=len(rows), train_seconds_per_row_median=round(med, 2), train_tokens_median=int(np.median([r['tokens'] for r in rows])),
            hours_per_epoch_1199_rows=round(med * 1199 / 3600, 1), hours_two_epochs=round(2 * med * 1199 / 3600, 1),
            inference_seconds_per_row=round(float(np.median(times[1:] or times)), 2), hours_test_3498_rows=round(float(np.median(times[1:] or times)) * 3498 / 3600, 1),
            all_losses_finite=all(r['loss_finite'] for r in rows), losses=[round(r['loss'], 3) for r in rows],
            gpu_gb_peak_training=[round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())])


try:
    main()
except Exception:
    say('FAILED', error=traceback.format_exc()[-2500:])
say('done')
