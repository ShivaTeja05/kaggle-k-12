"""CPU-only, offline: what the Kaggle image and the hosted Qwen3.8-27B copy contain. Uses no GPU quota.

Reports package versions (is bitsandbytes there?), the model files and their SHA256SUMS, which weights belong to the
vision tower, the layer types, and the prompt length of real training rows with thinking switched off.
"""
import importlib, importlib.metadata, json, os, sys, time
from pathlib import Path

report = {'python': sys.version, 'packages': {}, 'imports': {}}
for name in ['torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors', 'tokenizers', 'pillow', 'torchao', 'optimum', 'auto_gptq', 'autoawq', 'unsloth']:
    try: report['packages'][name] = importlib.metadata.version(name)
    except Exception as exc: report['packages'][name] = 'MISSING'
for name in ['torch', 'transformers', 'peft', 'bitsandbytes']:
    try: importlib.import_module(name); report['imports'][name] = 'ok'
    except Exception as exc: report['imports'][name] = repr(exc)[:300]
root = Path('/kaggle/input')
configs = [p for p in root.rglob('config.json') if 'qwen' in str(p).lower()]
report['config_paths'] = [str(p) for p in configs]
model = configs[0].parent
files = sorted((p.name, p.stat().st_size) for p in model.iterdir() if p.is_file())
report['model_dir'] = str(model); report['files'] = files; report['total_gb'] = round(sum(s for _, s in files) / 1e9, 2)
cfg = json.loads((model / 'config.json').read_text())
report['config'] = {k: cfg.get(k) for k in ('architectures', 'model_type', 'transformers_version', 'torch_dtype', 'dtype')}
text = cfg.get('text_config', {}); vis = cfg.get('vision_config', {})
report['text_config'] = {k: text.get(k) for k in ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size', 'intermediate_size')}
lt = text.get('layer_types') or text.get('layers_block_type')
report['layer_types_counts'] = {x: lt.count(x) for x in set(lt)} if isinstance(lt, list) else lt
report['vision_config'] = {k: vis.get(k) for k in ('model_type', 'hidden_size', 'depth', 'num_hidden_layers', 'patch_size', 'spatial_merge_size', 'out_hidden_size')}
index = json.loads((model / 'model.safetensors.index.json').read_text())
names = list(index['weight_map'])
tops = {}
for n in names: tops['.'.join(n.split('.')[:2])] = tops.get('.'.join(n.split('.')[:2]), 0) + 1
report['weight_name_prefixes'] = tops
report['vision_weight_names_sample'] = [n for n in names if 'visual' in n or 'vision' in n][:8]
layer0 = sorted({n.split('.layers.0.', 1)[1] for n in names if '.layers.0.' in n and 'visual' not in n and 'vision' not in n})
report['language_layer0_weights'] = layer0
gdn = next((i for i in range(64) if any(f'.layers.{i}.' in n and ('in_proj' in n or 'gdn' in n.lower() or 'delta' in n.lower()) for n in names)), None)
if gdn is not None:
    report['gdn_layer_index'] = gdn
    report['gdn_layer_weights'] = sorted({n.split(f'.layers.{gdn}.', 1)[1] for n in names if f'.layers.{gdn}.' in n and 'visual' not in n})
report['has_sha256sums'] = (model / 'SHA256SUMS').exists()
if report['has_sha256sums']: report['sha256sums_head'] = (model / 'SHA256SUMS').read_text().splitlines()[:25]
for extra in ['preprocessor_config.json', 'generation_config.json']:
    if (model / extra).exists(): report[extra] = json.loads((model / extra).read_text())
# prompt length on real rows, thinking off, same message layout as k12.vision.messages
try:
    import pandas as pd
    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    data = next(p.parent for p in root.rglob('train.csv') if (p.parent / 'taxonomy_v1.csv').exists())
    report['data_dir'] = str(data); report['data_listing'] = sorted(x.name for x in data.iterdir())[:20]
    t0 = time.time(); processor = AutoProcessor.from_pretrained(str(model), local_files_only=True); report['processor_class'] = type(processor).__name__
    report['processor_load_seconds'] = round(time.time() - t0, 1)
    df = pd.read_csv(data / 'train.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str}).head(6)
    lengths = []
    for row in df.itertuples(index=False):
        path = next(p for p in [data / 'images' / 'train' / str(row.image_file), data / str(row.image_file)] if p.is_file())
        with Image.open(path) as im: img = ImageOps.exif_transpose(im).convert('RGB')
        img.thumbnail((1024, 1024))
        msg = [{'role': 'user', 'content': [{'type': 'text', 'text': "Student's handwritten working:"}, {'type': 'image', 'image': img},
               {'type': 'text', 'text': f'Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\nAnswer key: {row.correct_answer}\nROOT:'}]}]
        out = {}
        for flag in (False, True):
            enc = processor.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=flag)
            out['thinking_' + str(flag)] = int(enc['input_ids'].shape[1])
            if len(lengths) == 0: report['prompt_tail_thinking_' + str(flag)] = processor.tokenizer.decode(enc['input_ids'][0, -12:])
        out['image_size'] = img.size; out['keys'] = sorted(enc.keys()); lengths.append(out)
    report['prompt_tokens'] = lengths
except Exception as exc:
    import traceback; report['prompt_check_error'] = traceback.format_exc()[-1500:]
report['disk'] = os.popen('df -h /kaggle/working /kaggle/input /tmp 2>/dev/null | tail -4').read()
report['ram_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1)
Path('/kaggle/working/q38_inspection.json').write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str))
