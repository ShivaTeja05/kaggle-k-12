# ---- runner body (kaggle/q35-9b-ft/runner.py); build.py puts CONFIG and SCRIPT_B64 above this line ----
"""Qwen3.5-9B fine-tune rehearsal on Kaggle 2x T4 with src/k12/q38_finetune.py (embedded above, unchanged).

Modes (CONFIG['mode']):
  inspect  environment, inputs, photo files, processor and token counts. Needs no GPU.
  smoke    inspect + a 12-row `train` (speed, memory per GPU, losses; retried with another memory split if it runs out of
           memory) + `head` on a few development rows with the smoke adapter (scoring speed, bundle check).
  main     inspect + full `train` under a clock (--max-minutes) + `head` on the development file + MAP@3 inside the job.
  test     `head` on the test file with the adapter of an earlier job (attached as a kernel source), under a deadline.

Offline, private, no submission. Every stage writes its result to /kaggle/working/status.json as it goes, also on failure,
and each stage has its own time limit so the job always ends by itself (Kaggle keeps outputs only then).
"""
import base64, hashlib, importlib.metadata, importlib.util, json, os, re, subprocess, sys, threading, time, traceback
from pathlib import Path

T0 = time.time()
WORK = Path(os.environ.get('K12_WORK', '/kaggle/working'))
INPUT = Path(os.environ.get('K12_INPUT', '/kaggle/input'))
CODE = WORK / 'code'
STATUS = {'mode': CONFIG['mode'], 'config': CONFIG, 'started_unix': T0, 'stages': []}
PEAK = {'now': {}}  # stage name -> peak MiB used per GPU, from nvidia-smi


def say(stage, **kw):
    row = {'stage': stage, 't': round(time.time() - T0, 1), **kw}
    STATUS['stages'].append(row)
    print('K12RUN ' + json.dumps(row, default=str), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'status.json').write_text(json.dumps(STATUS, indent=1, default=str))


def left():
    """Seconds until the job's own deadline (well before Kaggle's timeout)."""
    return CONFIG['job_seconds'] - (time.time() - T0)


def smi(query):
    try:
        out = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        return [line.strip() for line in out.stdout.strip().splitlines()] if out.returncode == 0 else []
    except Exception:
        return []


def watch_memory():
    while True:
        used = smi('memory.used')
        for name in list(PEAK['now']):
            best = PEAK.setdefault(name, [0] * len(used))
            for i, value in enumerate(used):
                try:
                    best[i] = max(best[i], int(value))
                except (ValueError, IndexError):
                    pass
        time.sleep(4)


def run(name, argv, limit, env):
    """Run one command of the fine-tune script; echo and keep its output; kill it at `limit` seconds. Never raises."""
    log = WORK / 'logs' / f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    rows, tail, start = [], [], time.time()
    PEAK['now'][name] = True
    try:
        proc = subprocess.Popen([sys.executable, '-u', '-m', 'k12.q38_finetune', *map(str, argv)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env, cwd=str(WORK))
        killer = threading.Timer(max(30, limit), proc.kill); killer.start()
        with log.open('w') as handle:
            for line in proc.stdout:
                handle.write(line); handle.flush()
                text = line.rstrip('\n')
                tail = (tail + [text])[-60:]
                if text.startswith('{'):
                    try:
                        rows.append(json.loads(text))
                    except ValueError:
                        pass
                    if len(rows) <= 12 or len(rows) % CONFIG.get('echo_every', 1) == 0 or '"train"' not in text:
                        print(text, flush=True)
                else:
                    print(text[:400], flush=True)
        code = proc.wait(); killer.cancel()
    except Exception:
        code, tail = -999, tail + traceback.format_exc().splitlines()
    PEAK['now'].pop(name, None)
    joined = '\n'.join(tail)
    oom = re.findall(r'GPU (\d+) has a total capacity', joined) if 'out of memory' in joined.lower() else []
    result = {'name': name, 'returncode': code, 'seconds': round(time.time() - start, 1), 'killed_at_limit': code in (-9, 137) and time.time() - start >= limit - 5,
              'limit_seconds': round(limit), 'peak_gpu_mib': PEAK.get(name), 'out_of_memory_on_gpu': oom, 'log': str(log)}
    if code != 0:
        result['tail'] = tail[-25:]
    return result, rows


def versions():
    out = {}
    for name in ['torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'safetensors', 'tokenizers', 'pillow', 'numpy', 'bitsandbytes',
                 'flash-linear-attention', 'causal-conv1d', 'triton', 'torchao', 'kernels']:
        try:
            out[name] = importlib.metadata.version(name)
        except Exception:
            out[name] = None
    out['import_fla'] = importlib.util.find_spec('fla') is not None
    out['import_causal_conv1d'] = importlib.util.find_spec('causal_conv1d') is not None
    return out


def walk(root, skip=('images',)):
    for folder, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if d not in skip]
        yield Path(folder), files


def locate():
    found = {'model': None, 'data': None, 'ft': None, 'adapter': None, 'configs': []}
    for folder, files in walk(INPUT):
        if 'config.json' in files and found['model'] is None:
            try:
                cfg = json.loads((folder / 'config.json').read_text())
            except ValueError:
                cfg = {}
            found['configs'].append([str(folder), cfg.get('model_type'), cfg.get('architectures')])
            if 'Qwen3_5ForConditionalGeneration' in (cfg.get('architectures') or []) or str(cfg.get('model_type', '')).startswith('qwen3_5'):
                found['model'] = folder
        if 'train.csv' in files and 'taxonomy_v1.csv' in files:
            found['data'] = folder
        if 'ft_train.jsonl' in files and 'classes.json' in files:
            found['ft'] = folder
        if 'head.pt' in files and 'adapter_config.json' in files and (CONFIG.get('adapter_under') or '') in str(folder):
            found['adapter'] = folder
    return found


def read_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                if 'messages' in r:
                    rows.append(r)
    return rows


def photos(row):
    return [p['image_url']['url'][5:] for m in row['messages'] if isinstance(m['content'], list) for p in m['content']
            if p.get('type') == 'image_url' and p['image_url']['url'].startswith('file:')]


def size_proxy(row):
    text = sum(len(m['content']) if isinstance(m['content'], str) else sum(len(p.get('text') or '') for p in m['content']) for m in row['messages'])
    return 3000 * len(photos(row)) + text + len(row.get('target', ''))


def map3(bundle, truth):
    import numpy as np
    z = np.load(bundle, allow_pickle=False)
    ids, classes, p = [str(x) for x in z['ids']], [str(x) for x in z['classes']], z['p']
    total, top1, n = 0., 0, 0
    for k, tid in enumerate(ids):
        if tid not in truth:
            continue
        order = [classes[j] for j in np.argsort(-p[k])[:3]]
        n += 1; top1 += order[0] == truth[tid]
        if truth[tid] in order:
            total += 1. / (order.index(truth[tid]) + 1)
    return {'rows': n, 'map3': round(total / max(1, n), 4), 'top1': round(top1 / max(1, n), 4)}


def train_summary(rows):
    """Speed and losses from the script's own 'train' lines (seconds are cumulative since training started)."""
    lines = [r for r in rows if r.get('stage') == 'train']
    out = {'targets': next((r for r in rows if r.get('stage') == 'targets'), None), 'start': next((r for r in rows if r.get('stage') == 'train_start'), None),
           'inputs': next((r for r in rows if r.get('stage') == 'inputs'), None), 'skipped': [r for r in rows if r.get('stage') == 'row_skipped'][:10]}
    if lines:
        steps, prev_t, prev_rows = [], 0., 0
        for r in lines:
            if r['rows'] > prev_rows:
                steps.append(round((r['seconds'] - prev_t) / (r['rows'] - prev_rows), 2))
            prev_t, prev_rows = r['seconds'], r['rows']
        ordered = sorted(steps)
        out.update(updates=len(lines), rows=lines[-1]['rows'], seconds=lines[-1]['seconds'], seconds_per_row_each_update=steps[:40],
                   seconds_per_row_median=ordered[len(ordered) // 2] if ordered else None,
                   seconds_per_row_overall=round(lines[-1]['seconds'] / max(1, lines[-1]['rows']), 2),
                   first=lines[0], last=lines[-1], text_loss=[r.get('text_loss') for r in lines][:: max(1, len(lines) // 40)],
                   head_loss=[r.get('head_loss') for r in lines][:: max(1, len(lines) // 40)],
                   head_top1_running=[r.get('head_top1_running') for r in lines][:: max(1, len(lines) // 40)])
    return out


def main():
    mode = CONFIG['mode']
    gpus = smi('name,memory.total,memory.used')
    say('environment', python=sys.version.split()[0], packages=versions(), gpus=gpus, cpus=os.cpu_count(),
        ram_gb=round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9, 1),
        disk=subprocess.run('df -h /kaggle/working /kaggle/input /tmp 2>/dev/null | tail -3', shell=True, capture_output=True, text=True).stdout)
    source = base64.b64decode(SCRIPT_B64)
    (CODE / 'k12').mkdir(parents=True, exist_ok=True)
    (CODE / 'k12' / '__init__.py').write_text(''); (CODE / 'k12' / 'q38_finetune.py').write_bytes(source)
    say('code', sha256=hashlib.sha256(source).hexdigest(), expected=SCRIPT_SHA256, bytes=len(source))

    found = locate()
    say('inputs', model=str(found['model']), data=str(found['data']), ft=str(found['ft']), adapter=str(found['adapter']), configs=found['configs'][:8],
        top=sorted(str(p) for p in INPUT.glob('*'))[:20])
    if not (found['model'] and found['data'] and found['ft']):
        say('STOP', reason='an input is missing'); return
    model, data, ft = found['model'], found['data'], found['ft']
    cfg = json.loads((model / 'config.json').read_text()); text = cfg.get('text_config', {})
    say('model_files', gb=round(sum(p.stat().st_size for p in model.iterdir() if p.is_file()) / 1e9, 2), model_type=cfg.get('model_type'),
        architectures=cfg.get('architectures'), saved_with=cfg.get('transformers_version'), hidden=text.get('hidden_size'), layers=text.get('num_hidden_layers'),
        vocab=text.get('vocab_size'), dtype=cfg.get('dtype') or cfg.get('torch_dtype') or text.get('dtype'), tie=cfg.get('tie_word_embeddings', text.get('tie_word_embeddings')),
        files=sorted(p.name for p in model.iterdir())[:30])

    files = {name: ft / f'{name}.jsonl' for name in ('ft_train', 'ft_dev', 'ft_test_1')}
    rows = {name: read_rows(path) for name, path in files.items() if path.exists()}
    missing = {name: [u for r in rs for u in photos(r) if not (data / u).is_file()] for name, rs in rows.items()}
    say('request_files', rows={k: len(v) for k, v in rows.items()}, photos={k: sum(len(photos(r)) for r in v) for k, v in rows.items()},
        missing_photos={k: len(v) for k, v in missing.items()}, missing_examples={k: v[:3] for k, v in missing.items() if v},
        labels=len({r.get('label') for r in rows.get('ft_train', [])}), classes=len(json.loads((ft / 'classes.json').read_text())))
    if any(missing.values()):
        say('STOP', reason='photo files referenced by the request files are missing'); return

    env = dict(os.environ, PYTHONPATH=str(CODE), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1')
    shared = ['--model', model, '--device', CONFIG.get('device', 'auto'), '--image-root', data] + (['--dtype', CONFIG['dtype']] if CONFIG.get('dtype') else []) + (['--answer-logits-only'] if CONFIG.get('lean') else [])

    # the longest training rows (two photos, long text): the smoke test must include them, memory is set by the longest row
    train_rows = rows.get('ft_train', [])
    longest = sorted(range(len(train_rows)), key=lambda i: -size_proxy(train_rows[i]))[:4]
    if 'encode' in CONFIG['stages']:
        try:
            sys.path.insert(0, str(CODE))
            from k12 import q38_finetune as q
            from transformers import AutoProcessor
            t = time.time(); processor = AutoProcessor.from_pretrained(str(model)); q.IMAGE_ROOT = str(data)
            counts = []
            for i in list(range(3)) + longest:
                inputs, at = q.encode(processor, train_rows[i], 900)
                counts.append({'row': i, 'photos': len(photos(train_rows[i])), 'tokens': int(inputs['input_ids'].shape[1]), 'prompt_tokens': int(at),
                               'target_tokens': int(inputs['input_ids'].shape[1] - at), 'keys': sorted(inputs)})
            dev_inputs, dev_at = q.encode(processor, rows['ft_dev'][0])
            say('encode', seconds=round(time.time() - t, 1), processor=type(processor).__name__, train_rows=counts, dev_prompt_tokens=int(dev_at),
                prompt_end=processor.tokenizer.decode(dev_inputs['input_ids'][0, -8:]))
            del processor
        except Exception:
            say('encode_failed', error=traceback.format_exc()[-2500:])

    if mode == 'inspect' or (not gpus and CONFIG.get('require_gpu', True)):
        say('STOP', reason='inspection only' if mode == 'inspect' else 'no GPU in this session'); return
    if len(gpus) < CONFIG.get('min_gpus', 2) and CONFIG.get('require_gpu', True):
        say('STOP', reason='fewer GPUs than the 9B model needs in float16 (two T4s)', gpus=gpus); return
    threading.Thread(target=watch_memory, daemon=True).start()
    plans = list(CONFIG['gpu_memory_plans'])

    def shifted(plan, gpu):
        """A split with 1.5 GiB less of the weights on the GPU that ran out."""
        a, b = plan.split(',')
        return f"{float(a.replace('GiB', '')) + (-1.5 if gpu == 0 else 1.5):g}GiB,{b}"

    plan = plans[0]
    if 'smoke_train' in CONFIG['stages']:
        smoke_file = WORK / 'smoke_train.jsonl'
        picked = longest + [i for i in range(CONFIG['smoke_rows']) if i not in longest][:CONFIG['smoke_rows'] - len(longest)]
        smoke_file.write_text(''.join(json.dumps(train_rows[i]) + '\n' for i in picked))
        tried, ok = [], False
        while len(tried) < 3 and left() > 600:
            tried.append(plan)
            result, lines = run(f'smoke_train_{len(tried)}', ['train', *shared, '--gpu-memory', plan, '--train', smoke_file, '--output', WORK / 'smoke',
                                                             '--classes', ft / 'classes.json', '--epochs', 1, '--accumulation', 2, '--log-every', 1,
                                                             '--save-minutes', 0, '--skip-failed-rows', 2], min(CONFIG['smoke_train_limit'], left() - 420), env)
            summary = train_summary(lines)
            ok = result['returncode'] == 0 and not summary['skipped'] and (WORK / 'smoke' / 'adapter' / 'head.pt').exists()
            say('smoke_train', plan=plan, ok=ok, result=result, summary=summary)
            if ok:
                break
            hit = result['out_of_memory_on_gpu'] or [m for r in summary['skipped'] for m in re.findall(r'GPU (\d+) has a total capacity', r.get('error', ''))]
            if not hit:
                break  # not a memory problem: a new split will not help
            plan = shifted(plan, int(hit[0]))
        say('smoke_train_verdict', ok=ok, plans_tried=tried, plan=plan if ok else None)
        if ok and 'smoke_head' in CONFIG['stages'] and left() > 360:
            bundle = WORK / 'smoke_dev_head.npz'
            result, lines = run('smoke_head', ['head', *shared, '--gpu-memory', plan, '--requests', files['ft_dev'], '--limit', CONFIG['smoke_head_rows'],
                                               '--adapter', WORK / 'smoke' / 'adapter', '--head', WORK / 'smoke' / 'adapter' / 'head.pt', '--output', bundle,
                                               '--log-every', 1, '--deadline', T0 + CONFIG['job_seconds'] - 120], min(CONFIG['smoke_head_limit'], left() - 90), env)
            marks = [r for r in lines if r.get('stage') == 'head']
            per_row = [round(b['seconds'] - a['seconds'], 2) for a, b in zip(marks, marks[1:])]
            extra = {}
            try:
                import numpy as np
                z = np.load(bundle, allow_pickle=False)
                extra = {'rows': int(len(z['ids'])), 'p_shape': list(z['p'].shape), 'p_finite': bool(np.isfinite(z['p']).all()),
                         'p_row_sums': [round(float(x), 4) for x in z['p'].sum(1)[:4]], 'final_ok': [round(float(x), 3) for x in z['final_ok'][:8]],
                         'top_codes': [str(z['classes'][j]) for j in z['p'].argmax(1)[:8]]}
            except Exception:
                extra = {'bundle_error': traceback.format_exc()[-600:]}
            say('smoke_head', result=result, seconds_per_row=per_row, median=sorted(per_row)[len(per_row) // 2] if per_row else None, bundle=extra)

    adapter = WORK / 'ft' / 'adapter'
    if 'train' in CONFIG['stages']:
        reserve = CONFIG['after_train_reserve_seconds']  # development scoring (model load + rows) and the end of the job
        minutes = min(CONFIG['train_minutes'], (left() - reserve - CONFIG['load_allowance_seconds']) / 60)
        if minutes < CONFIG.get('min_train_minutes', 10):
            say('STOP', reason='no time left to train', minutes=minutes); return
        result, lines = run('train', ['train', *shared, '--gpu-memory', plan, '--train', files['ft_train'], '--output', WORK / 'ft', '--classes', ft / 'classes.json',
                                      '--epochs', CONFIG['epochs'], '--accumulation', CONFIG['accumulation'], '--log-every', CONFIG['log_every'],
                                      '--max-minutes', round(minutes, 1), '--save-minutes', CONFIG['save_minutes'], '--skip-failed-rows', CONFIG['skip_failed_rows'],
                                      *CONFIG.get('train_extra', [])], minutes * 60 + CONFIG['load_allowance_seconds'] + 600, env)
        say('train', plan=plan, max_minutes=round(minutes, 1), result=result, summary=train_summary(lines),
            report=json.loads((WORK / 'ft' / 'train_report.json').read_text()) if (WORK / 'ft' / 'train_report.json').exists() else None,
            adapter_saved=(adapter / 'head.pt').exists(), adapter_from='end of training' if result['returncode'] == 0 else 'last periodic save (training did not finish)')
    elif mode == 'test':
        adapter = found['adapter']

    def score(name, requests, bundle, reserve_after):
        if not adapter or not (Path(adapter) / 'head.pt').exists():
            say(name + '_skipped', reason='no adapter with a classification layer'); return None
        if left() < reserve_after + 300:
            say(name + '_skipped', reason='no time left', left=round(left())); return None
        deadline = T0 + CONFIG['job_seconds'] - reserve_after
        result, lines = run(name, ['head', *shared, '--gpu-memory', plan, '--requests', requests, '--adapter', adapter, '--head', Path(adapter) / 'head.pt',
                                   '--output', bundle, '--log-every', 25, '--deadline', deadline,
                                   *(['--limit', CONFIG['score_limit']] if CONFIG.get('score_limit') else [])], left() - reserve_after + 60, env)
        marks = [r for r in lines if r.get('stage') in ('head', 'head_done')]
        speed = round((marks[-1]['seconds'] - marks[0]['seconds']) / max(1, marks[-1]['rows'] - marks[0]['rows']), 3) if len(marks) > 1 else None
        say(name, result=result, rows_scored=marks[-1]['rows'] if marks else 0, seconds_per_row=speed, bundle=str(bundle), bundle_exists=Path(bundle).exists())
        return bundle if Path(bundle).exists() else None

    if 'dev_head' in CONFIG['stages']:
        bundle = score('dev_head', files['ft_dev'], WORK / 'dev_head.npz', 150)
        if bundle:
            try:
                import csv
                with open(data / 'train.csv', newline='') as handle:
                    truth = {r['trace_id']: r['root_code'] for r in csv.DictReader(handle)}
                say('dev_score', note='alone, classification layer, all development rows scored', **map3(bundle, truth))
            except Exception:
                say('dev_score_failed', error=traceback.format_exc()[-1500:])
    if 'test_head' in CONFIG['stages']:
        requests = files['ft_test_1']
        if ORDER:  # least-confident rows of the current system first, so a time-limited run covers the rows that matter most
            rank = {tid: k for k, tid in enumerate(ORDER)}
            ordered = sorted(rows['ft_test_1'], key=lambda r: rank.get(r['trace_id'], len(rank)))
            requests = WORK / 'test_requests_ordered.jsonl'
            requests.write_text(''.join(json.dumps(r) + '\n' for r in ordered))
            say('test_order', rows=len(ordered), ranked=sum(r['trace_id'] in rank for r in ordered), first=[r['trace_id'] for r in ordered[:3]])
        score('test_head', requests, WORK / 'test_head.npz', 150)
        if ORDER and requests.exists():
            requests.unlink()  # prompts only; no need to keep a copy in the job output


try:
    main()
except Exception:
    say('FAILED', error=traceback.format_exc()[-3000:])
STATUS['peak_gpu_mib'] = {k: v for k, v in PEAK.items() if k != 'now'}
say('done', total_seconds=round(time.time() - T0, 1))
