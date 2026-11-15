# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm==0.33.0", "transformers==5.20.2", "huggingface_hub", "hf_xet", "ninja", "pillow"]
# ///
"""Hugging Face Job: Qwen3.8-Flash-Next-FP8 (180B) zero-shot with thinking on development photos, served by vLLM on 2 GPUs.
Mounts: /inputs (vlm_zero_shot_client.py and request files), /out (bucket). Everything has a deadline measured from the start."""
import json, os, subprocess, sys, time, urllib.request
import shutil
T0 = time.time(); OUT = '/out/flash'; os.makedirs(OUT, exist_ok=True); BIN = os.path.dirname(sys.executable)
VLLM = shutil.which('vllm') or os.path.join(BIN, 'vllm')  # the official vLLM image keeps it in /usr/local/bin
DEADLINE = T0 + 60 * float(os.environ.get('DEADLINE_MINUTES', '27')); ENV = {**os.environ, 'PATH': BIN + ':' + os.environ['PATH']}


def say(stage, **kw):
    row = {'stage': stage, 'minutes': round((time.time() - T0) / 60, 1), **kw}; print(json.dumps(row, default=str), flush=True)
    with open(f'{OUT}/status.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')


import torch, vllm
say('environment', vllm=vllm.__version__, torch=torch.__version__, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    gb=[round(torch.cuda.get_device_properties(i).total_memory / 2**30) for i in range(torch.cuda.device_count())])
from huggingface_hub import snapshot_download
t = time.time(); path = snapshot_download('Qwen/Qwen3.8-Flash-Next-FP8', local_dir='/tmp/flash', max_workers=32)
size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
say('downloaded', gb=round(size / 1e9, 1), seconds=round(time.time() - t), gb_per_s=round(size / 1e9 / (time.time() - t), 2))
if os.environ.get('KERNELS', '1') == '1':  # not needed in the official vLLM image, which ships the kernels and a CUDA compiler
    subprocess.run([VLLM, 'download-kernels'], env=ENV, capture_output=True, timeout=600)
n = torch.cuda.device_count()
base = [VLLM, 'serve', path, '--served-model-name', 'q38', '--port', '8000', '--max-model-len', '32768', '--gpu-memory-utilization', os.environ.get('GPU_UTIL', '0.92'),
        '--reasoning-parser', 'qwen3', '--limit-mm-per-prompt', '{"image": 2}', '--max-num-seqs', os.environ.get('SEQS', '64'), '--seed', '2026', '--tensor-parallel-size', str(n),
        # the job image has no CUDA compiler: switch off the two parts that compile at start-up (all-reduce fusion, GDN prefill)
        '--compilation-config', '{"pass_config": {"fuse_allreduce_rms": false}}', '--gdn-prefill-backend', 'triton']
variants = [base] if n == 1 else [base + ['--enable-expert-parallel', '--moe-backend', 'triton'], base + ['--enable-expert-parallel']]  # one GPU: no parallel flags
up, server = False, None
for k, cmd in enumerate(variants):
    log = f'{OUT}/serve_{k}.log'; say('serve_start', variant=k, extra=cmd[len(base):])
    server = subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT, env=ENV)
    while time.time() < DEADLINE - 240 and server.poll() is None:
        try:
            urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); up = True; break
        except Exception:
            time.sleep(5)
    if up:
        break
    tail = open(log).read()[-1200:]; say('serve_FAILED', variant=k, tail=tail)
    if server.poll() is None:
        server.kill(); break  # still loading at the deadline: no point trying another variant
if not up:
    say('job_end', reason='server did not start'); sys.exit(1)
say('serve_ready')
for req in os.environ.get('REQUESTS', '/inputs/q38_dev_sent158.jsonl').split():
    name = os.path.basename(req); say('answers_start', file=name)
    left = (DEADLINE - time.time()) / 60
    if left < 1.5:
        break
    try:
        r = subprocess.run([sys.executable, '/inputs/vlm_zero_shot_client.py', '--requests', req, '--out', f'{OUT}/flash_{name}', '--mode', 'think',
                            '--effort', os.environ.get('EFFORT', 'medium'), '--n', '1', '--max-tokens', '14000', '--workers', os.environ.get('WORKERS', '40'),
                            '--deadline', str(DEADLINE - 45)], capture_output=True, text=True, timeout=60 * left)
        open(f'{OUT}/client_{name}.log', 'w').write(r.stdout + r.stderr); say('answers_done', file=name, tail=r.stdout[-400:])
    except subprocess.TimeoutExpired:
        say('answers_timeout', file=name)
server.kill(); say('job_end', reason='finished')
