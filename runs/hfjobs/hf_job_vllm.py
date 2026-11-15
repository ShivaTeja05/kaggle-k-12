# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm==0.33.0", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "pillow", "numpy",
#                 "flash-linear-attention", "ninja", "safetensors"]
# ///
"""Hugging Face Job: merge the fine-tuned adapter into Qwen3.8-27B, serve it with vLLM, and collect its written answers.

Mounts: /model (hf://models/Qwen/Qwen3.8-27B), /inputs (q38_finetune.py, vlm_zero_shot_client.py, request files),
/out (read-write bucket holding the adapter from the training job). Environment: ADAPTER, REQUESTS (space separated),
RUN, SAMPLES, DEADLINE_MINUTES. Answers go to /out/<RUN>/written_<request file name>.
"""
import json, os, subprocess, sys, time, urllib.request

T0 = time.time(); RUN = os.environ.get('RUN', 'ft1'); OUT = f'/out/{RUN}'; os.makedirs(OUT, exist_ok=True)
DEADLINE = T0 + 60 * float(os.environ.get('DEADLINE_MINUTES', '40')); BIN = os.path.dirname(sys.executable)
ADAPTER = os.environ.get('ADAPTER', '/out/ft1/ft/adapter'); MERGED = '/tmp/q38ft'


def say(stage, **kw):
    row = {'stage': stage, 'minutes': round((time.time() - T0) / 60, 1), **kw}
    print(json.dumps(row, default=str), flush=True)
    with open(f"{OUT}/vllm_status_{os.environ.get('TAG', 'written')}.jsonl", 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')


def sh(name, cmd, minutes, **kw):
    say(name + '_start')
    try:
        r = subprocess.run(cmd, timeout=60 * minutes, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    except subprocess.TimeoutExpired:
        say(name + '_timeout'); return False, ''
    open(f"{OUT}/{os.environ.get('TAG', 'written')}_{name}.log", 'w').write(r.stdout)
    say(name + ('_done' if r.returncode == 0 else '_FAILED'), tail=r.stdout[-700:])
    return r.returncode == 0, r.stdout


import torch, vllm, transformers
say('environment', torch=torch.__version__, vllm=vllm.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0))
if os.environ.get('MERGE', '1') == '1':  # MERGE=0 serves the untrained model as it is
    ok, text = sh('merge', [sys.executable, '/inputs/q38_finetune.py', 'merge', '--model', '/model', '--adapter', ADAPTER, '--output', MERGED], 25)
    if not (ok and 'K12_Q38_MERGE_COMPLETE' in text):
        say('job_end', reason='merge failed'); sys.exit(1)
else:
    MERGED = '/model'
sh('kernels', [os.path.join(BIN, 'vllm'), 'download-kernels'], 10, env={**os.environ, 'PATH': BIN + ':' + os.environ['PATH']})
say('serve_start')
server = subprocess.Popen([os.path.join(BIN, 'vllm'), 'serve', MERGED, '--served-model-name', 'q38', '--port', '8000', '--max-model-len', os.environ.get('MAXLEN', '16384'),
                           '--gpu-memory-utilization', '0.93', '--reasoning-parser', 'qwen3', '--limit-mm-per-prompt', '{"image": 2}',
                           '--max-num-seqs', '32', '--seed', '2026'], stdout=open(f"{OUT}/serve_{os.environ.get('TAG', 'written')}.log", 'w'), stderr=subprocess.STDOUT,
                          env={**os.environ, 'PATH': BIN + ':' + os.environ['PATH']})
up = False
while time.time() < DEADLINE - 180 and server.poll() is None:
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); up = True; break
    except Exception:
        time.sleep(5)
if not up:
    say('serve_FAILED', tail=open(f"{OUT}/serve_{os.environ.get('TAG', 'written')}.log").read()[-1500:]); server.kill(); say('job_end', reason='server did not start'); sys.exit(1)
say('serve_ready')
for req in os.environ.get('REQUESTS', '/inputs/ft_dev.jsonl').split():
    name = os.path.basename(req)
    tag = os.environ.get('TAG', 'written')
    sh(f'answers_{tag}_' + name.replace('.jsonl', ''),
       [sys.executable, '/inputs/vlm_zero_shot_client.py', '--requests', req, '--out', f'{OUT}/{tag}_{name}', '--mode', os.environ.get('THINK', 'nothink'),
        '--effort', os.environ.get('EFFORT', 'medium'), '--n', os.environ.get('SAMPLES', '3'), '--max-tokens', os.environ.get('MAXTOK', '1200'),
        '--workers', os.environ.get('WORKERS', '24'), '--deadline', str(DEADLINE - 60)], (DEADLINE - time.time()) / 60)
server.kill(); say('job_end', reason='finished')
