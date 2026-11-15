# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchvision", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "pillow", "numpy",
#                 "flash-linear-attention", "safetensors"]
# ///
"""Hugging Face Job: fine-tune Qwen3.8-27B with the classification layer (PREPARED 15 Nov 2026; needs the owner's yes to run).

Mounts expected: /model (hf://models/Qwen/Qwen3.8-27B, read-only), /inputs (our files: q38_finetune.py, ft_train.jsonl,
ft_dev.jsonl, classes.json, optionally ft_test_*.jsonl), /out (read-write bucket: every result is written here as it is made,
so a job stopped by its time limit loses nothing that was saved).
MODE=smoke  environment report, 12 training rows, 6 dev rows scored, 6 rows of internal states. A few minutes.
MODE=main   training for TRAIN_MINUTES, dev scores, test scores if SCORE_TEST=1, then the untrained model's internal states
            for train+dev while time remains before DEADLINE_MINUTES (measured from the start of the job).
Launch (from the repo root), for example:
  hf jobs uv run --flavor a100-large --timeout 20m -e MODE=smoke -v hf://models/Qwen/Qwen3.8-27B:/model \
      -v ./runs/hfjobs/inputs:/inputs -v ./runs/hfjobs/out:/out:rw runs/hfjobs/hf_job.py
"""
import glob, json, os, subprocess, sys, time

T0 = time.time(); MODE = os.environ.get('MODE', 'smoke'); OUT = '/out/' + os.environ.get('RUN', MODE)
TRAIN_MINUTES = float(os.environ.get('TRAIN_MINUTES', '240')); DEADLINE = T0 + 60 * float(os.environ.get('DEADLINE_MINUTES', '15'))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, '/inputs')


def say(stage, **kw):
    row = {'stage': stage, 'minutes': round((time.time() - T0) / 60, 1), **kw}
    print(json.dumps(row, default=str), flush=True)
    with open(f'{OUT}/status.jsonl', 'a') as f:
        f.write(json.dumps(row, default=str) + '\n')


def run(name, args, minutes):
    """One command of the fine-tune script; its log goes to /out; returns True when it printed its completion marker."""
    log = f'{OUT}/{name}.log'; say(name + '_start', args=' '.join(args))
    # the JSON progress lines are echoed to the job's own log so the run can be watched live; everything goes to the file
    proc = subprocess.Popen([sys.executable, '-u', '/inputs/q38_finetune.py', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    limit, lines = time.time() + 60 * minutes, []
    import threading
    def pump():
        with open(log, 'w') as f:
            for line in proc.stdout:
                f.write(line); f.flush(); lines.append(line)
                if line.startswith('{'):
                    print(name, line.rstrip()[:600], flush=True)
    t = threading.Thread(target=pump, daemon=True); t.start()
    while proc.poll() is None and time.time() < limit:
        time.sleep(5)
    if proc.poll() is None:
        proc.kill(); t.join(10); say(name + '_timeout', minutes=minutes); return False
    t.join(30); text = ''.join(lines); ok = 'COMPLETE' in text
    say(name + ('_done' if ok else '_FAILED'), tail=text[-1500:] if not ok else [l for l in text.splitlines() if l.startswith('{')][-2:])
    return ok


import torch, transformers, peft
say('environment', torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
    gpu=torch.cuda.get_device_name(0), gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    model_files=len(os.listdir('/model')), inputs=sorted(os.listdir('/inputs')))
common = ['--model', '/model', '--answer-logits-only']
if MODE == 'smoke':
    run('smoke_train', ['train', *common, '--train', '/inputs/ft_train.jsonl', '--classes', '/inputs/classes.json', '--output', f'{OUT}/smoke',
                        '--limit', '12', '--epochs', '1', '--accumulation', '4', '--save-minutes', '0', '--log-every', '1'], 12)
    if os.path.exists(f'{OUT}/smoke/adapter/head.pt'):
        run('smoke_head', ['head', *common, '--adapter', f'{OUT}/smoke/adapter', '--head', f'{OUT}/smoke/adapter/head.pt',
                           '--requests', '/inputs/ft_dev.jsonl', '--limit', '6', '--output', f'{OUT}/smoke_dev_head.npz', '--log-every', '1'], 6)
    run('smoke_features', ['features', *common, '--requests', '/inputs/ft_dev.jsonl', '--limit', '6', '--output', f'{OUT}/smoke_features.npz', '--log-every', '2'], 6)
elif MODE == 'generate':  # written answers of the fine-tuned model for the dev photos (adapter from an earlier run in /out)
    src = os.environ.get('ADAPTER', '/out/ft1/ft/adapter')
    run('generate', ['generate', *common, '--adapter', src, '--requests', os.environ.get('REQUESTS', '/inputs/ft_dev.jsonl'),
                     '--output', f'{OUT}/dev_written.jsonl', '--max-new-tokens', '700', '--log-every', '5', '--deadline', str(DEADLINE - 120)],
        (DEADLINE - time.time()) / 60 - 1)
else:
    ok = run('train', ['train', *common, '--train', '/inputs/ft_train.jsonl', '--classes', '/inputs/classes.json', '--output', f'{OUT}/ft',
                       '--epochs', os.environ.get('EPOCHS', '2'), '--max-minutes', str(TRAIN_MINUTES), '--save-minutes', '30', '--log-every', '5',
                       '--skip-failed-rows', '20'], TRAIN_MINUTES + 25)
    if os.path.exists(f'{OUT}/ft/adapter/head.pt'):
        run('head_dev', ['head', *common, '--adapter', f'{OUT}/ft/adapter', '--head', f'{OUT}/ft/adapter/head.pt', '--requests', '/inputs/ft_dev.jsonl',
                         '--output', f'{OUT}/dev_head.npz', '--log-every', '50'], 30)
        tests = sorted(glob.glob('/inputs/ft_test_*.jsonl')); left = (DEADLINE - time.time()) / 60 - 3
        if tests and left > 10 and os.environ.get('SCORE_TEST', '0') == '1':
            run('head_test', ['head', *common, '--adapter', f'{OUT}/ft/adapter', '--head', f'{OUT}/ft/adapter/head.pt', '--requests', *tests,
                              '--output', f'{OUT}/test_head.npz', '--log-every', '200', '--deadline', str(DEADLINE - 120)], left)
    # last, because it does not depend on the training: the untrained model's internal states for small classifiers on the Mac
    if (DEADLINE - time.time()) / 60 > 25:
        run('features', ['features', *common, '--requests', '/inputs/ft_train.jsonl', '/inputs/ft_dev.jsonl', '--output', f'{OUT}/features_train_dev.npz',
                         '--log-every', '100', '--deadline', str(DEADLINE - 120)], (DEADLINE - time.time()) / 60 - 2)
say('job_end')
