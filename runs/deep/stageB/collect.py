"""Watch the Stage B pod from this Mac: copy results back every few minutes, and when the job ends (success,
failure or the spend limit) copy everything, then terminate the pod. Never sends credentials to the pod.

Usage (from the repo root): caffeinate -i .venv/bin/python -I runs/deep/stageB/collect.py POD_ID HOST PORT
"""
from datetime import datetime, timezone
import os
import json
from pathlib import Path
import subprocess
import sys
import time

POD, HOST, PORT = sys.argv[1:4]
# Optional: status file name on the pod, local output folder, spend limit, and whether the final copy takes weights.
STATUS = sys.argv[4] if len(sys.argv) > 4 else 'v5b.status'
OUT = Path(sys.argv[5] if len(sys.argv) > 5 else 'runs/deep/stageB/pod_out')
LIMIT_USD = float(sys.argv[6]) if len(sys.argv) > 6 else 5.6
WEIGHTS = (sys.argv[7] if len(sys.argv) > 7 else '1') == '1'
POLL, MAX_SSH_FAILS = 180, int(os.environ.get('K12_MAX_SSH_FAILS', '10'))  # consecutive failed polls before giving up
KEY = str(Path.home() / '.runpod/ssh/runpodctl-ssh-key')
SSH = ['ssh', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'BatchMode=yes',
       '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR',  # new pods have new host keys
       '-i', KEY, '-p', PORT,
       f'root@{HOST}']


def say(*a):
    print(datetime.now(timezone.utc).strftime('%H:%M:%S'), *a, flush=True)


def remote(command, timeout=60):
    r = subprocess.run(SSH + [command], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if r.returncode:
        raise RuntimeError(r.stderr.strip()[-200:])
    return r.stdout.strip()


def pull(everything):
    """Copy /workspace/runs here. Weights (adapter, checkpoint) only on the final copy."""
    OUT.mkdir(parents=True, exist_ok=True)
    exclude = '' if everything else "--exclude='*.safetensors' --exclude='*.bin'"
    send = subprocess.Popen(SSH + [f'tar czf - -C /workspace/runs {exclude} .'], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    take = subprocess.run(['tar', 'xzf', '-', '-C', str(OUT)], stdin=send.stdout, capture_output=True,
                          timeout=1800 if everything else 300)
    send.wait()
    if send.returncode or take.returncode:
        raise RuntimeError(f'copy failed ({send.returncode}, {take.returncode})')


def cost():
    b = json.loads(Path('reports/runpod_budget.json').read_text())
    p = next(x for x in b['pods'] if x['id'] == POD)
    if p.get('terminated_at'):
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(p['created_at'])).total_seconds() / 3600 * p['cost_per_hr']


def finish(reason):
    say('ending:', reason)
    for attempt in range(3):
        try:
            pull(everything=WEIGHTS)
            say('final copy done ->', OUT)
            break
        except Exception as exc:  # keep trying; the pod is still billed while we wait
            say('final copy attempt', attempt + 1, 'failed:', exc)
            time.sleep(20)
    else:
        say('FINAL COPY FAILED; the last periodic copy is what we have')
    r = subprocess.run([sys.executable, '-I', 'scripts/runpod_pod.py', 'terminate', '--pod', POD],
                       capture_output=True, text=True)
    say(r.stdout.strip() or r.stderr.strip()[-300:])


fails, last = 0, None
while True:
    spent = cost()
    if spent is None:
        say('pod is already terminated'); break
    try:
        status = remote(f'cat /workspace/runs/{STATUS}')
        fails = 0
    except Exception as exc:
        fails += 1
        say(f'ssh failed ({fails}/{MAX_SSH_FAILS}):', exc)
        if fails >= MAX_SSH_FAILS:  # pod unreachable for ~30 minutes: stop paying for it
            finish('pod unreachable'); break
        time.sleep(POLL); continue
    if status != last:
        say('status:', status, f'| spent ${spent:.2f}'); last = status
    if status.startswith('exit'):
        finish(f'job ended with "{status}"'); break
    if spent >= LIMIT_USD:
        finish(f'spend limit ${LIMIT_USD} reached'); break
    try:
        pull(everything=False)
    except Exception as exc:
        say('periodic copy failed:', exc)
    time.sleep(POLL)
