"""Detached guard for the pointwise-ranker pod. It is not tied to the Claude session, so it keeps working if the
session restarts. Once a minute: if the pod is recorded as terminated, stop. If the session's collector is not running,
become the collector (copy results, terminate the pod when the job ends or at the spend limit).
Usage (from the repo root): python runs/deep/pointwise/guard.py POD HOST PORT STATUS_FILE OUT_DIR LIMIT_USD"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
POD, HOST, PORT, STATUS, OUT, LIMIT = sys.argv[1:7]
ARGS = [POD, HOST, PORT, STATUS, OUT, LIMIT, '1']
if os.fork():            # detach: the parent returns at once, the child gets its own session
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)
log = open('runs/deep/pointwise/guard.log', 'a', buffering=1)
say = lambda *a: print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), *a, file=log)
say('guard started, pid', os.getpid())
while True:
    pod = next(x for x in json.load(open('reports/runpod_budget.json'))['pods'] if x['id'] == POD)
    if pod.get('terminated_at'):
        say('pod is terminated; guard exits'); break
    alive = subprocess.run(['pgrep', '-f', f'collect.py {POD}'], capture_output=True).returncode == 0
    if not alive:
        say('no collector running: starting one')
        r = subprocess.run(['caffeinate', '-i', sys.executable, '-I', 'runs/deep/stageB/collect.py', *ARGS], stdout=log, stderr=log)
        say('collector ended with code', r.returncode)
    time.sleep(60)
