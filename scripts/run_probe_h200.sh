#!/bin/sh
# Zero-shot Qwen3-235B probe on one H200: create (with retries), set up, run, collect, terminate.
# A timer terminates the pod before its cost can reach the user's $15 probe cap.
set -u
cd "$(dirname "$0")/.."
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
PY=.venv/bin/python
CAP=15
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i $KEY"

touch runs/heartbeat
if ! $PY -I scripts/runpod_pod.py create --gpu-count 1 --gpu-types "NVIDIA H200" "NVIDIA H200 NVL" \
    --max-rate 5.29 --disk 280 --attempts 280 > runs/probe235_create.log 2>&1; then
  echo "no H200 became available"; tail -2 runs/probe235_create.log; exit 1
fi
POD=$(tail -1 runs/probe235_create.log | $PY -c "import sys,json; print(json.load(sys.stdin)['id'])")
RATE=$(tail -1 runs/probe235_create.log | $PY -c "import sys,json; print(json.load(sys.stdin)['costPerHr'])")
echo "pod $POD at \$$RATE/hr"

# Hard cap: terminate after CAP/RATE hours minus five minutes, whatever state the probe is in.
LIMIT=$($PY -c "print(int($CAP / $RATE * 3600) - 300)")
( sleep "$LIMIT"; $PY -I scripts/runpod_pod.py terminate --pod "$POD" > /dev/null 2>&1; echo "cap timer fired" ) &
TIMER=$!

ADDR=""
while [ -z "$ADDR" ]; do
  sleep 15
  ADDR=$($PY -I scripts/runpod_pod.py status | $PY -c "
import sys, re, json
for block in re.findall(r'\{\n \"id\": \"$POD\".*?\n\}', sys.stdin.read(), re.S):
    for p in (json.loads(block).get('runtime') or {}).get('ports') or []:
        if p['privatePort'] == 22 and p['isIpPublic']: print(p['ip'], p['publicPort'])")
done
HOST=${ADDR% *}; PORT=${ADDR#* }
echo "ssh -p $PORT root@$HOST"
sleep 20
$SSH -p "$PORT" "root@$HOST" 'mkdir -p /workspace/k12 /workspace/data /workspace/runs' < /dev/null
COPYFILE_DISABLE=1 tar czf - src | $SSH -p "$PORT" "root@$HOST" 'tar xzf - -C /workspace/k12 2>/dev/null'
scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PORT" data/raw/train.csv data/raw/taxonomy_v1.csv runs/pod/tx/tx_dev.json \
  "root@$HOST:/workspace/data/"
scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PORT" scripts/probe_setup.sh "root@$HOST:/workspace/probe_setup.sh"
$SSH -n -p "$PORT" "root@$HOST" 'setsid nohup sh /workspace/probe_setup.sh > /workspace/setup.log 2>&1 < /dev/null & echo started'

until $SSH -p "$PORT" "root@$HOST" 'test -f /workspace/runs/probe235.status' < /dev/null 2>/dev/null; do
  touch runs/heartbeat
  kill -0 "$TIMER" 2>/dev/null || { echo "stopped by cap timer before finishing"; exit 1; }
  sleep 60
done
mkdir -p runs/pod/probe235
scp -q -r -i "$KEY" -P "$PORT" "root@$HOST:/workspace/runs/probe235" runs/pod/ 2>/dev/null
$SSH -p "$PORT" "root@$HOST" 'cat /workspace/runs/probe235.status; tail -5 /workspace/runs/probe235.log; tail -3 /workspace/setup.log' < /dev/null
$PY -I scripts/runpod_pod.py terminate --pod "$POD" | tail -1
kill "$TIMER" 2>/dev/null
cat runs/pod/probe235/metrics.json 2>/dev/null
