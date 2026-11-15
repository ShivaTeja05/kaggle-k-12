#!/bin/sh
# Few-shot Qwen3-235B on the test rows, on one Secure Cloud H200: create (with retries), set up, run, collect, terminate.
# A timer terminates the pod before its cost reaches CAP; the probe stops starting chunks 12 minutes before that,
# and rows it never reached stay uniform in test.npz, which leaves them to the other ensemble members.
set -u
cd "$(dirname "$0")/.."
KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
PY=.venv/bin/python
CAP=16
OUT=runs/pod/probe235/fewshot235_test
SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $KEY"
mkdir -p "$OUT"

touch runs/heartbeat
if ! $PY -I scripts/runpod_pod.py create --gpu-count 1 --gpu-types "NVIDIA H200" "NVIDIA H200 NVL" \
    --max-rate 5.29 --disk 280 --attempts 280 > "$OUT/create.log" 2>&1; then
  echo "no H200 became available"; tail -2 "$OUT/create.log"; exit 1
fi
POD=$(tail -1 "$OUT/create.log" | $PY -c "import sys,json; print(json.load(sys.stdin)['id'])")
RATE=$(tail -1 "$OUT/create.log" | $PY -c "import sys,json; print(json.load(sys.stdin)['costPerHr'])")
echo "pod $POD at \$$RATE/hr, created $(date -u +%H:%M)"

LIMIT=$($PY -c "print(int($CAP / $RATE * 3600) - 300)")
( sleep "$LIMIT"; $PY -I scripts/runpod_pod.py terminate --pod "$POD" > /dev/null 2>&1; echo "cap timer fired" ) &
TIMER=$!
DEADLINE=$($PY -c "import time; print(int(time.time() + $LIMIT - 720))")

ADDR=""
while [ -z "$ADDR" ]; do
  sleep 15
  touch runs/heartbeat
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
scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PORT" data/raw/train.csv data/raw/test.csv data/raw/taxonomy_v1.csv \
  runs/pod/tx/tx_test.json "root@$HOST:/workspace/data/"
scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PORT" scripts/probe_test_setup.sh "root@$HOST:/workspace/setup.sh"
echo "$DEADLINE" | $SSH -p "$PORT" "root@$HOST" 'cat > /workspace/runs/probe_deadline'
# setsid in its own session, all streams redirected, so this ssh returns at once.
$SSH -n -p "$PORT" "root@$HOST" 'setsid nohup sh /workspace/setup.sh > /workspace/setup.log 2>&1 < /dev/null & echo started'

while ! $SSH -p "$PORT" "root@$HOST" 'test -f /workspace/runs/fewshot235_test.status' < /dev/null 2>/dev/null; do
  touch runs/heartbeat
  kill -0 "$TIMER" 2>/dev/null || { echo "stopped by cap timer before finishing"; break; }
  # Keep a copy of the partial results in case the pod disappears.
  scp -q -i "$KEY" -P "$PORT" "root@$HOST:/workspace/runs/fewshot235_test/*" "$OUT/" 2>/dev/null
  sleep 120
done
scp -q -i "$KEY" -P "$PORT" "root@$HOST:/workspace/runs/fewshot235_test/*" "$OUT/" 2>/dev/null
scp -q -i "$KEY" -P "$PORT" "root@$HOST:/workspace/runs/fewshot235_test.log" "root@$HOST:/workspace/setup.log" \
  "root@$HOST:/workspace/runs/fewshot235_test.status" "$OUT/" 2>/dev/null
$PY -I scripts/runpod_pod.py terminate --pod "$POD" | tail -1
kill "$TIMER" 2>/dev/null; pkill -f "sleep $LIMIT" 2>/dev/null
echo "finished $(date -u +%H:%M)"; cat "$OUT/fewshot235_test.status" "$OUT/metrics.json" 2>/dev/null
