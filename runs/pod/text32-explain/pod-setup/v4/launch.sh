#!/bin/sh
# Mac-side: stage data/code, create one A100 pod, start the spend watchdog, upload, launch run.sh.
# Usage: launch.sh <subset jsonl path> <explanation dataset jsonl path>   (run from the repo root)
set -eu
SUBSET=$1; NAME=$(basename "$SUBSET"); DATASET=$2
PY=.venv/bin/python; KEY=$HOME/.runpod/ssh/runpodctl-ssh-key
HERE=runs/pod/text32-explain/pod-setup/v4
ROWS=$(grep -c . "$SUBSET")
# Training at ~2 s/row plus ~2,200 s of dev/test scoring, with 25% headroom.
LIMIT=$(( (1199 + ROWS) * 2 * 125 / 100 + 2750 ))
$PY -I -c "
import json, sys, pandas as pd
ids = [json.loads(l)['trace_id'] for l in open('$SUBSET')]
dev = set(json.load(open('data/v3/dev_ids.json'))); train = set(pd.read_csv('data/raw/train.csv').trace_id)
assert not (set(ids) & dev) and not (set(ids) & train), 'subset overlaps train/dev'
print('subset ok', len(ids), 'rows')"
echo "timeout ${LIMIT}s"
STAGE=$(mktemp -d); mkdir -p $STAGE/data/competition_data $STAGE/k12/src $STAGE/runs
cp data/raw/*.csv data/pseudo/x/competition_data/unlabelled.csv "$SUBSET" $STAGE/data/competition_data/
cp "$DATASET" runs/pod/tx/tx_dev.json runs/pod/tx/tx_test.json $STAGE/data/
cp -R src/k12 $STAGE/k12/src/; find $STAGE/k12 -name __pycache__ -prune -exec rm -rf {} +
cp runs/pod/text32-explain/pod-setup/get32.py $HERE/run.sh $STAGE/
touch runs/heartbeat
PYTHONPATH=src:scripts $PY -I scripts/runpod_pod.py create --gpu-types "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --attempts 20 | tail -1 > $HERE/pod.json
POD=$($PY -c "import json; print(json.load(open('$HERE/pod.json'))['id'])"); echo "pod $POD"
pgrep -f "runpod_pod.py watchdog" > /dev/null || { PYTHONPATH=src:scripts nohup $PY -I scripts/runpod_pod.py watchdog > runs/watchdog_32v4.log 2>&1 & }
for i in $(seq 1 40); do
  ADDR=$(PYTHONPATH=src:scripts $PY -I scripts/runpod_pod.py status 2>/dev/null | $PY -c "
import sys, json, re
text = sys.stdin.read(); pods = re.findall(r'^\{\n.*?^\}', text, re.S | re.M)
for p in map(json.loads, pods):
    if p['id'] == '$POD' and p.get('runtime'):
        for port in p['runtime']['ports'] or []:
            if port['privatePort'] == 22 and port['isIpPublic']: print(port['ip'], port['publicPort'])")
  [ -n "$ADDR" ] && ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i $KEY -p ${ADDR#* } root@${ADDR% *} true 2>/dev/null && break
  ADDR=""; sleep 15
done
[ -n "$ADDR" ] || { echo "pod never reachable over SSH"; exit 1; }
echo "$ADDR" > $HERE/addr.txt; echo "ssh root@${ADDR% *} -p ${ADDR#* }"
touch runs/heartbeat
(cd $STAGE && COPYFILE_DISABLE=1 tar czf - data k12 get32.py run.sh runs) | ssh -o ConnectTimeout=20 -i $KEY -p ${ADDR#* } root@${ADDR% *} \
  "mkdir -p /workspace && cd /workspace && tar xzf - 2>/dev/null && chmod +x run.sh && (setsid nohup /workspace/run.sh $NAME $LIMIT $(basename "$DATASET") > /workspace/runs/run.log 2>&1 < /dev/null &) && echo launched"
rm -rf $STAGE
