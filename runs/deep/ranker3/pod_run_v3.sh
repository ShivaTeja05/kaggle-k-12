#!/bin/sh
# Ranker v3 (yes/no judge that reads Qwen3.8's description of the photo), on the proven Qwen3-14B code path.
# Usage: pod_run_v3.sh <hf model id> <train timeout minutes> <unix time after which no test part is started, 0 = no test> [wait|now] [group]
#   wait = start on a pod that is still running stage A (pod_run_q38_jobs.sh): wait for it to end, stop vLLM, reuse its venv.
# Needs /workspace/k12/src (k12 package), /workspace/data/raw/taxonomy_v1.csv, /workspace/data/v7/dev_slices.json and
# /workspace/data/v10/pw_{train,dev,test}.jsonl, then the marker /workspace/data/v10/ready.
# Stages in /workspace/runs/v3.status: WAIT_STAGE_A, SETUP, WAIT_DATA, SMOKE, TRAIN, GATE, TEST, then "exit 0" or "exit <reason>".
HF=$1; TMIN=$2; TEST=$3; WAIT=$4; GROUP=${5:-6}; R=/workspace/runs; ST=$R/v3.status; D=/workspace/data/v10; M=/workspace/models/base
mkdir -p $R; cd /workspace
if [ "$WAIT" = wait ]; then
  echo WAIT_STAGE_A > $ST
  while ! grep -q '^exit' $R/q38j.status 2>/dev/null; do sleep 15; done
  pkill -f "vllm serv[e]"; sleep 10
fi
echo SETUP > $ST
PY=/workspace/venv/bin/python; REUSED=0
if [ -x $PY ]; then
  REUSED=1; $PY -m pip install -q peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 hf_xet > $R/v3_setup.log 2>&1
else
  python -m venv /workspace/venv && $PY -m pip install -q --upgrade pip > $R/v3_setup.log 2>&1 && \
  $PY -m pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 >> $R/v3_setup.log 2>&1 && \
  $PY -m pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub hf_xet >> $R/v3_setup.log 2>&1 || { echo "exit setup-failed" > $ST; exit 1; }
fi
$PY -c "from huggingface_hub import snapshot_download as s; s('$HF', local_dir='$M', max_workers=16)" >> $R/v3_setup.log 2>&1 || { echo "exit download-failed" > $ST; exit 1; }
echo WAIT_DATA > $ST
i=0; while [ ! -f $D/ready ]; do i=$((i+1)); [ $i -gt 360 ] && { echo "exit data-timeout" > $ST; exit 1; }; sleep 10; done
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
ARGS="--model $M --data /workspace/data/raw --max-length 1280"; TRAIN="--claude-rows 0 --group $GROUP"  # --claude-rows is a train-only flag
echo SMOKE > $ST
head -n 12 $D/pw_dev.jsonl > $D/pw_dev12.jsonl
smoke() { timeout 15m $PY -m k12.pointwise_ranker train $ARGS $TRAIN --train $D/pw_train.jsonl --limit 40 --predict $D/pw_dev12.jsonl \
  --output $R/v3_smoke --no-save-adapter --log-every 5 > $R/v3_smoke.log 2>&1; grep -q K12_POINTWISE_TRAIN_COMPLETE $R/v3_smoke.log; }
if ! smoke; then
  [ $REUSED -eq 1 ] || { echo "exit smoke-failed" > $ST; exit 1; }
  # the vLLM environment did not work for training: build the environment ranker v1 used and try once more
  cp $R/v3_smoke.log $R/v3_smoke_reused_env.log
  python -m venv /workspace/venv_pw && PY=/workspace/venv_pw/bin/python && $PY -m pip install -q --upgrade pip >> $R/v3_setup.log 2>&1 && \
  $PY -m pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 >> $R/v3_setup.log 2>&1 && \
  $PY -m pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub >> $R/v3_setup.log 2>&1 || { echo "exit setup2-failed" > $ST; exit 1; }
  smoke || { echo "exit smoke-failed" > $ST; exit 1; }
fi
echo TRAIN > $ST
timeout ${TMIN}m $PY -m k12.pointwise_ranker train $ARGS $TRAIN --train $D/pw_train.jsonl --predict $D/pw_dev.jsonl \
  --output $R/v3_full --log-every 25 > $R/v3_full.log 2>&1 || { echo "exit train-failed" > $ST; exit 1; }
echo GATE > $ST
$PY - > $R/v3_gate.json <<'PYEOF'
import json
import numpy as np
b = np.load('/workspace/runs/v3_full/predict_pw_dev/predictions.npz', allow_pickle=True); classes = list(b['classes'])
sl = json.load(open('/workspace/data/v7/dev_slices.json'))
truth = {json.loads(l)['trace_id']: json.loads(l)['truth'] for l in open('/workspace/data/v10/pw_dev.jsonl')}
pts = {}
for t, p in zip(b['ids'], b['p']):
    top = [classes[j] for j in np.argsort(-p)[:3]]
    pts[str(t)] = next((1 / (i + 1) for i, c in enumerate(top) if c == truth[str(t)]), 0.)
part = lambda keys: round(float(np.mean([v for t, v in pts.items() if sl[t] in keys])), 4)
print(json.dumps({'rows': len(pts), 'all': round(float(np.mean(list(pts.values()))), 4), 'hard_rows': part(('new_pair', 'unseen_q')),
                  'seen_pair': part(('seen_pair',)), 'none': part(('none',)) if any(v == 'none' for v in sl.values()) else None}))
PYEOF
# Test rows come in parts, least-confident rows first (pw_test_1.jsonl ...); each part is saved before the next starts.
if [ "$TEST" != 0 ]; then
  echo TEST > $ST
  for F in $D/pw_test_*.jsonl; do
    [ "$(date -u +%s)" -ge "$TEST" ] && break
    N=$(basename $F .jsonl)
    timeout 40m $PY -m k12.pointwise_ranker predict $ARGS --adapter $R/v3_full/adapter --input $F \
      --output $R/v3_full/predict_$N --log-every 50 >> $R/v3_test.log 2>&1 || break
  done
fi
echo "exit 0" > $ST
