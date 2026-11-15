#!/bin/sh
# DEEP_PLAN steps 3-4 on one pod (prepared, NOT launched; needs the user's explicit approval).
# Pod-side: install, fetch Qwen3-8B, 100-step smoke with an automatic go/no-go, then the full epoch on v5b with an interim
# abort at 50%, then dev + test predict. Expects /workspace/k12 (src, scripts), /workspace/data/raw and /workspace/data/v5b
# (runs/deep/stageB/data incl. dev_slices.json), copied over SSH from the Mac. No credentials on the pod.
# Usage: pod_run_v5b.sh <pod $/hr>.  Progress in /workspace/runs/v5b.status.
RATE=$1; ST=/workspace/runs/v5b.status; mkdir -p /workspace/runs
cd /workspace
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub hf_xet && \
echo PIP_DONE > $ST && /workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', local_dir='/workspace/models/qwen3-8b', max_workers=16)" && \
echo MODEL_DONE > $ST || { echo "exit setup-failed" > $ST; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/workspace/venv/bin/python; D=/workspace/data/v5b; M=/workspace/models/qwen3-8b
# Smoke: 400 rows (phase a, contrast and phase b), 50 dev rows.
(grep '"phase": "a"' $D/cand_train.jsonl | head -300; grep '"phase": "b"' $D/cand_train.jsonl | head -100) > $D/smoke400.jsonl
head -n 50 $D/cand_dev.jsonl > $D/dev50.jsonl
echo SMOKE_START > $ST
timeout 40m $PY -m k12.candidate_student train --model $M --train $D/smoke400.jsonl --data /workspace/data/raw \
  --output /workspace/runs/v5b_smoke --batch-size 4 --log-every 10 --gen-batch-size 32 --no-save-adapter \
  --predict $D/dev50.jsonl > /workspace/runs/v5b_smoke.log 2>&1
GATE=$($PY - "$RATE" <<'EOF'
import json, math, sys
rate = float(sys.argv[1])
m = json.load(open('/workspace/runs/v5b_smoke/predict_dev50/predict_metrics.json'))
logs = [json.loads(l) for l in open('/workspace/runs/v5b_smoke.log') if l.startswith('{') and '"stage": "train"' in l]
rps = logs[-1]['rows_per_s']
finite = all(math.isfinite(x['loss']) for x in logs)
falling = logs[-1]['loss'] < logs[0]['loss']
cost = 9705 / rps / 3600 * rate
ok = m['parse_rate'] >= .95 and rps >= 1.5 and finite and falling and cost < 4.
print(json.dumps({'parse_rate': m['parse_rate'], 'rows_per_s': rps, 'loss_first': logs[0]['loss'], 'loss_last': logs[-1]['loss'],
                  'epoch_usd': round(cost, 2), 'go': ok}))
EOF
)
echo "$GATE" > /workspace/runs/v5b_smoke_gate.json
case "$GATE" in *'"go": true'*) ;; *) echo "exit smoke-gate-failed" > $ST; exit 1;; esac
echo FULL_START > $ST
timeout 6h $PY -m k12.candidate_student train --model $M --train $D/cand_train.jsonl --data /workspace/data/raw \
  --output /workspace/runs/v5b_full --batch-size 4 --log-every 25 --save-every 500 --gen-batch-size 32 \
  --interim-predict $D/cand_dev.jsonl --interim-at .5 --interim-limit 100 \
  --predict $D/cand_dev.jsonl $D/cand_test.jsonl > /workspace/runs/v5b_full.log 2>&1 &
TRAIN=$!
# Interim abort (DEEP_PLAN step 4): student alone on the first 100 dev rows, new/unseen MAP < .35 or parse rate < .9.
while kill -0 $TRAIN 2>/dev/null; do
  F=$(ls /workspace/runs/v5b_full/interim_step*/predict_metrics.json 2>/dev/null | head -1)
  if [ -n "$F" ] && [ ! -f /workspace/runs/v5b_interim_gate.json ]; then
    $PY - "$F" > /workspace/runs/v5b_interim_gate.json <<'EOF'
import json, sys
import numpy as np
from pathlib import Path
d = Path(sys.argv[1]).parent
m = json.load(open(d / 'predict_metrics.json')); sl = json.load(open('/workspace/data/v5b/dev_slices.json'))
truth = {json.loads(l)['trace_id']: json.loads(l)['truth'] for l in open('/workspace/data/v5b/cand_dev.jsonl')}
b = np.load(d / 'predictions.npz', allow_pickle=True); classes = list(b['classes'])
s = []
for t, p in zip(b['ids'], b['p']):
    if sl[str(t)] in ('new_pair', 'unseen_q'):
        top = [classes[j] for j in np.argsort(-p)[:3]]
        s.append(next((1 / (i + 1) for i, c in enumerate(top) if c == truth[str(t)]), 0.))
nu = float(np.mean(s))
print(json.dumps({'parse_rate': m['parse_rate'], 'new_unseen_map3': round(nu, 4), 'n': len(s),
                  'go': m['parse_rate'] >= .9 and nu >= .35}))
EOF
    case "$(cat /workspace/runs/v5b_interim_gate.json)" in *'"go": true'*) ;; *) kill $TRAIN; echo "exit interim-gate-failed" > $ST; exit 1;; esac
  fi
  sleep 60
done
wait $TRAIN; echo "exit $?" > $ST
