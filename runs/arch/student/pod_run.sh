#!/bin/sh
# Pod-side: install, fetch Qwen3-8B, 200-step smoke with an automatic go/no-go, then the full epoch + dev predict.
# Usage: pod_run.sh <pod $/hr>.  Progress in /workspace/runs/cand.status; test predict is launched separately after the dev gates.
RATE=$1; ST=/workspace/runs/cand.status; mkdir -p /workspace/runs
cd /workspace
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub hf_xet && \
echo PIP_DONE > $ST && /workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', local_dir='/workspace/models/qwen3-8b', max_workers=16)" && \
echo MODEL_DONE > $ST || { echo "exit setup-failed" > $ST; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/workspace/venv/bin/python; D=/workspace/data/v5; M=/workspace/models/qwen3-8b
(grep '"phase": "a"' $D/cand_train.jsonl | head -600; grep '"phase": "b"' $D/cand_train.jsonl | head -200) > $D/smoke800.jsonl
head -n 20 $D/cand_dev.jsonl > $D/dev20.jsonl
echo SMOKE_START > $ST
timeout 45m $PY -m k12.candidate_student train --model $M --train $D/smoke800.jsonl --data /workspace/data/raw \
  --output /workspace/runs/cand_smoke --batch-size 4 --log-every 10 --gen-batch-size 32 --no-save-adapter \
  --predict $D/dev20.jsonl > /workspace/runs/cand_smoke.log 2>&1
GATE=$($PY - "$RATE" <<'EOF'
import json, sys
rate = float(sys.argv[1])
m = json.load(open('/workspace/runs/cand_smoke/predict_dev20/predict_metrics.json'))
logs = [json.loads(l) for l in open('/workspace/runs/cand_smoke.log') if l.startswith('{') and '"stage": "train"' in l]
rps = logs[-1]['rows_per_s']
cost = 8505 / rps / 3600 * rate
ok = m['parse_rate'] >= .95 and cost < 4.
print(json.dumps({'parse_rate': m['parse_rate'], 'rows_per_s': rps, 'epoch_usd': round(cost, 2), 'go': ok}))
EOF
)
echo "$GATE" > /workspace/runs/cand_smoke_gate.json
case "$GATE" in *'"go": true'*) ;; *) echo "exit smoke-gate-failed" > $ST; exit 1;; esac
echo FULL_START > $ST
timeout 6h $PY -m k12.candidate_student train --model $M --train $D/cand_train.jsonl --data /workspace/data/raw \
  --output /workspace/runs/cand_full --batch-size 4 --log-every 25 --save-every 500 --gen-batch-size 32 \
  --interim-predict $D/cand_dev.jsonl --interim-at .5 --interim-limit 100 \
  --predict $D/cand_dev.jsonl > /workspace/runs/cand_full.log 2>&1
echo "exit $?" > $ST
