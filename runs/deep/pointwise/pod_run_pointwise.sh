#!/bin/sh
# Pointwise yes/no ranker test (prepared 13 Nov 2026; needs the user's explicit approval before launch).
# Usage: pod_run_pointwise.sh <hf model id, e.g. Qwen/Qwen3-14B> <rows of Claude data, e.g. 2800>
# Stages in /workspace/runs/pw.status: SETUP, SMOKE, TRAIN, GATE, TEST, then "exit 0" or "exit <reason>".
# Trains on fit rows only, scores the 301 dev rows, and scores the test rows only if the ranker alone
# reaches 0.45 on the hard dev rows (the current blend's level there).
HF=$1; CLAUDE=${2:-2800}; ST=/workspace/runs/pw.status; R=/workspace/runs; D=/workspace/data/v7
mkdir -p $R; cd /workspace; echo SETUP > $ST
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub hf_xet && \
/workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$HF', local_dir='/workspace/models/base', max_workers=16)" || { echo "exit setup-failed" > $ST; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/workspace/venv/bin/python; M=/workspace/models/base
echo SMOKE > $ST
head -n 20 $D/pw_dev.jsonl > $D/pw_dev20.jsonl
timeout 20m $PY -m k12.pointwise_ranker train --model $M --train $D/pw_train.jsonl --data /workspace/data/raw --limit 80 --claude-rows 40 \
  --predict $D/pw_dev20.jsonl --output $R/pw_smoke --no-save-adapter --log-every 5 > $R/pw_smoke.log 2>&1
grep -q K12_POINTWISE_TRAIN_COMPLETE $R/pw_smoke.log || { echo "exit smoke-failed" > $ST; exit 1; }
echo TRAIN > $ST
timeout 6h $PY -m k12.pointwise_ranker train --model $M --train $D/pw_train.jsonl --data /workspace/data/raw --claude-rows $CLAUDE \
  --predict $D/pw_dev.jsonl --output $R/pw_full --log-every 25 > $R/pw_full.log 2>&1 || { echo "exit train-failed" > $ST; exit 1; }
echo GATE > $ST
$PY - > $R/pw_gate.json <<'EOF'
import json
import numpy as np
b = np.load('/workspace/runs/pw_full/predict_pw_dev/predictions.npz', allow_pickle=True); classes = list(b['classes'])
sl = json.load(open('/workspace/data/v7/dev_slices.json'))
truth = {json.loads(l)['trace_id']: json.loads(l)['truth'] for l in open('/workspace/data/v7/pw_dev.jsonl')}
pts = {}
for t, p in zip(b['ids'], b['p']):
    top = [classes[j] for j in np.argsort(-p)[:3]]
    pts[str(t)] = next((1 / (i + 1) for i, c in enumerate(top) if c == truth[str(t)]), 0.)
hard = float(np.mean([v for t, v in pts.items() if sl[t] in ('new_pair', 'unseen_q')]))
print(json.dumps({'all': round(float(np.mean(list(pts.values()))), 4), 'hard_rows': round(hard, 4), 'go': hard >= .45}))
EOF
case "$(cat $R/pw_gate.json)" in *'"go": true'*) ;; *) echo "exit dev-gate-failed" > $ST; exit 1;; esac
echo TEST > $ST
timeout 3h $PY -m k12.pointwise_ranker predict --model $M --adapter $R/pw_full/adapter --input $D/pw_test.jsonl --data /workspace/data/raw \
  --output $R/pw_full/predict_pw_test --log-every 50 > $R/pw_test.log 2>&1
echo "exit $?" > $ST
