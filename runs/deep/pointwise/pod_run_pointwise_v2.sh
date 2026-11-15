#!/bin/sh
# Ranker v2 (approved by the user on 13 Nov 2026): continue the saved Qwen3-14B pointwise adapter on Claude-labelled rows the
# first run did not see, plus one more pass over the teacher rows.
# Usage: pod_run_pointwise_v2.sh <hf model id, e.g. Qwen/Qwen3-14B>
# Stages in /workspace/runs/pw2.status: SETUP, SMOKE, TRAIN, GATE, TEST, then "exit 0" or "exit <reason>".
# Test rows are scored only if the ranker alone reaches 0.725 on the 301 dev rows (first run: 0.7087).
HF=$1; ST=/workspace/runs/pw2.status; R=/workspace/runs; D=/workspace/data/v8; A=/workspace/adapter_v1
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
# 1. the uploaded adapter must reproduce the first run's scores on the first 20 dev rows
timeout 15m $PY -m k12.pointwise_ranker predict --model $M --adapter $A --input $D/pw_dev20.jsonl --data /workspace/data/raw \
  --output $R/pw2_check > $R/pw2_check.log 2>&1 || { echo "exit adapter-check-failed" > $ST; exit 1; }
$PY - > $R/pw2_check.json <<'EOF'
import json
import numpy as np
new = np.load('/workspace/runs/pw2_check/raw_scores.npy'); old = np.load('/workspace/data/v8/v1_dev_raw_scores.npy')[:len(new)]
m = np.isfinite(new)
same = bool((m == np.isfinite(old)).all())
diff = np.abs(new[m] - old[m]) if same else np.array([9.])
top = float(np.mean(np.where(m, new, -1e9).argmax(1) == np.where(np.isfinite(old), old, -1e9).argmax(1)))
print(json.dumps({'same_candidates': same, 'mean_abs_diff': round(float(diff.mean()), 4), 'max_abs_diff': round(float(diff.max()), 4),
                  'top1_agree': top, 'ok': bool(same and diff.mean() < .3 and top >= .85)}))
EOF
case "$(cat $R/pw2_check.json)" in *'"ok": true'*) ;; *) echo "exit adapter-check-failed" > $ST; exit 1;; esac
# 2. a short continuation must run end to end
timeout 20m $PY -m k12.pointwise_ranker train --model $M --train $D/pw_train_v2.jsonl --data /workspace/data/raw --limit 80 --claude-rows -1 \
  --init-adapter $A --lr 5e-5 --predict $D/pw_dev20.jsonl --output $R/pw2_smoke --no-save-adapter --log-every 5 > $R/pw2_smoke.log 2>&1
grep -q K12_POINTWISE_TRAIN_COMPLETE $R/pw2_smoke.log || { echo "exit smoke-failed" > $ST; exit 1; }
echo TRAIN > $ST
timeout 6h $PY -m k12.pointwise_ranker train --model $M --train $D/pw_train_v2.jsonl --data /workspace/data/raw --claude-rows -1 \
  --init-adapter $A --lr 5e-5 --predict $D/pw_dev.jsonl --output $R/pw2_full --log-every 25 > $R/pw2_full.log 2>&1 || { echo "exit train-failed" > $ST; exit 1; }
echo GATE > $ST
$PY - > $R/pw2_gate.json <<'EOF'
import json
import numpy as np
b = np.load('/workspace/runs/pw2_full/predict_pw_dev/predictions.npz', allow_pickle=True); classes = list(b['classes'])
sl = json.load(open('/workspace/data/v8/dev_slices.json'))
truth = {json.loads(l)['trace_id']: json.loads(l)['truth'] for l in open('/workspace/data/v8/pw_dev.jsonl')}
pts = {}
for t, p in zip(b['ids'], b['p']):
    top = [classes[j] for j in np.argsort(-p)[:3]]
    pts[str(t)] = next((1 / (i + 1) for i, c in enumerate(top) if c == truth[str(t)]), 0.)
hard = float(np.mean([v for t, v in pts.items() if sl[t] in ('new_pair', 'unseen_q')]))
score = json.load(open('/workspace/runs/pw2_full/predict_pw_dev/predict_metrics.json'))['map3']  # same scorer as the 0.7087
print(json.dumps({'all': round(score, 4), 'hard_rows': round(hard, 4), 'first_run_all': .7087, 'go': score >= .725}))
EOF
case "$(cat $R/pw2_gate.json)" in *'"go": true'*) ;; *) echo "exit dev-gate-failed" > $ST; exit 1;; esac
echo TEST > $ST
timeout 3h $PY -m k12.pointwise_ranker predict --model $M --adapter $R/pw2_full/adapter --input $D/pw_test.jsonl --data /workspace/data/raw \
  --output $R/pw2_full/predict_pw_test --log-every 50 > $R/pw2_test.log 2>&1
echo "exit $?" > $ST
