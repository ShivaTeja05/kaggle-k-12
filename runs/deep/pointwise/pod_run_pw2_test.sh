#!/bin/sh
# Score the test rows with the saved ranker v2 adapter, so v1 and v2 can be averaged (prepared 13 Nov 2026; needs the user's approval).
# No training. Usage: pod_run_pw2_test.sh <hf model id, e.g. Qwen/Qwen3-14B>
# Stages in /workspace/runs/pw2t.status: SETUP, WAIT_ADAPTER, CHECK, TEST, then "exit 0" or "exit <reason>".
HF=$1; ST=/workspace/runs/pw2t.status; R=/workspace/runs; D=/workspace/data/v8; A=/workspace/adapter_v2
mkdir -p $R; cd /workspace; echo SETUP > $ST
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy huggingface_hub hf_xet && \
/workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$HF', local_dir='/workspace/models/base', max_workers=16)" || { echo "exit setup-failed" > $ST; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/workspace/venv/bin/python; M=/workspace/models/base
# the adapter (514 MB) is uploaded while setup runs; wait for the marker written after its checksum is verified
echo WAIT_ADAPTER > $ST
i=0; while [ ! -f $A/.uploaded ]; do i=$((i+1)); [ $i -gt 240 ] && { echo "exit adapter-upload-timeout" > $ST; exit 1; }; sleep 10; done
echo CHECK > $ST
head -n 20 $D/pw_dev.jsonl > $D/pw_dev20.jsonl
# the uploaded v2 adapter must reproduce its own dev scores on the first 20 dev rows
timeout 15m $PY -m k12.pointwise_ranker predict --model $M --adapter $A --input $D/pw_dev20.jsonl --data /workspace/data/raw \
  --output $R/pw2t_check > $R/pw2t_check.log 2>&1 || { echo "exit adapter-check-failed" > $ST; exit 1; }
$PY - > $R/pw2t_check.json <<'EOF'
import json
import numpy as np
new = np.load('/workspace/runs/pw2t_check/raw_scores.npy'); old = np.load('/workspace/data/v8/v2_dev_raw_scores.npy')[:len(new)]
m = np.isfinite(new)
same = bool((m == np.isfinite(old)).all())
diff = np.abs(new[m] - old[m]) if same else np.array([9.])
top = float(np.mean(np.where(m, new, -1e9).argmax(1) == np.where(np.isfinite(old), old, -1e9).argmax(1)))
print(json.dumps({'same_candidates': same, 'mean_abs_diff': round(float(diff.mean()), 4), 'max_abs_diff': round(float(diff.max()), 4),
                  'top1_agree': top, 'ok': bool(same and diff.mean() < .3 and top >= .85)}))
EOF
case "$(cat $R/pw2t_check.json)" in *'"ok": true'*) ;; *) echo "exit adapter-check-failed" > $ST; exit 1;; esac
echo TEST > $ST
timeout 2h $PY -m k12.pointwise_ranker predict --model $M --adapter $A --input $D/pw_test.jsonl --data /workspace/data/raw \
  --output $R/pw2t/predict_pw_test --log-every 50 > $R/pw2t_test.log 2>&1
echo "exit $?" > $ST
