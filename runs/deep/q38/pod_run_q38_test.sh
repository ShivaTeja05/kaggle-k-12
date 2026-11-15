#!/bin/sh
# Qwen3.8-27B (FP8, thinking, effort medium: the settings validated on dev) on the least-confident test photos.
# Approved by the user on 14 Nov 2026: two RTX 6000 Ada pods, about $7.80, working limit $9.50 across both.
# Usage: pod_run_q38_test.sh <unix time after which no new request is started> <part numbers, e.g. "1 3 5 7">
# Request files arrive as /workspace/data/part_N.jsonl with a marker part_N.done once their checksum is verified.
# Stages in /workspace/runs/q38t.status: SETUP, DOWNLOAD, SERVE_FP8, RUN_PART_N ..., exit ...
DEADLINE=$1; PARTS=$2; R=/workspace/runs; ST=$R/q38t.status; D=/workspace/data; PY=/workspace/venv/bin/python
mkdir -p $R $D; cd /workspace; echo SETUP > $ST
if [ ! -x /workspace/venv/bin/vllm ]; then
python -m venv /workspace/venv && $PY -m pip install -q -U pip uv > $R/setup.log 2>&1 && \
{ /workspace/venv/bin/uv pip install -q --python $PY vllm==0.33.0 "transformers==5.20.2" ninja --torch-backend auto >> $R/setup.log 2>&1 || \
  $PY -m pip install -q vllm==0.33.0 "transformers==5.20.2" ninja >> $R/setup.log 2>&1; } || { echo "exit setup-failed" > $ST; exit 1; }
fi
# vLLM's sampling kernels are otherwise compiled at start-up, which needs ninja and nvcc
export PATH=/workspace/venv/bin:$PATH
/workspace/venv/bin/vllm download-kernels >> $R/setup.log 2>&1
$PY -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" > $R/versions.txt 2>&1
echo DOWNLOAD > $ST
$PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B-FP8', local_dir='/workspace/models/q38fp8', max_workers=16)" > $R/download.log 2>&1 || { echo "exit download-failed" > $ST; exit 1; }
echo SERVE_FP8 > $ST
/workspace/venv/bin/vllm serve /workspace/models/q38fp8 --served-model-name q38 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.93 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 48 --seed 2026 > $R/serve_fp8.log 2>&1 &
PID=$!; i=0; UP=0
while [ $i -lt 144 ]; do
  curl -s -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1 && { UP=1; break; }
  kill -0 $PID 2>/dev/null || break
  i=$((i+1)); sleep 5
done
[ $UP -eq 1 ] || { echo "exit serve-failed" > $ST; exit 1; }
C=/workspace/vlm_zero_shot_client.py
for N in $PARTS; do
  i=0; while [ ! -f $D/part_$N.done ]; do i=$((i+1)); [ $i -gt 360 ] && { echo "exit upload-timeout" > $ST; exit 1; }; sleep 10; done
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && break
  echo RUN_PART_$N > $ST
  $PY $C --requests $D/part_$N.jsonl --out $R/q38_test_part_$N.jsonl --mode think --effort medium --n 1 --workers 40 --deadline $DEADLINE >> $R/run_think.log 2>&1
done
echo "exit 0" > $ST
