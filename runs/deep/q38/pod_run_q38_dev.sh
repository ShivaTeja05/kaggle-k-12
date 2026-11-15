#!/bin/sh
# Zero-shot Qwen3.8-27B on the 301 development photos (training photos only), thinking off then on.
# Approved by the user on 14 Nov 2026: one RTX 6000 Ada, about $1, working limit $1.50. No test data, no training.
# Usage: pod_run_q38_dev.sh <unix time after which no new request is started>
# Stages in /workspace/runs/q38.status: SETUP, DOWNLOAD, SERVE_FP8, (DOWNLOAD_INT4, SERVE_INT4), WAIT_REQUESTS, RUN_NOTHINK, RUN_THINK, exit ...
DEADLINE=$1; R=/workspace/runs; ST=$R/q38.status; D=/workspace/data; PY=/workspace/venv/bin/python
mkdir -p $R $D; cd /workspace; echo SETUP > $ST
if [ ! -x /workspace/venv/bin/vllm ]; then  # skipped when the script is restarted on a pod that is already set up
python -m venv /workspace/venv && $PY -m pip install -q -U pip uv > $R/setup.log 2>&1 && \
{ /workspace/venv/bin/uv pip install -q --python $PY -U vllm "transformers>=5.8.0" ninja --torch-backend auto >> $R/setup.log 2>&1 || \
  $PY -m pip install -q -U vllm "transformers>=5.8.0" ninja >> $R/setup.log 2>&1; } || { echo "exit setup-failed" > $ST; exit 1; }
fi
# vLLM's sampling kernels are otherwise compiled at start-up, which needs ninja and nvcc; the first start failed on that
export PATH=/workspace/venv/bin:$PATH
[ -d /usr/local/cuda/bin ] && export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
/workspace/venv/bin/vllm download-kernels >> $R/setup.log 2>&1
$PY -c "import vllm, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))" > $R/versions.txt 2>&1
fetch() { $PY -c "from huggingface_hub import snapshot_download as s; s('$1', local_dir='$2', max_workers=16)" >> $R/download.log 2>&1; }
serve() {  # $1 model folder, $2 log name; returns 0 when the server answers, 1 when it died or took over 12 minutes
  /workspace/venv/bin/vllm serve $1 --served-model-name q38 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.93 \
    --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 48 --seed 2026 > $R/$2 2>&1 &
  PID=$!; i=0
  while [ $i -lt 144 ]; do
    curl -s -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1 && return 0
    kill -0 $PID 2>/dev/null || return 1
    i=$((i+1)); sleep 5
  done
  kill $PID 2>/dev/null; return 1
}
echo DOWNLOAD > $ST
fetch Qwen/Qwen3.8-27B-FP8 /workspace/models/q38fp8 || { echo "exit download-failed" > $ST; exit 1; }
echo SERVE_FP8 > $ST; BUILD=fp8
if ! serve /workspace/models/q38fp8 serve_fp8.log; then
  pkill -f "vllm serve" 2>/dev/null; sleep 5; rm -rf /workspace/models/q38fp8
  echo DOWNLOAD_INT4 > $ST
  fetch RedHatAI/Qwen3.8-27B-INT4 /workspace/models/q38int4 || { echo "exit download-failed" > $ST; exit 1; }
  echo SERVE_INT4 > $ST; BUILD=int4
  serve /workspace/models/q38int4 serve_int4.log || { echo "exit serve-failed" > $ST; exit 1; }
fi
echo $BUILD > $R/build.txt
echo WAIT_REQUESTS > $ST
i=0; while [ ! -f $D/.uploaded ]; do i=$((i+1)); [ $i -gt 180 ] && { echo "exit upload-timeout" > $ST; exit 1; }; sleep 10; done
C=/workspace/vlm_zero_shot_client.py; Q=$D/q38_dev_requests.jsonl
# four rows first, one at a time, so a broken setup shows up in the log before the full run
$PY $C --requests $Q --out $R/smoke_nothink.jsonl --mode nothink --n 1 --limit 4 --workers 1 > $R/smoke.log 2>&1
echo RUN_NOTHINK > $ST
$PY $C --requests $Q --out $R/q38_dev_nothink.jsonl --mode nothink --n 3 --workers 32 --deadline $DEADLINE > $R/run_nothink.log 2>&1
echo RUN_THINK > $ST
$PY $C --requests $Q --out $R/q38_dev_think.jsonl --mode think --effort medium --n 1 --workers 40 --deadline $DEADLINE > $R/run_think.log 2>&1
echo "exit $?" > $ST
