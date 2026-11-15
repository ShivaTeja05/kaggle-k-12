#!/bin/sh
# Qwen3.8-27B (FP8, vLLM 0.33.0) job runner: serves the model once, then runs the request files listed in a jobs file
# one after another. Used for the lean second-opinion plan approved by the user on 14 Nov 2026.
# Usage: pod_run_q38_jobs.sh <unix time after which no new request is started> <jobs file>
# Jobs file: one job per line, "name mode effort seed max_tokens" (mode think|nothink). Requests are read from
# /workspace/data/<name>.jsonl once /workspace/data/<name>.done exists; answers go to /workspace/runs/<name>.jsonl.
# Parallel requests per job: the number in /workspace/workers when the job starts (default 40; 20 keeps the KV cache below full).
# Stages in /workspace/runs/q38j.status: SETUP, DOWNLOAD, SERVE_FP8, RUN_<name> ..., exit ...
DEADLINE=$1; JOBS=$2; R=/workspace/runs; ST=$R/q38j.status; D=/workspace/data; PY=/workspace/venv/bin/python
mkdir -p $R $D; cd /workspace; echo SETUP > $ST
if [ ! -x /workspace/venv/bin/vllm ]; then
python -m venv /workspace/venv && $PY -m pip install -q -U pip uv > $R/setup.log 2>&1 && \
{ /workspace/venv/bin/uv pip install -q --python $PY vllm==0.33.0 "transformers==5.20.2" ninja --torch-backend auto >> $R/setup.log 2>&1 || \
  $PY -m pip install -q vllm==0.33.0 "transformers==5.20.2" ninja >> $R/setup.log 2>&1; } || { echo "exit setup-failed" > $ST; exit 1; }
fi
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
while read NAME MODE EFFORT SEED MAXTOK; do
  [ -z "$NAME" ] && continue
  i=0; while [ ! -f $D/$NAME.done ]; do i=$((i+1)); [ $i -gt 360 ] && { echo "exit upload-timeout" > $ST; exit 1; }; sleep 10; done
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && break
  echo RUN_$NAME > $ST
  $PY $C --requests $D/$NAME.jsonl --out $R/$NAME.jsonl --mode $MODE --effort $EFFORT --seed $SEED --max-tokens $MAXTOK --n 1 --workers $(cat /workspace/workers 2>/dev/null || echo 40) --deadline $DEADLINE >> $R/run_$NAME.log 2>&1 < /dev/null
done < $JOBS
echo "exit 0" > $ST
