#!/bin/sh
# Fine-tune Qwen3.8-27B (bf16, LoRA + classification layer) on the teachers' labels, score dev with the classification layer,
# merge, serve with vLLM, score dev by written answers, then (on a go signal) score test.
# PREPARED 15 Nov 2026; NOT approved to run. Needs an 80 GB card (A100 80GB) and about 200 GB of disk.
# Usage: pod_run_q38ft.sh <training minutes> <epochs> <unix time after which no new test part starts> [samples per photo, default 3]
# Needs /workspace/k12/src/k12/q38_finetune.py, /workspace/vlm_zero_shot_client.py and, in /workspace/data/v11:
#   ft_train.jsonl, ft_dev.jsonl, classes.json, then the marker "ready"; test parts ft_test_<k>.jsonl each with a marker ft_test_<k>.done.
# Mac side after DEV: score runs/ft_dev_head.npz (classification layer) and runs/ft_dev.jsonl (written answers), then write the
# go signal: `echo both > /workspace/go_test` (or `gen` / `head` to score test one way only), or `touch /workspace/stop` (fail).
# Stages in /workspace/runs/ft.status: SETUP, DOWNLOAD, WAIT_DATA, FEATURES (test photos too if /workspace/features_test exists), SMOKE, TRAIN, HEAD_DEV, MERGE, SERVE, DEV, WAIT_GO, TEST_<k>,
# HEAD_TEST, "exit ...".
TMIN=$1; EPOCHS=$2; DEADLINE=$3; NS=${4:-3}
R=/workspace/runs; ST=$R/ft.status; D=/workspace/data/v11; PY=/workspace/venv/bin/python; BASE=/workspace/models/q38bf16; FT=/workspace/models/q38ft
mkdir -p $R $D; cd /workspace; echo SETUP > $ST
if [ ! -x /workspace/venv/bin/vllm ]; then
python -m venv /workspace/venv && $PY -m pip install -q -U pip uv > $R/setup.log 2>&1 && \
{ /workspace/venv/bin/uv pip install -q --python $PY vllm==0.33.0 "transformers==5.20.2" ninja --torch-backend auto >> $R/setup.log 2>&1 || \
  $PY -m pip install -q vllm==0.33.0 "transformers==5.20.2" ninja >> $R/setup.log 2>&1; } || { echo "exit setup-failed" > $ST; exit 1; }
fi
$PY -m pip install -q peft==0.21.2 accelerate==1.15.0 pillow hf_xet flash-linear-attention >> $R/setup.log 2>&1 || { echo "exit setup-failed" > $ST; exit 1; }
# optional speed-up for the linear-attention layers; training still works (slower) without it
timeout 12m $PY -m pip install -q causal-conv1d >> $R/setup.log 2>&1 || echo "causal-conv1d not installed" >> $R/setup.log
export PATH=/workspace/venv/bin:$PATH PYTHONPATH=/workspace/k12/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
/workspace/venv/bin/vllm download-kernels >> $R/setup.log 2>&1
$PY -c "import vllm, transformers, torch, peft; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__, torch.cuda.get_device_name(0))" > $R/versions.txt 2>&1
echo DOWNLOAD > $ST
$PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B', local_dir='$BASE', max_workers=16)" > $R/download.log 2>&1 || { echo "exit download-failed" > $ST; exit 1; }
echo WAIT_DATA > $ST
i=0; while [ ! -f $D/ready ]; do i=$((i+1)); [ $i -gt 360 ] && { echo "exit data-timeout" > $ST; exit 1; }; sleep 10; done
# Untrained 27B state per photo (no training): input for small classifiers fitted on the Mac while the fine-tune runs.
echo FEATURES > $ST
timeout 60m $PY -m k12.q38_finetune features --model $BASE --requests $D/ft_train.jsonl $D/ft_dev.jsonl \
  --output $R/ft_features_train_dev.npz --log-every 100 > $R/ft_features.log 2>&1 || echo "features failed" >> $R/ft_features.log
if [ -f /workspace/features_test ] && ls $D/ft_test_*.done > /dev/null 2>&1; then
  timeout 120m $PY -m k12.q38_finetune features --model $BASE --requests $(ls $D/ft_test_*.jsonl) \
    --output $R/ft_features_test.npz --log-every 200 >> $R/ft_features.log 2>&1 || echo "test features failed" >> $R/ft_features.log
fi
echo SMOKE > $ST
timeout 25m $PY -m k12.q38_finetune train --model $BASE --train $D/ft_train.jsonl --classes $D/classes.json --output $R/ft_smoke --limit 12 --epochs 1 --accumulation 4 \
  --save-minutes 0 --log-every 1 > $R/ft_smoke.log 2>&1
grep -q K12_Q38_FINETUNE_COMPLETE $R/ft_smoke.log || { echo "exit smoke-failed" > $ST; exit 1; }
rm -rf $R/ft_smoke
echo TRAIN > $ST
timeout $((TMIN + 30))m $PY -m k12.q38_finetune train --model $BASE --train $D/ft_train.jsonl --classes $D/classes.json --output $R/ft_full --epochs $EPOCHS \
  --max-minutes $TMIN > $R/ft_full.log 2>&1 || { echo "exit train-failed" > $ST; exit 1; }
echo HEAD_DEV > $ST
timeout 30m $PY -m k12.q38_finetune head --model $BASE --adapter $R/ft_full/adapter --head $R/ft_full/adapter/head.pt \
  --requests $D/ft_dev.jsonl --output $R/ft_dev_head.npz --log-every 25 > $R/ft_head_dev.log 2>&1 || echo "head dev scoring failed" >> $R/ft_head_dev.log
echo MERGE > $ST
timeout 40m $PY -m k12.q38_finetune merge --model $BASE --adapter $R/ft_full/adapter --output $FT > $R/ft_merge.log 2>&1
grep -q K12_Q38_MERGE_COMPLETE $R/ft_merge.log || { echo "exit merge-failed" > $ST; exit 1; }
echo SERVE > $ST
/workspace/venv/bin/vllm serve $FT --served-model-name q38 --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.95 \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 48 --seed 2026 > $R/serve_ft.log 2>&1 &
PID=$!; i=0; UP=0
while [ $i -lt 180 ]; do
  curl -s -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1 && { UP=1; break; }
  kill -0 $PID 2>/dev/null || break
  i=$((i+1)); sleep 5
done
[ $UP -eq 1 ] || { echo "exit serve-failed" > $ST; exit 1; }
C=/workspace/vlm_zero_shot_client.py; W=$(cat /workspace/workers 2>/dev/null || echo 32)
echo DEV > $ST
$PY $C --requests $D/ft_dev.jsonl --out $R/ft_dev.jsonl --mode nothink --n 3 --max-tokens 1500 --workers $W > $R/run_ft_dev.log 2>&1 < /dev/null
echo WAIT_GO > $ST
i=0; while [ ! -f /workspace/go_test ]; do
  [ -f /workspace/stop ] && { echo "exit stopped-after-dev" > $ST; exit 0; }
  i=$((i+1)); [ $i -gt 180 ] && { echo "exit no-go-signal" > $ST; exit 0; }; sleep 10
done
MODE=$(cat /workspace/go_test | tr -d ' \n'); [ -z "$MODE" ] && MODE=both
K=1
while [ "$MODE" != head ] && { [ -f $D/ft_test_$K.jsonl ] || [ $K -le 1 ]; }; do
  [ "$(date -u +%s)" -ge "$DEADLINE" ] && break
  i=0; while [ ! -f $D/ft_test_$K.done ]; do i=$((i+1)); [ $i -gt 180 ] && { echo "exit upload-timeout" > $ST; exit 1; }; sleep 10; done
  echo TEST_$K > $ST
  $PY $C --requests $D/ft_test_$K.jsonl --out $R/ft_test_$K.jsonl --mode nothink --n $NS --max-tokens 1500 \
    --workers $(cat /workspace/workers 2>/dev/null || echo 32) --deadline $DEADLINE >> $R/run_ft_test.log 2>&1 < /dev/null
  K=$((K+1))
done
if [ "$MODE" != gen ] && [ "$(date -u +%s)" -lt "$DEADLINE" ]; then
  echo HEAD_TEST > $ST
  kill $PID; sleep 15  # the classification layer needs the GPU that vLLM holds
  timeout 150m $PY -m k12.q38_finetune head --model $FT --head $FT/head.pt --requests $(ls $D/ft_test_*.jsonl) \
    --output $R/ft_test_head.npz --log-every 100 --deadline $DEADLINE > $R/ft_head_test.log 2>&1
fi
echo "exit 0" > $ST
