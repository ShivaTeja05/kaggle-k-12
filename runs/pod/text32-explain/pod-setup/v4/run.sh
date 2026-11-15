#!/bin/sh
# Pod-side: install, fetch Qwen3-32B, smoke-test, then the improved-32B run.
# Usage: run.sh <pseudo jsonl name in /workspace/data/competition_data> <timeout seconds> <explanation dataset name in /workspace/data>
PSEUDO=$1; LIMIT=$2; DATASET=$3
cd /workspace
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 scikit-learn pillow joblib scipy huggingface_hub bitsandbytes && \
echo PIP_DONE && /workspace/venv/bin/python /workspace/get32.py && echo MODEL_DONE || { echo "exit setup-failed" > /workspace/runs/text32-v4.status; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--data /workspace/data/competition_data --model /workspace/models/qwen3-32b --load-4bit --pseudo /workspace/data/competition_data/$PSEUDO --dataset /workspace/data/$DATASET --explain-weight .3 --max-length 3072 --epochs 1 --batch-size 4"
timeout 1500 /workspace/venv/bin/python -m k12.text_diagnose $COMMON --output /workspace/runs/smoke-v4 --limit 8 --validation-transcripts /workspace/data/tx_dev.json > /workspace/runs/smoke-v4.log 2>&1
if ! grep -q K12_TEXT_RUN_COMPLETE /workspace/runs/smoke-v4.log; then echo "exit smoke-failed" > /workspace/runs/text32-v4.status; exit 1; fi
echo SMOKE_OK
timeout $LIMIT /workspace/venv/bin/python -m k12.text_diagnose $COMMON --output /workspace/runs/text32-v4 --validation-transcripts gold /workspace/data/tx_dev.json --test-transcripts /workspace/data/tx_test.json > /workspace/runs/text32-v4.log 2>&1
echo "exit $?" > /workspace/runs/text32-v4.status
