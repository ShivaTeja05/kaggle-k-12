#!/bin/sh
# Improved 32B without taxonomy descriptions: descriptions tripled prompt length (5.85 s/row, ~13 h > timeout).
cd /workspace/k12-explain
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 25200 /workspace/venv/bin/python -m k12.text_diagnose --data /workspace/data/competition_data --model /workspace/models/qwen3-32b --load-4bit --output /workspace/runs/text32-explain --pseudo /workspace/data/competition_data/claude_unl_v2.jsonl --dataset /workspace/data/train_v3.jsonl --explain-weight .3 --max-length 3072 --validation-transcripts gold /workspace/runs/tx_dev.json --test-transcripts /workspace/runs/tx_test.json --epochs 1 --batch-size 4 > /workspace/runs/text32-explain.log 2>&1
echo "exit $?" >> /workspace/runs/text32-explain.status
