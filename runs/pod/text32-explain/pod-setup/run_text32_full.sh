#!/bin/sh
cd /workspace/k12-text
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 36000 /workspace/venv/bin/python -m k12.text_diagnose --data /workspace/data/competition_data --model /workspace/models/qwen3-32b --load-4bit --output /workspace/runs/text32-full --full-train --pseudo /workspace/data/competition_data/claude_unl.jsonl --test-transcripts /workspace/runs/tx_test.json --batch-size 4 --epochs 2 > /workspace/runs/text32-full.log 2>&1
echo "exit $?" >> /workspace/runs/text32-full.status
