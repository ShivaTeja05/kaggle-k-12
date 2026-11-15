#!/bin/zsh
# after the full training run: head on dev, then untrained features on 400 fit rows + 301 dev rows
cd /Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55 PYTHONPATH=src
PY=$HOME/.cache/k12-api-venv/bin/python; M=$HOME/.cache/k12-models/Qwen3.5-2B
while pgrep -f "python -u runs/mac_2b/run_mps.py train" > /dev/null; do python3 -c "import time; time.sleep(20)"; done
date -u
if [ -f runs/mac_2b/full/adapter/head.pt ]; then
  caffeinate -i $PY -u runs/mac_2b/run_mps.py head --model $M --adapter runs/mac_2b/full/adapter --head runs/mac_2b/full/adapter/head.pt --requests data/v11/ft_dev.jsonl --output runs/mac_2b/full/head_dev.npz --log-every 25 --device mps > runs/mac_2b/full/head_log.txt 2>&1
fi
date -u
mkdir -p runs/mac_2b/features
caffeinate -i $PY -u runs/mac_2b/run_mps.py features --model $M --requests /private/tmp/claude-501/-Users-gaddamshivateja-Documents-ChatGPT-kaggle-K-12/3c20ecda-6940-4ca3-af6c-c57b844f0174/scratchpad/mac2b/ft_train_400.jsonl data/v11/ft_dev.jsonl --output runs/mac_2b/features/untrained_400_dev.npz --layers=-1,-4,-7,-10,-13 --log-every 50 --device mps > runs/mac_2b/features/log.txt 2>&1
date -u; echo AFTER_TRAIN_DONE
