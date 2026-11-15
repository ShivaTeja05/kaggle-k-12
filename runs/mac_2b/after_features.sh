#!/bin/zsh
# after the untrained features: state check + 2 generated answers, then features of the fine-tuned model (dev first, then fit rows, until 09:55 UTC)
cd /Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55 PYTHONPATH=src
PY=$HOME/.cache/k12-api-venv/bin/python; M=$HOME/.cache/k12-models/Qwen3.5-2B
while pgrep -f "after_train.sh" > /dev/null; do python3 -c "import time; time.sleep(15)"; done
date -u
caffeinate -i $PY -u runs/mac_2b/check_state_and_generate.py $M runs/mac_2b/full/adapter mps 2 runs/mac_2b/reports/state_and_generation.json > runs/mac_2b/full/check_log.txt 2>&1
date -u
caffeinate -i $PY -u runs/mac_2b/run_mps.py features --model $M --adapter runs/mac_2b/full/adapter --requests data/v11/ft_dev.jsonl /private/tmp/claude-501/-Users-gaddamshivateja-Documents-ChatGPT-kaggle-K-12/3c20ecda-6940-4ca3-af6c-c57b844f0174/scratchpad/mac2b/ft_train_400.jsonl --output runs/mac_2b/features/finetuned_dev_400.npz --layers=-1,-4,-7,-10,-13 --log-every 50 --deadline 1794736500 --device mps > runs/mac_2b/features/log_finetuned.txt 2>&1
date -u; echo AFTER_FEATURES_DONE
