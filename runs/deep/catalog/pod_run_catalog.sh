#!/bin/sh
# Paired test of the error catalogue (approved by the user on 12 Nov 2026, one RTX 6000 Ada, about $3.50, limit $5).
# Same recipe and seed twice: 1,199 fold-fit teacher rows + 2,856 Claude rows, then the same plus the 1,503 synthetic
# catalogue rows. Both score the 301 dev rows on machine and teacher transcripts. Progress in /workspace/runs/cat.status.
ST=/workspace/runs/cat.status; mkdir -p /workspace/runs; cd /workspace; echo SETUP > $ST
python -m venv /workspace/venv && /workspace/venv/bin/pip install -q --upgrade pip && \
/workspace/venv/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 && \
/workspace/venv/bin/pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 pandas==2.3.3 numpy scikit-learn joblib pillow huggingface_hub hf_xet && \
/workspace/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-8B', local_dir='/workspace/models/qwen3-8b', max_workers=16)" || { echo "exit setup-failed" > $ST; exit 1; }
cd /workspace/k12
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
PY=/workspace/venv/bin/python; D=/workspace/data; R=/workspace/runs
COMMON="--data $D/competition_data --model /workspace/models/qwen3-8b --pseudo $D/scale_25.jsonl --dataset $D/train_v3_claude.jsonl --explain-weight .3 --max-length 3072 --epochs 1 --batch-size 4 --validation-transcripts gold $D/tx_dev.json"
echo SMOKE > $ST
timeout 20m $PY -m k12.text_diagnose $COMMON --synthetic $D/synthetic.jsonl --limit 8 --output $R/cat_smoke > $R/cat_smoke.log 2>&1
if ! grep -q K12_TEXT_RUN_COMPLETE $R/cat_smoke.log || ! grep -q '"synthetic_rows", "added": 8' $R/cat_smoke.log; then echo "exit smoke-failed" > $ST; exit 1; fi
rm -rf $R/cat_smoke/adapter $R/cat_smoke/head.pt
echo BASE_START > $ST
timeout 170m $PY -m k12.text_diagnose $COMMON --output $R/cat_base > $R/cat_base.log 2>&1 || { echo "exit base-failed" > $ST; exit 1; }
echo SYN_START > $ST
timeout 200m $PY -m k12.text_diagnose $COMMON --synthetic $D/synthetic.jsonl --output $R/cat_syn > $R/cat_syn.log 2>&1
echo "exit $?" > $ST
