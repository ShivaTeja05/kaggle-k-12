#!/bin/sh
# Runs on the H200 pod: few-shot Qwen3-235B on the 3,498 test rows (same settings as the 0.650 dev run).
# Always writes /workspace/runs/fewshot235_test.status so the orchestrator never waits on a dead run.
mkdir -p /workspace/runs /workspace/models
STATUS=/workspace/runs/fewshot235_test.status

# Model download (about 520 MB/s with 16 workers) runs in parallel with the slow vLLM install.
python -m venv /workspace/dlvenv && /workspace/dlvenv/bin/pip install -q huggingface_hub hf_xet
/workspace/dlvenv/bin/python - > /workspace/dl.log 2>&1 <<'EOF' &
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-235B-A22B-GPTQ-Int4", local_dir="/workspace/models/q235", max_workers=16))
EOF
DL=$!

python -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip uv
/workspace/venv/bin/uv pip install -q --python /workspace/venv/bin/python vllm==0.33.0 pandas \
  || /workspace/venv/bin/pip install -q vllm==0.33.0 pandas \
  || { echo "setup-failed pip" > $STATUS; exit 1; }
echo PIP_DONE
wait $DL || { echo "setup-failed download" > $STATUS; exit 1; }
echo MODEL_DONE

# vLLM 0.33 JIT kernels: CUDA 13 NVRTC and nvcc from the venv, ninja on PATH, no FlashInfer sampler build.
C=/workspace/venv/lib/python3.11/site-packages/nvidia/cu13
export CUDA_HOME=$C VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$C/bin:/workspace/venv/bin:$PATH
export LD_LIBRARY_PATH=$C/lib:${LD_LIBRARY_PATH:-}
cd /workspace/k12
PYTHONPATH=src timeout 10800 /workspace/venv/bin/python -m k12.scale_probe --model /workspace/models/q235 \
  --data /workspace/data --transcripts /workspace/data/tx_test.json --output /workspace/runs/fewshot235_test \
  --split test --shots 4 --tensor-parallel 1 --thinking --max-tokens 6000 --max-model-len 12800 --gpu-memory .95 \
  --chunk 350 > /workspace/runs/fewshot235_test.log 2>&1
echo "exit $?" > $STATUS
