#!/bin/sh
# Runs on the probe pod: install vLLM, fetch the official 4-bit Qwen3-235B, run the zero-shot probe.
set -e
GPUS=$(nvidia-smi -L | wc -l)
mkdir -p /workspace/runs /workspace/models
python -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -q vllm pandas huggingface_hub
echo PIP_DONE
/workspace/venv/bin/python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-235B-A22B-GPTQ-Int4", local_dir="/workspace/models/q235", max_workers=16))
EOF
echo MODEL_DONE
# vLLM's JIT MoE kernels need the venv's CUDA 13 NVRTC builtins, which are not on the default path.
export PATH=/workspace/venv/bin:$PATH  # torch.compile needs the venv's ninja
export LD_LIBRARY_PATH=/workspace/venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
cd /workspace/k12
set +e  # always write the status file so the orchestrator stops waiting
PYTHONPATH=src timeout 7200 /workspace/venv/bin/python -m k12.scale_probe --model /workspace/models/q235 \
  --data /workspace/data --transcripts /workspace/data/tx_dev.json --output /workspace/runs/probe235 \
  --tensor-parallel "$GPUS" --thinking --max-tokens 3000 --max-model-len 8192 --gpu-memory .95 > /workspace/runs/probe235.log 2>&1
echo "exit $?" > /workspace/runs/probe235.status
