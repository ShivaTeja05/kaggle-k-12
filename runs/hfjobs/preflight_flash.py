# /// script
# requires-python = ">=3.11"
# dependencies = ["vllm==0.33.0", "huggingface_hub", "hf_xet"]
# ///
"""CPU-only preflight for the Flash-Next test (costs well under a cent): how fast can this model's files be fetched on
Hugging Face's own machines, by download and through the mounted folder, and does vLLM 0.33.0 know the flags the recipe uses."""
import glob, os, subprocess, sys, time
from huggingface_hub import hf_hub_download, list_repo_files
repo = 'Qwen/Qwen3.8-Flash-Next-FP8'
files = sorted(f for f in list_repo_files(repo) if f.endswith('.safetensors')); print('weight files', len(files), flush=True)
t = time.time(); n = 0
for f in files[:2]:
    p = hf_hub_download(repo, f, local_dir='/tmp/dl'); n += os.path.getsize(p)
dt = time.time() - t; print(f'download: {n / 1e9:.1f} GB in {dt:.0f} s = {n / 1e9 / dt:.2f} GB/s (two files, one after another)', flush=True)
if os.path.isdir('/model'):
    t = time.time(); n = 0
    with open(sorted(glob.glob('/model/*.safetensors'))[2], 'rb') as fh:
        while True:
            b = fh.read(64 << 20)
            if not b: break
            n += len(b)
    dt = time.time() - t; print(f'mounted folder: {n / 1e9:.1f} GB in {dt:.0f} s = {n / 1e9 / dt:.2f} GB/s', flush=True)
h = subprocess.run([os.path.join(os.path.dirname(sys.executable), 'vllm'), 'serve', '--help=all'], capture_output=True, text=True).stdout
for flag in ('--enable-expert-parallel', '--moe-backend', '--tensor-parallel-size', '--engram-config', '--attention-config', '--kv-cache-dtype', '--reasoning-parser', '--speculative-config'):
    print(flag, 'KNOWN' if flag in h else 'not found', flush=True)
import vllm; print('vllm', vllm.__version__, '| disk free GB', round(os.statvfs('/tmp').f_bavail * os.statvfs('/tmp').f_frsize / 1e9), flush=True)
