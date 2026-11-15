from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-32B", local_dir="/workspace/models/qwen3-32b"))
