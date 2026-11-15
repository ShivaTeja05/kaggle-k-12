"""Mac-only launcher for k12.q38_finetune: the same commands and code, but the MPS allocator's cache is emptied before every
row. Without this the cache fills with freed blocks of the wrong size (row lengths differ) and the run dies with
"MPS backend out of memory" or pushes the Mac into swap. The module itself is not changed.

Usage: PYTHONPATH=src python runs/mac_2b/run_mps.py train --device mps ...   (same arguments as python -m k12.q38_finetune)
"""
import torch
import k12.q38_finetune as Q

_encode = Q.encode


def encode(*args, **kwargs):
    torch.mps.empty_cache()
    return _encode(*args, **kwargs)


Q.encode = encode
Q.main()
