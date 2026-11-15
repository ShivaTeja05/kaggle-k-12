"""Geometric mean of prediction bundles that cover the same rows (used to average ranker v1 and v2).

Usage: python scripts/average_bundles.py --inputs A.npz B.npz --out OUT.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from k12.core import aligned_bundle, fingerprint, normalize, save_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    with np.load(ROOT / a.inputs[0], allow_pickle=False) as b:
        ids, classes = b['ids'].tolist(), b['classes'].tolist()
    P = [aligned_bundle(ROOT / p, ids, classes) for p in a.inputs]
    avg = normalize(np.exp(np.mean([np.log(np.clip(p, 1e-12, None)) for p in P], axis=0)))
    out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(out, ids, classes, avg, kind='geometric_mean', inputs=a.inputs, input_sha256=[fingerprint(ROOT / p) for p in a.inputs])
    top = [p.argmax(1) for p in P]
    print({'rows': len(ids), 'top1_disagree_first_two': int((top[0] != top[1]).sum()), 'out': str(out)})


if __name__ == '__main__':
    main()
