#!/bin/bash
# Runs the proxy sweep, 6 fits in parallel (CPU only).
cd /Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12
{
for l2 in 1e-4 3e-4 1e-3 3e-3; do
  for K in 1 2 3 5; do echo "ce $K 0 $l2"; done
  for lam in 0.3 0.5; do echo "poe 1 $lam $l2"; done
done
} | xargs -P 6 -L 1 sh -c 'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python runs/deep/test-W2/w2_fit.py $0 $1 $2 $3' >> runs/deep/test-W2/sweep.log 2>&1
echo DONE >> runs/deep/test-W2/sweep.log
