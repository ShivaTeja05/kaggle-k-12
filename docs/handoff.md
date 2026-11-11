# K-12 handoff — read this first

Last updated 11 November 2026, ~04:00 UTC. Written so another engineer or LLM can continue without the chat history. Keep it current when the state changes.

## Goal and hard rules

- Competition: [Misconception Detection in K-12 Student Math Workings](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings). Predict up to 3 ranked root-misconception codes (75 taxonomy codes or `NONE`) per test photo; metric MAP@3. Upload `submission.csv` (`trace_id,root_code`). Not a code competition.
- Deadline: **17 November 2026, 00:00 UTC** (treat 16 Nov as the last working day). 5 submissions/day; select 2 final entries.
- User goal: top 3 (leaders ≈ 0.846 public). Immediate milestone 0.70+.
- Read `AGENTS.md`. Key rules: no Kaggle GPU jobs for K-12 (account quota is exhausted by another project); RunPod is allowed with a **$50 total cap** tracked in `reports/runpod_budget.json`; never copy Kaggle credentials to a pod; never hand-label test rows; keep competition data private.
- Winners must share full training + inference code under an open licence, so keep everything reproducible.

## Scores so far

| What | Score |
|---|---|
| Public best: 8B transcription-trained classifier, blended 0.7/0.3 with 4B + 10% prior (submission 58221584) | **0.62383** public, rank 182/403 |
| Previous public: 4B + prior | 0.59800 |

Development fold = fold 0 of `make_folds` (301 rows); the other 1,199 rows are the fitting set. Dev MAP@3 (raw / with 25% prior at T=2):

| Run | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| 4B original (reference, submitted recipe) | | | 0.607 calibrated |
| A: 8B Qwen3-VL classifier | 0.550 | 0.604 | 0.619 |
| B: A + teacher-transcription LM loss (weight 0.15) | 0.557 | 0.615 | **0.632** |
| D: A + soft secondary labels + rare oversampling | 0.529 | 0.611 | 0.618 (no gain) |
| E: A + error-family and has-error aux heads | 0.524 | 0.617 | 0.613 (no gain) |
| v3: B with evidence-first targets from `train_v3.jsonl` | 0.539 | **0.627** | running |
| 32B QLoRA classifier | 0.403 (stopped) | | |
| Best dev ensemble: B^0.7 · 4B^0.3, then 10% prior | | | **0.647** |

Read-then-diagnose (new direction):

- Transcriber (Qwen3-VL-8B LoRA trained to write the teacher transcription): on dev photos, number recall 0.88, number precision 0.91, median character error 0.33 (mostly formatting). Older OCR got 0.12–0.22 recall.
- Text reasoner (Qwen3-8B LoRA + classifier head over question + transcript): epoch 1 dev MAP@3 0.570 with teacher transcripts, 0.542 with machine transcripts. Reading costs only ~0.03; diagnosis is now the bottleneck.

Main failure mode (8B classifier): 68% of lost points are "saw an error but named the wrong error family". Rare codes (≤5 examples) score 0.26.

## What is running right now

Pods are listed in `reports/runpod_budget.json` (spend is computed from it by `scripts/runpod_pod.py status`). SSH key: `~/.runpod/ssh/runpodctl-ssh-key`. Pod IPs/ports change if a pod restarts; get them from `python scripts/runpod_pod.py status`.

| Pod | GPU | $/hr | SSH | Job | Output |
|---|---|---|---|---|---|
| `jktv441e4crx35` | RTX A6000 | 0.59 | `-p 22123 root@194.68.245.215` | v3 dev run; text reasoner with 3 same-question examples + taxonomy descriptions | `/workspace/runs/dev-v3/`, `/workspace/runs/text-examples/` |
| `xkakvh2v50n06c` | L40S | 1.09 | `-p 15342 root@202.181.159.211` | Transcribing 3,498 test photos (resumable) | `/workspace/runs/tx_test.json`, status in `tx_status.txt` |
| `ocdo52ziviwftb` | RTX A6000 | 0.59 | `-p 33295 root@38.147.83.23` | Text reasoner scored on teacher vs machine dev transcripts | `/workspace/runs/text-stage2/` |

A local watchdog (`python -I scripts/runpod_pod.py watchdog`) terminates all K-12 pods when total spend reaches cap − $1. It only runs while this Mac is awake. Terminate everything with `python -I scripts/runpod_pod.py terminate` (one pod: `--pod ID`). The RunPod API key lives in `~/.runpod/config.toml`; never print it.

Pod layout: code in `/workspace/k12*` (copy `src/` with `COPYFILE_DISABLE=1 tar czf - src | ssh ... 'tar xzf - -C /workspace/k12-next'`), venv `/workspace/k12/venv` (pod 4: `/workspace/venv`), data in `/workspace/data/competition_data/` (train/test/taxonomy CSVs, `images/{train,test,questions}`; pod 1 also has `images/unlabelled` and `unlabelled.csv`), models in `/workspace/models/{qwen3-vl-8b,qwen3-8b}`. Library versions: torch 2.11 cu130, transformers 5.20.2, peft 0.21.2.

## Local files (private, gitignored under `data/` and `runs/`)

- `data/raw/`: canonical CSVs. `data/bundle/k12_images.tar`: train/test/question photos. `data/bundle/unl/k12_unlabelled.tar`: 50,000 unlabelled photos. Kaggle per-file downloads are rate-limited; bulk files came from private CPU notebooks `kaggle/image-bundle` and `kaggle/unlabelled-bundle`.
- `data/v3/train_v3.jsonl`: dataset v3 (evidence-first targets, families, soft codes, weights). `data/v3/pseudo_v3.jsonl`: 1,565 confidence-filtered soft pseudo-labels. `data/v3/dev_ids.json`: fold-0 IDs.
- `runs/pod/`: downloaded dev/test predictions, checkpoints for A (`dev-cls/epoch-3`) and B (`dev-read/epoch-3`), pseudo scores, dev transcripts (`tx/tx_dev.json`).
- `artifacts/submission-8b-read-blend/`: the 0.62383 submission.

## Code map

- `src/k12/pod_train.py`: Qwen3-VL LoRA classifier. Options: `--reading-weight`, `--dataset train_v3.jsonl`, `--pseudo pseudo_v3.jsonl`, `--aux-weight`, `--secondary-weight`, `--rare-copies`, `--load-4bit`, `--lora-vision`, `--full-train`, `--predict-test`. CPU-thread prefetching keeps the GPU busy.
- `src/k12/pod_predict.py`: score any CSV with a checkpoint (test or `--split unlabelled`).
- `src/k12/transcriber.py`: `train` / `transcribe` (resumable JSON output).
- `src/k12/text_diagnose.py`: text reasoner; `--validation-transcripts gold path.json`, `--test-transcripts`, `--examples K`, `--descriptions`, `--full-train`.
- `scripts/evaluate_pod_run.py` (dev comparison and 4B blends), `scripts/blend_submission.py`, `scripts/validate_submission.py`, `scripts/evaluate_transcripts.py`, `scripts/build_dataset_v3.py`, `scripts/build_pseudo_labels.py`, `scripts/runpod_pod.py`.

## Next steps, in order

1. Collect epoch-3 dev results: v3 image model, reasoner on machine transcripts, reasoner with examples. Copy their `validation*_epoch_3.npz` locally (`scp -P PORT root@HOST:/workspace/runs/...`).
2. Ensemble on dev: v3/B image probabilities × text reasoner (machine transcripts) × 4B, then a light prior. Pick weights on dev, but keep the grid small.
3. When `tx_test.json` is complete, copy it to the reasoner pod and score test with `--test-transcripts`.
4. Final models: retrain the winning image recipe with `--full-train --predict-test` (optionally `--pseudo`), and the reasoner with `--full-train --test-transcripts`. Then blend with `scripts/blend_submission.py` (adapt for 3 inputs), validate, and submit. Machine transcripts for test come from a transcriber trained on 1,199 rows; retraining it on all 1,500 is a cheap upgrade.
5. Promising next levers: a larger text reasoner (Qwen3-14B/32B, QLoRA) since diagnosis is the bottleneck; pseudo-labels from the ensemble; same-question examples.
6. Terminate idle pods. Record every submission in `reports/status.json`.
