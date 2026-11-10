# CoLearn K-12 misconception detection

An offline competition project with a measured CPU baseline, private Kaggle vision experiments, error analysis and a rules/research record. **Top three is the longer-term objective; approximately 0.82 public MAP@3 is the immediate target. Neither is achieved yet.** Kaggle free compute only, with a hard cap of **10 of the account's 30 weekly GPU hours**, including quota already consumed.

The best verified submission scored **0.59800 public MAP@3**, up from **0.58216**, using a temperature-softened Qwen3-VL-4B classifier mixed with a 25% question/topic prior. The original classifier scored **0.59524** on the 301-row development fold. Public scores, local development scores and unmeasured experiments are recorded separately in `reports/status.json`.

Start with [the clone and setup guide](docs/workspace_setup.md) to reproduce the CPU baseline or restore existing model outputs from Kaggle. For experiment context, see [the research and strategy](docs/competition_research.md) and [the data audit](reports/data_audit.json). Competition data, weights, logs and predictions stay in ignored local directories.

Current work is CPU-only. The Kaggle GPU commands below document the historical training setup and are gated by [the current budget policy](reports/gpu_budget.json).

The most recent [paired model audit](reports/model_error_audit.json) measured **0.65006 development MAP@3** for a CPU reranker using cached 4B and 27B rankings on 301 rows. The 27B cache covers development only. The [handwriting research follow-up](docs/htr_survey_takeaways.md) records the OCR comparison and teacher-location diagnostic.

The [0.82 experiment log](reports/target_082.md) records the current step-by-step fixes and distinguishes completed work from unmeasured candidates.

## Reproduce the working CPU baseline

The existing Kaggle account downloaded the canonical `competition_data/train.csv`, `competition_data/test.csv` and `competition_data/taxonomy_v1.csv` into `data/raw`. The legacy root train/test CSVs have different topic names; we compared them and aligned with notebook data version 2. See `reports/data_versions.json`. Never commit data files or credentials.

```bash
PYTHONPATH=src python3 -m k12.baseline
PYTHONPATH=src python3 -m k12.analyze \
  --predictions artifacts/baseline/oof.npz \
  --folds artifacts/baseline/folds.csv
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Executed baseline: **0.48533 MAP@3 OOF**, top-3 recall **0.67267**. Artifacts include `model.joblib`, ID/class-aligned OOF/test bundles, `submission.csv`, fold assignments, metrics and per-class/grade/support/confusion reports. This baseline uses question/topic priors, not handwriting; it is not a prize-ready model.

## Vision model

`src/k12/vision.py` trains language LoRA on official Qwen3-VL-4B, with a root classifier and auxiliary correctness head. Input fields match test time. Both working photos and available question diagrams are used. Teacher transcriptions, correctness and misconception annotations are never classification inputs. Correctness is auxiliary supervision only.

`scripts/build_kaggle.py` bundles our code into a private notebook, pins official model version 1, disables internet, and selects free T4 compute. It does not upload local competition files, public reference notebooks, credentials or predictions.

```bash
python3 scripts/build_kaggle.py --username YOUR_KAGGLE_USERNAME \
  --output kaggle/smoke --smoke
python3 scripts/launch_kaggle.py kaggle/smoke --timeout 1200
# Only after reviewing the successful smoke result:
python3 scripts/build_kaggle.py --username YOUR_KAGGLE_USERNAME \
  --output kaggle/train
python3 scripts/launch_kaggle.py kaggle/train --timeout 21600
```

Training runs one development fold by default. A fold result is **not full OOF**. Do not report the four-row smoke score as model performance. Before final use, choose the epoch on development data, check a reserved audit fold, then train the fixed recipe on all labelled rows with `--full-train`. Rebuild the notebook after changing source.

The launcher reads current Kaggle quota and reserves active jobs' remaining maximum runtime plus overhead and a safety margin. It refuses jobs that could exceed the 10-hour cap. Use it for every GPU launch, and keep `reports/gpu_budget.json` current. The example six-hour timeout is not automatically approved by the budget check.

For an offline inference-only run, pass `--checkpoint /path/to/epoch-N --predict-test` with the same model/config and class order. The checkpoint includes adapter weights, both heads, processor files, trained IDs, seed, dependency versions and data/config hashes. The base model remains a separately attached, versioned input.

`src/k12/taxonomy_ranker.py` is an experimental alternative that reads images and all taxonomy descriptions, with same-question examples from the fitting fold only. Run its module on a CUDA machine with the GPU dependencies and attached official weights. Its generated top-three scores are rankings, **not calibrated probabilities**. It has not earned promotion until measured.

On Kaggle, the taxonomy ranker balances weights across both T4s to leave memory for its long prompt. The first single-device attempt ran out of memory before producing predictions. The classifier's end-to-end smoke run passed after bypassing an incompatible, unused optional torchao dispatcher for unquantized FP16 LoRA; both findings are recorded in the research report.

`src/k12/blend.py` aligns IDs/classes and selects a small blend grid on development folds, reporting the reserved audit fold separately. Supply complete, genuine OOF bundles; never relabel in-sample predictions as OOF. Rank-derived scores need separate calibration or rank-fusion evaluation before probability blending.

## Adaptation experiment

[The adaptation strategy](docs/adaptation_strategy.md) reviews internal LoRA, AdaLoRA, DoRA, QLoRA, reading supervision and the data/compute limits. The implemented screening experiment adds a zero-initialized bottleneck residual between the frozen trained Qwen representation and its original classifier. It compares this with a linear-head refit on the same fitting examples, with tempered class weights and auxiliary major-error supervision. It does not change Qwen's transformer blocks or use gold transcriptions as prediction inputs.

Cache representations once on a budget-admitted private GPU run:

```bash
python3 scripts/build_adaptive_job.py --username YOUR_KAGGLE_USERNAME
python3 scripts/launch_kaggle.py kaggle/adaptive-features --timeout 1800
```

After downloading the completed cache, train both small heads on a local CPU with PyTorch:

```bash
PYTHONPATH=src python3 -m k12.adaptive train \
  --features runs/adaptive-features/adaptive_features \
  --output artifacts/adaptive
```

The feature manifest enforces checkpoint/data/class identity and fitting/validation separation. A measured development gain is required before spending quota on test features. The development fold has already been used for previous selection; it is not an untouched audit. No adapter improvement is claimed merely from passing functional tests.

Measured outcome: the rank-32 output adapter scored 0.57032 against the same cached baseline at 0.59801. A smaller, regularized rank-8 follow-up scored 0.59911, with a difference interval spanning zero. Neither is promoted. See `reports/adaptation_comparison.json` and the strategy document for the overfitting diagnosis and next supervision experiment.

`src/k12/reading_finetune.py` resumes the existing internal LoRA with response-only teacher reading/root-cause supervision. The classification head reads the prefix before the teacher response. A real-model isolation check, two-GPU training, checkpoint saving and sharded prediction passed functional smoke tests. The full development experiment launched on 8 November at 16:19 UTC but was cancelled while queued after an unrelated account GPU job was discovered. It has no full-development quality result. The user then chose to keep that other job and continue K-12 on CPU; further K-12 GPU launches are disabled in the budget policy.

`src/k12/holdout_calibration.py` reproduces the submitted CPU blend. Its full-development tuned score is 0.60742; question-group cross-fitting within the reused development fold scored 0.59801. This is not untouched image-model OOF. `src/k12/prior_correction.py` tested a bounded conditional-prior odds adjustment: cross-fit MAP@3 0.60742, exactly the existing fixed blend's score, with paired difference interval spanning zero. It was rejected without another public submission. Frozen-feature nearest-neighbour memory also failed the improvement screen.

## Submission and continued work

No leaderboard submission is made automatically. The prior and smoke outputs are not substitutes for a validated image model. Use `scripts/validate_submission.py` on a complete candidate before submission. Five submissions/day and two final selections are allowed under the rules checked on 8 November.

This project does not set up a recurring automation, reserve paid hardware, or promise a final leaderboard rank. Current execution status is recorded in `reports/status.json`.

Read the saved results of a run without starting another GPU job:

```bash
python3 scripts/sync_kaggle.py YOUR_KAGGLE_USERNAME/k12-vision-development \
  --output runs/vision-development
```

For a genuine completed single-fold validation bundle, use `k12.analyze --allow-holdout` with that run's `folds.csv`. For the taxonomy ranker, bundle a notebook with `scripts/build_kaggle.py --recipe taxonomy`; it evaluates development images before generating any full test file.
