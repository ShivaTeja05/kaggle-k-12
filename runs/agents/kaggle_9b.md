# Kaggle worker log: Qwen3.5-9B fine-tune rehearsal on 2x T4 (15 Nov 2026)

Owner approval (AGENTS.md, 15 Nov ~04:15 UTC): "okay start the free runs on mac and kaggle". Kaggle GPU for this job only,
at most 10 GPU hours of the 30-hour window (refresh 21 Nov), at least 20 h stay unused. No submission, no internet GPU
kernels, nothing public. All times UTC.

## Actions

- 04:36 Read AGENTS.md, HANDOFF.md section 7, `scripts/launch_kaggle.py`, `src/k12/q38_finetune.py`, precedent kernels.
  `kaggle quota`: GPU 0.00 h used, 30.00 h remaining, refresh 2026-11-21.
- 04:38 `reports/gpu_budget.json` updated for the approval: `gpu_launches_allowed: true`, `hard_cap_total_used_hours: 10.0`,
  `minimum_remaining_hours: 20.0` (so the launcher itself enforces "20 h stay unused"), reason text with the owner's words.
  Other fields kept.
- 04:40 Active sessions on the account: one, `shivateja005/qcc-job-m6p-648fit` (another project). Pulled its metadata:
  `enable_gpu: false`, so it uses no GPU quota. Not touched.
- 04:41 `src/k12/q38_finetune.py` changed (small, default behaviour unchanged; tested on the tiny model on this Mac):
  - `--skip-failed-rows N` (default 0 = stop at the first, as before): carry on past N rows that run out of GPU memory or
    give a non-finite loss; the row and the gradients since the last update are dropped and a `row_skipped` line is printed.
  - `--gpu-memory` accepts one value per GPU ("11GiB,13GiB").
  - `train` log lines also carry `max_gpu_gb_each` (every GPU) and `rows_skipped`; `train_report.json` has `rows_skipped`.
- 04:43 Private dataset created: `shivateja005/k12-q35-ft-inputs` (ft_train.jsonl, ft_dev.jsonl, ft_test_1.jsonl,
  classes.json, ft_build_report.json; 35 MB; no photos). Confirmed `isPrivate: true`, status ready.
- 04:50 Kernel code written: `kaggle/q35-9b-ft/runner.py` (stages, time limits, status.json) and `kaggle/q35-9b-ft/build.py`
  (writes `run.py` = config + the fine-tune script embedded byte for byte + runner; modes inspect / smoke / main / test).
  Dry-run on this Mac with the tiny random model: smoke and main modes pass end to end.
- 04:50 CPU-only inspection kernel pushed (no GPU quota): `shivateja005/k12-q35-9b-inspect` version 1 (private, offline).
- 04:52 Inspection kernel finished (54 s, CPU). Output in `runs/kaggle-q35-9b/inspect/`. Findings: Kaggle image has
  torch 2.11.0, transformers 5.20.2 (has `Qwen3_5ForConditionalGeneration`), peft 0.21.2, accelerate 1.15.0;
  `flash-linear-attention`, `causal-conv1d`, `bitsandbytes` NOT installed (linear-attention layers use the reference
  PyTorch code). Model at `/kaggle/input/models/shelterw/qwen3.5/transformers/qwen3.5-9b/1` (19.33 GB, hidden 4096,
  32 layers, vocab 248,320, output layer not tied to the embeddings). Competition data at
  `/kaggle/input/competitions/misconception-detection-in-k-12-student-math-workings/competition_data`. All 5,999 photo
  references in the three request files exist. Measured tokens per training row: 2,580-2,919 with one photo, up to 3,850
  with two photos (question diagram + working); dev prompt 2,583; the prompt ends with the empty think block.
- 04:54 Second change to `src/k12/q38_finetune.py`: `--answer-logits-only` (default off). Vocabulary scores are computed
  only at the answer positions (training) or not at all (`head`). Reason: at 3,850 tokens the full score matrix plus its
  gradient is about 4 GB on the second T4, which also holds 7-8 GB of weights. Checked on the tiny model: identical
  losses, identical saved adapter (max abs diff 0.0), identical `head` probabilities.
- 04:56 `kaggle/q35-9b-ft/admit.py` written: registers other projects' active sessions in `reports/gpu_budget.json` with
  zero GPU time only after reading `enable_gpu: false` from their metadata; exits 2 if any unknown active session uses GPU.
  Registered `shivateja005/qcc-job-m6p-648fit` (CPU-only).
- 04:57 First launcher calls: one silent no-op (the launcher file read as empty for a moment: iCloud), then HTTP 429 from
  Kaggle (too many status calls). No job was created by either. Waited 100 s.
- 05:00 **GPU launch 1 (smoke)** through `scripts/launch_kaggle.py kaggle/q35-9b-ft --timeout 2700`:
  kernel `shivateja005/k12-q35-9b-ft` version 1. Launcher budget report: used 0.0 h, candidate 2,700 s, projected worst
  case 1.2 h of cap 10.0 h, projected remaining 28.8 h (minimum 20.0), admitted. Internal job deadline 2,280 s.
  Plan: 12 training rows (the 4 longest + 8 first), float16, split "9.5GiB,13GiB", then `head` on 10 dev rows.
- 05:11 **Smoke job ended by itself** (COMPLETE; script ran 603 s; `kaggle quota` afterwards: 0.17 h used, so two T4s are
  billed as wall-clock time, not double). Output in `runs/kaggle-q35-9b/smoke1/` (status.json, logs/, smoke adapter, bundle).
  Measured on 2x Tesla T4 (15,360 MiB each), float16 + GradScaler, LoRA r16 on 248 linear layers (43.3 M trainable):
  - Model load: about 75 s the first time, 24 s when the files are cached.
  - Split "9.5GiB,13GiB": rows of 3,650-3,680 tokens ran out of memory on GPU 1 (it also holds the output layer); the
    script skipped two of them as designed and stopped at the third. Split "11GiB,13GiB": all 12 rows trained, including
    the 4 longest (up to 3,850 tokens); peak allocated 14.4 / 14.6 GB of 15.6 GB (nvidia-smi 14,539 / 14,797 MiB of 15,360).
    Little headroom: `--skip-failed-rows 40` stays on for the main run.
  - Training speed: 14.7-15.5 s per row on ordinary one-photo rows (about 2,600-2,900 tokens), 19-21 s on the two-photo
    rows; 18.4 s per row over the 12 smoke rows (which over-represent long rows). About 160-170 tokens per second.
  - Losses finite. Text loss 0.80 -> 0.65 over 12 rows (the base model already writes the teacher format fairly well);
    classification loss 4.2-4.9 (chance is ln 76 = 4.33), as expected after 12 rows.
  - `head` scoring with the smoke adapter: 3.5 s per row (median of 9; mean 3.4), probabilities finite and sum to 1,
    bundle format correct (10 x 76 plus final_ok).
  Consequence: one epoch (1,306 rows) costs about 5.8 h on 2x T4; test scoring (3,498 rows) about 3.4 h. Both do not fit
  in 10 h with a full epoch, so the main job trains for as long as one 6-hour session allows (about 0.9 epoch) and the
  test job, if it is run at all, scores the least-confident test rows first.
- 05:19 **GPU launch 2 (main)** through `scripts/launch_kaggle.py kaggle/q35-9b-ft --timeout 21600`:
  kernel `shivateja005/k12-q35-9b-ft` version 2. Launcher budget report: used 0.17 h, candidate 21,600 s, projected worst
  case 6.62 h of cap 10.0 h, projected remaining 23.38 h (minimum 20.0), admitted. Config: split "11GiB,13GiB",
  `--answer-logits-only`, epochs 2 with the clock governing (`--max-minutes` about 319), accumulation 8, lr 1e-4,
  head-lr 1e-3, rank 16, save every 30 min, skip-failed-rows 40; then `head` on all 301 dev rows and MAP@3 inside the job.
  Internal deadline 21,060 s. Expected end about 11:05-11:10 UTC.
- 05:20 Prepared the optional test job (NOT launched): `kaggle/q35-9b-test/order.json` holds the 3,498 test ids ordered
  by the current best system's top probability (least confident first, from `artifacts/submission-3way-pw14-q38/test.npz`);
  `build.py test` embeds the order so a time-limited `head` run covers the rows that matter most first. Dry-run locally.

## State while the main job runs (for whoever picks this up)

- Running: `shivateja005/k12-q35-9b-ft` version 2, launched 05:18:55 UTC, Kaggle timeout 21,600 s, ends by itself at
  about 11:05-11:10 UTC. Check: `kaggle kernels status shivateja005/k12-q35-9b-ft`.
- When it ends: `kaggle kernels output shivateja005/k12-q35-9b-ft -p runs/kaggle-q35-9b/main` then
  `python kaggle/q35-9b-ft/score_dev.py runs/kaggle-q35-9b/main/dev_head.npz --report reports/q35_9b_dev.json` and
  `PYTHONPATH=src:scripts python scripts/fuse_v3.py --all-rows --dev-v3 runs/kaggle-q35-9b/main/dev_head.npz --report reports/q35_9b_fuse.json`.
- Quota used so far: 0.17 h (smoke). Main job worst case 6.0 h. Whatever is left under 10 h (minus the launcher's
  0.35-0.45 h margin) is the most a test job may take: `python kaggle/q35-9b-ft/build.py test job_seconds=<timeout-600>`,
  `python kaggle/q35-9b-ft/admit.py <scratch dir>`, `python scripts/launch_kaggle.py kaggle/q35-9b-test --timeout <s>`.
  Only if dev alone >= 0.70 or the fused gain is above 0.

## Main job result (11:03 UTC)

- 11:03 Main job ended by itself (COMPLETE) after 20,551 s (5.71 h). Output in `runs/kaggle-q35-9b/main/`
  (status.json, logs/train.log, logs/dev_head.log, ft/adapter/ with adapter_model.safetensors 173 MB and head.pt,
  ft/train_report.json, dev_head.npz, kernel log).
- Training: 1,192 rows seen (0.91 of one epoch), 149 updates, schedule completed on the clock (320.3 min), 16.23 s per
  row, 165 tokens per second, no row skipped, no out-of-memory, peak 14.5 / 14.8 GB allocated (nvidia-smi 14,719 /
  14,753 MiB of 15,360). Text loss 0.58 (first third) -> 0.49 -> 0.46 (last third). Classification loss 4.09 -> 2.94 ->
  2.45. Top-1 of the classification layer on rows before it trained on them: 0.31 -> 0.43 -> 0.46.
- Dev scoring: 301 rows in 1,089 s = 3.6 s per row, peak 11.8 / 11.5 GB.
- Dev, classification layer alone (`reports/q35_9b_dev.json`): MAP@3 **0.562** [0.511, 0.612], top-1 0.485.
  By slice: none 0.888 (101 rows), seen_pair 0.571 (94), new_pair 0.260 (84), unseen_q 0.182 (22).
  `final_ok` output against the teachers' final_answer_correct: accuracy **0.914** (33% of dev rows are true).
- Fused (`reports/q35_9b_fuse.json`, `fuse_v3.py --all-rows`): current fused 0.796, new cross-fitted **0.783**,
  difference **-0.013 [-0.023, -0.005]**. Pass mark (+0.02) failed. In-sample best setting 0.798.
- 11:04 `kaggle quota`: **11.22 h used, 18.78 h remaining** on the account. K-12 used about 5.88 h (0.17 + 5.71). The
  rest is another project: `gemma-4-k3-t02-timing`, `gemma-4-g4-e4b-adapter`, `gemma-4-g5-e4b-om2` (NvidiaL4, GPU on,
  first run 08:17 UTC; the last one still RUNNING at 11:04). Read only (status, metadata); not touched.
- Decision: **no test job.** Neither condition holds (alone 0.562 < 0.70; fused gain negative), and the account is past
  both limits the launcher checks (10 h used, 20 h unused), so it would refuse anyway.
- 11:05 `reports/gpu_budget.json`: `gpu_launches_allowed: false` again with the reason, quota snapshot 11.22 / 18.78,
  and a `k12_usage_this_window` record. Total K-12 GPU use: 5.88 h in two launches.
