# Mac rehearsal of the Qwen3.8 fine-tune on Qwen3.5-2B (local, free)

All times UTC, 15 Nov 2026. Python `~/.cache/k12-api-venv/bin/python`, `PYTHONPATH=src`, model `~/.cache/k12-models/Qwen3.5-2B`.
Outputs under `runs/mac_2b/`.

## 04:35 start
- torch 2.15.0, transformers 5.22.2, peft 0.21.2, sklearn 1.9.1, MPS available. Disk: 12 GiB free.
- Model config: `Qwen3_5ForConditionalGeneration`, text width 2048, 24 layers (18 linear_attention + 6 full_attention), 4.5 GB safetensors.
- Data: `ft_train.jsonl` 1,306 rows (fields trace_id, messages, target, label, final_ok); `ft_dev.jsonl` header + 301; 76 classes.
- Two unrelated CPU-bound Python jobs from other projects are running on this Mac (satpy.py, cmstest.py), each at ~100% of one core.
- Note before running: `features` default `--layers -1,-9,-17,-25,-33` assumes the 27B depth; the 2B has 25 hidden states, so -33 is out of range. Will pass layers explicitly.

## 04:37 speed test 1 (stopped at 04:44, memory problem)
Command: `train --model ~/.cache/k12-models/Qwen3.5-2B --train data/v11/ft_train.jsonl --output runs/mac_2b/speed --limit 12 --epochs 1 --accumulation 4 --save-minutes 0 --log-every 1 --classes data/v11/classes.json --device mps` (log `runs/mac_2b/speed/log.txt`).
- Model loads on MPS (float32, 8.9 GB of weights). LoRA targets: 186 modules (out_proj/in_proj_qkv/in_proj_z/in_proj_b/in_proj_a x18, gate/up/down_proj x24, q/k/v/o_proj x6); 16.8 M trainable; gradient checkpointing on.
- `inputs` line: keys = used = `input_ids, attention_mask, mm_token_type_ids, pixel_values, image_grid_thw`; 2,919 tokens, 157 target tokens. Photos reach the model.
- transformers warnings (expected): `causal_conv1d_fn` and `chunk_gated_delta_rule` fall back to reference PyTorch code.
- Update 1 (4 rows) at 115 s: text_loss 0.7477, head_loss 4.2888, 0.035 rows/s (29 s/row, includes warm-up). `max_gpu_gb` prints 0 (it only reads CUDA).
- Update 2 had not arrived 5 minutes later. `footprint` of the process: 31 GB, 30 GB of it "IOAccelerator (graphics)"; system swap 16.6 GB used. The MPS allocator keeps freed buffers of every size it has seen (sequence lengths differ per row), so the process grew until the Mac swapped. Killed.

## 04:46 diagnostic (scratch script, same functions as the module; 3 forward-only rows, 4 training rows without optimizer)
- Forward only (as `features`/`head`): 5.7 / 4.3 / 6.3 s for 2,762 / 2,369 / 3,103 tokens.
- Training row: forward 5.1-9.4 s + backward 11.9-37.9 s; MPS "current" memory 11.7-13.1 GB, but "driver" memory grew 16.9 -> 23.4 -> 33.4 GB over 4 rows (cache growth, then swap).
- Same with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5` (recommended max is 30.2 GB): driver memory stays 15-17 GB; rows take 23 / 25 / 24 / 33 s. Use these two variables for every MPS run. No code change needed.
- No MPS operation was unsupported; `PYTORCH_ENABLE_MPS_FALLBACK` not needed.

## 04:50 speed test 2: MPS out of memory with the 0.7 cap
`RuntimeError: MPS backend out of memory (MPS allocated: 16.95 GiB, other allocations: 276.45 MiB, max allowed: 19.66 GiB). Tried to allocate 3.06 GiB on shared pool.` at the first `backward()` (line 253 of the file as it was at 04:50). The 3 GiB blocks are the vocabulary logits for the whole 2,900-token sequence in float32 (2,919 x 248,320 x 4 bytes = 2.9 GB, and again for their gradient). Raised the cap.

## 04:53 speed test 3: completed (log `runs/mac_2b/speed/log3.txt`)
Same command with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55`.
- update 1 (rows 1-4): 101.6 s, text 0.7477, head 4.2888, running top-1 0.0
- update 2 (rows 5-8): 175.6 s, text 1.0099, head 6.5223, top-1 0.0
- update 3 (rows 9-12): 257.4 s, text 0.6117, head 3.5601, top-1 0.5
- 12 rows in 257 s = 21.5 s/row including warm-up, 18.5-20.5 s/row after the first four; 125 tokens/s. Losses finite. Twelve rows are too few to call a trend (each update is the mean of 4 different rows).
- Process footprint 23-24 GB (weights 8.9 GB float32; the rest is activations and the allocator's cache). System swap already 7-8 GB used by other things; no growth.
- Adapter saved: `adapter_model.safetensors` 67 MB, `head.pt` 0.65 MB, `train_report.json`.
- The file changed under me at 04:40 (coordinator added `--skip-failed-rows`, per-GPU `--gpu-memory`); test 3 ran the new version. Copy of the version used for the full run: `runs/mac_2b/q38_finetune_as_run_0500.py`.

## 04:58 reload check
`head --adapter runs/mac_2b/speed/adapter --head .../head.pt --requests data/v11/ft_dev.jsonl --limit 6 --device mps`: 6 rows in 29.8 s (5.0 s/row), bundle has ids, classes, p (6 x 76, rows sum to 1), final_ok (0.41-0.43), meta. Save -> load path works.

## 05:00 full rehearsal started
`train --model ~/.cache/k12-models/Qwen3.5-2B --train data/v11/ft_train.jsonl --output runs/mac_2b/full --epochs 1 --accumulation 4 --max-minutes 120 --save-minutes 30 --log-every 5 --classes data/v11/classes.json --device mps` (rank 16, lr 1e-4, head lr 1e-3, head weight 0.5, seed 2026: defaults). Accumulation 4 rather than the default 8 because only about 360 rows fit in 120 minutes and I want about 90 updates.
- 05:06 first log line: update 5, 20 rows, 401 s (20 s/row), text 0.8228, head 4.26. Footprint 23 GB.

## 05:03 fusion script dry run (CPU) and evicted files
- `scripts/fuse_v3.py --all-rows --dev-v3 runs/mac_2b/speed/head_dev4.npz` first failed: `EOFError: No data left in file` in `aligned_bundle` (core.py line 107). Two of its inputs were iCloud placeholders (`ls -lO` shows `compressed,dataless`, reads return 0 bytes): `runs/pod/dev-distill/validation_epoch_2.npz` and `runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz`. Restored both with `brctl download <path>` (88 KB and 20 KB, the user's own files from their iCloud). The three test bundles `runs/pod/text32-full/test.npz`, `runs/pod/text-full/test.npz` are also dataless (not needed here, left alone).
- After that the script runs (3 s): current fused 0.7962 (none 0.929, seen_pair 0.856, hard 0.616), matching the reference.

## 05:10 ceiling of a "final answer correct" signal (CPU, `runs/mac_2b/final_ok_rule.py`, report `runs/mac_2b/reports/final_ok_ceiling.json`)
- On dev the teacher flag equals "truth is NONE" on 293 of 301 rows (97.3%); in the 1,306 training rows it is true on 455 and NONE is the label on 441.
- The current fused system's top-1 "is NONE" already agrees with the flag on 94.7% of dev rows.
- Rule: multiply p(NONE) by e^a when the flag is true, by e^-b when false, (a, b) cross-fitted over the 5 question folds. With the teacher's true flag (a perfect predictor): 0.7962 -> 0.8151 (+0.019, interval +0.009 to +0.032); 11 rows improve, none get worse. That is the most this output can add through this rule.
- 05:13 update 10, 40 rows, 818 s (20.4 s/row), text 0.7682, head 4.0049, running top-1 0.4. Footprint 23-25 GB, swap steady.

## 05:18 queued after training (`runs/mac_2b/after_train.sh`)
1. `head --adapter runs/mac_2b/full/adapter --head .../head.pt --requests data/v11/ft_dev.jsonl --output runs/mac_2b/full/head_dev.npz`
2. `features` (untrained 2B) on the first 600 non-blind rows of `ft_train.jsonl` (scratch copy) + all 301 dev rows, `--layers=-1,-4,-7,-10,-13`, output `runs/mac_2b/features/untrained_600_dev.npz`. Only 600 fit rows because a forward pass costs 5 s here (1,500 rows would take 2.1 hours).
- Found: `--layers -1,-9` (with a space) is rejected by argparse: `error: argument --layers: expected one argument` (a value that starts with `-` and is not a plain number is taken for an option). It must be written `--layers=-1,-9,...`. The default value works when the option is omitted.
- Found: the default `--layers -1,-9,-17,-25,-33` is for the 27B (65 states); the 2B has 25 states, so `-33` would raise IndexError. Not a bug for the paid run.

## 05:23 full run crashed after about 62 rows (24 minutes lost; folder kept as `runs/mac_2b/full_crashed_0523`)
- Log: update 5 (20 rows) text 0.8228 head 4.26; update 10 (40 rows) text 0.7682 head 4.0049; update 15 (60 rows, 1,211 s) text 0.7758 head 4.0652, running top-1 0.4.
- Then at `backward()`: `RuntimeError: MPS backend out of memory (MPS allocated: 20.74 GiB, other allocations: 283.33 MiB, max allowed: 23.87 GiB). Tried to allocate 2.94 GiB on shared pool.`
- Not an unusually long row: rows 61-65 of the run order have 2,606-3,131 tokens, and a 3,736-token row had passed earlier. The allocator's cache was full of blocks of other sizes (fragmentation), so the 2.94 GiB block did not fit under the cap.
- Nothing had been saved (`--save-minutes 30`), and the script has no way to resume from a saved adapter. `--skip-failed-rows` would not have helped: it catches `torch.cuda.OutOfMemoryError` and `FloatingPointError`; the MPS error is a plain `RuntimeError`.
- Token counts of all 1,306 training rows (CPU, `runs/mac_2b/train_token_counts.json`): mean 2,676, median 2,691, 90th percentile 3,036, 99th 3,604, max 3,850, min 1,637; target tokens mean 251, max 713, none at the 900 cap. Encoding costs 0.012 s per row.
- The queued chain had started `features` at 05:23; stopped it at 05:33 to give the GPU back to training.

## 05:33 full run restarted through a Mac-only launcher
- `runs/mac_2b/run_mps.py`: imports `k12.q38_finetune`, wraps `encode` so `torch.mps.empty_cache()` runs before every row, then calls the module's `main()`. The module is untouched.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55`; `train ... --epochs 1 --accumulation 4 --max-minutes 110 --save-minutes 15 --log-every 5 --classes data/v11/classes.json --device mps`, output `runs/mac_2b/full`.
- Queued after it: `head` on the 301 dev rows, then untrained `features` on 400 fit rows + 301 dev rows (reduced from 600 to recover the lost time).

## 07:23 full run finished (`runs/mac_2b/full`, log `log.txt`)
- 356 rows seen (27% of one pass over 1,306), 89 updates of 4 rows, 6,633 s = 18.6 s/row, about 140 tokens/s; 0 rows skipped; footprint 20-23 GB between rows; adapter saved every 15 minutes and at the end (67 MB + head.pt 0.65 MB).
- Schedule followed the clock: lr factor rose to 0.87 by update 20, fell to 0.0 at update 89 (progress 1.005). The last update is applied with factor 0.
- Per 20 rows (update, rows, text loss, head loss [code cross-entropy + 0.5 x final_ok BCE], running head top-1, lr factor, seconds):
```
5 20 0.8228 4.26 0.1 0.3031 394.7
10 40 0.7682 4.0047 0.4 0.5902 791.8
15 60 0.7755 4.0648 0.4 0.848 1170.9
20 80 0.7463 5.2572 0.15 0.8707 1545.7
25 100 0.6656 3.7733 0.35 0.8057 1918.2
30 120 0.6442 3.3351 0.5 0.7324 2284.5
35 140 0.5827 3.1745 0.45 0.6507 2657.0
40 160 0.5949 3.8647 0.4 0.5648 3026.9
45 180 0.6546 3.5923 0.35 0.4733 3412.1
50 200 0.6085 3.8286 0.25 0.3928 3754.0
55 220 0.7725 3.9852 0.25 0.3092 4122.8
60 240 0.5461 3.7955 0.25 0.2272 4512.6
65 260 0.6226 3.849 0.25 0.1573 4886.2
70 280 0.7049 3.8809 0.25 0.0991 5254.7
75 300 0.6338 3.2979 0.5 0.0541 5613.4
80 320 0.5727 3.6299 0.5 0.0232 5956.9
85 340 0.6692 4.3667 0.3 0.0038 6342.3
89 356 0.4864 3.4871 0.5 0.0 6632.6
```
- 07:24 `head` on the 301 dev rows started (queued chain).

## 07:48 head on dev finished (`runs/mac_2b/full/head_dev.npz`, 301 rows, 1,425 s = 4.7 s/row)
Scored with `runs/mac_2b/score_bundle.py` (report `runs/mac_2b/reports/head_dev_score.json`):

| slice | rows | MAP@3 | top-1 | prior-only MAP@3 |
|---|---|---|---|---|
| all | 301 | 0.376 (95% interval 0.327-0.425) | 0.316 | 0.366 |
| none | 101 | 0.954 | 0.921 | 1.000 |
| seen_pair | 94 | 0.106 | 0.021 | 0.030 |
| new_pair | 84 | 0.071 | 0.000 | 0.065 |
| unseen_q | 22 | 0.038 | 0.000 | 0.038 |
| hard | 106 | 0.064 | 0.000 | 0.060 |

- "Prior-only" = the same three most common fit codes for every row. The head after 356 rows is at that level: it puts NONE first on 92% of dev rows (truth 34%) and uses 4 distinct codes as top-1. Mean probability on the true code 0.155.
- `final_ok` output against the teacher flag (true on 32.9% of dev rows): accuracy at 0.5 = 0.688 (always-false baseline 0.671), AUC 0.629; mean output 0.376 when true, 0.320 when false. p(NONE) from the same head has AUC 0.591 for the flag; the two outputs correlate 0.66. The output works mechanically (saved, loaded, written to the bundle as `final_ok`), but it has learned little.

## 07:49 fusion and rule (CPU)
- `scripts/fuse_v3.py --all-rows --dev-v3 runs/mac_2b/full/head_dev.npz --report runs/mac_2b/reports/fuse_head_dev.json`: current fused 0.7962; new cross-fitted 0.7835 (none 0.929, seen_pair 0.849, hard 0.586); difference -0.013 [-0.023, -0.004]; pass mark failed on both conditions. Fold choices all took the smallest weight on the new bundle the grid offers (w = 0.25); the grid has no w = 0, so a bundle with no information shows as a loss.
- `final_ok` rule with the 2B output (`runs/mac_2b/reports/final_ok_rule_head.json`): cross-fitted 0.7940 (-0.002 [-0.006, 0.000]); 0 rows improved, 2 worse. With the 2B p(NONE) instead: 0.7913.
- Head vs prior on the 200 non-NONE dev rows, NONE removed: true code in the head's top 3 of the 75 other codes on 21.5% of rows (prior order: 22.0%), top 5 31.5% (28.0%), top 10 55.5% (50.0%), median rank 9 (11). Mean p(NONE) 0.395 on NONE rows, 0.341 on other rows. Top-1 codes used: NONE 276, CAL-01 15, EXP-04 9, FRM-01 1. So after 356 rows the classification layer has learned the class frequencies and very little about the individual photo.

## 07:50 queued (`runs/mac_2b/after_features.sh`)
After the untrained features: (1) `runs/mac_2b/check_state_and_generate.py` (state at the last prompt token with and without the target appended; 2 greedy answers on dev rows), (2) `features --adapter runs/mac_2b/full/adapter` on dev (301) then the same 400 fit rows, `--deadline` 09:55 UTC, for a probe on the fine-tuned model.

## 08:24 stopped at the coordinator's request (closing entry)
- Killed `run_mps.py features` (untrained 2B features, at 450 of 701 rows after 2,043 s = 4.5 s/row), `after_train.sh`, `after_features.sh` and the memory watcher. `ps` shows no process of this work left. `features` writes its file only every 500 rows and at the end, so no features file exists and no probe was run.
- Not run: probe on untrained 2B features, features/probe of the fine-tuned model, the state check (same state at the last prompt token with and without the target appended), generated written answers, `merge`, a `--head-weight 0` or second-seed comparison.
- Kept: `runs/mac_2b/full/` (adapter 67 MB, head.pt, log, train_report.json, head_dev.npz, mem.txt), `runs/mac_2b/speed/`, `runs/mac_2b/full_crashed_0523/` (log only), `runs/mac_2b/reports/` (head_dev_score.json, fuse_head_dev.json, final_ok_ceiling.json, final_ok_rule_head.json, fuse_smoke6.json), scripts `run_mps.py`, `score_bundle.py`, `final_ok_rule.py`, `check_state_and_generate.py` (unused), `train_token_counts.json`, two copies of the module as run. Total 130 MB.
- No change was made to `src/k12/q38_finetune.py`, `scripts/q38_probe.py` or `scripts/fuse_v3.py`.
