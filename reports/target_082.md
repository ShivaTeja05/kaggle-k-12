# Working toward 0.82 on free Kaggle compute

Latest verified state, 8 November 16:29 UTC: **public MAP@3 0.59800** (submission 58133839), improved from 0.58216 by CPU calibration. The 27B model completed at 0.58970 development MAP@3 and skipped test inference under its quality gate. Output adapters, feature memory and conditional-prior odds correction did not establish a gain over the current recipe. Reading/root-explanation training passed functional smoke tests; its full run was cancelled while queued after discovering an unrelated Gemma GPU job. The user chose to keep Gemma running and continue K-12 on CPU. Quota at the account-wide guard verification was 8.86 hours used; the guard rejected unregistered concurrency. Further K-12 GPU launches are disabled. The historical entries below describe earlier states; current machine-readable status is `reports/status.json`.

8 November 2026. The immediate target is approximately **0.82 public MAP@3**; the longer-term objective remains top three on the final private leaderboard. Neither has been achieved yet. The user set a hard GPU cap of **10 of the account's 30 weekly hours**, counting quota already used. The latest check at the time of that instruction showed 1.21 hours used and 28.79 remaining.

| Step | Evidence and action | State |
|---|---|---|
| Establish a real reference | Question/topic prior scored 0.48533 full OOF; fold-0 reference is 0.49723. | Complete |
| Remove training/test input mismatch | The image classifier consumes only the fields available at test time; correctness is supervision, not an input. | Execution verified by smoke test; quality pending |
| Fix dependency failure | Disable incompatible unused torchao dispatch for unquantized FP16 LoRA. | Smoke passed |
| Fix long-prompt OOM | Split taxonomy model over both T4s and request final-token vocabulary logits only. | Live inference passed 50 rows without the previous OOM |
| Measure image quality | Classifier and 301-row taxonomy development experiments run on the same held-out fold. | Running |
| Fix launch reporting | Kaggle returned exit zero on a concurrent-GPU rejection; wrapper now requires the explicit successful-push response. | Tested |
| Respect free compute | Quota check showed 29.16 of 30 GPU hours remaining, refreshing 14 November. Two concurrent GPU sessions are allowed. | No paid resources used |
| Test a stronger model | Prepare a random 48-row Qwen3.5-9B evidence-based pilot; its Apache 2.0 Kaggle model version was verified. | Prepared; GPU slot unavailable at initial launch |
| Fix inference throughput | Current Transformers ranker takes about ten seconds/image. A CPU-only job prepares an offline vLLM wheel bundle for batched inference. | Dependency job running |
| Evaluate before promotion | Compare each completed image run with the prior on identical IDs; report rare classes, NONE false alarms and question-bootstrap uncertainty. | Evaluation tool tested |
| Separate reading from reasoning | Gold-transcription word/character classifier scored 0.534 full OOF. This is a diagnostic, not deployable, and not a ceiling for stronger reasoning models. | Complete |
| Score an actual submission | Generate and validate all 3,498 rows only after a competitive validation result; obtain the public score through Kaggle. | Pending |

The 48-row pilot is a screening experiment, not enough evidence to establish 0.82. The 301-row development fold can guide changes, but repeated tuning makes its best observed score optimistic. Reserve fold 4 for a fixed-recipe check. Never use test labels, leaderboard probing or handwritten test annotations as training signal.

The extra public notebooks reviewed in this pass were not adopted on the strength of titles or narrative scores. In particular, substituting question text for missing student working does not solve the image-reading task.

The live logs are saved under ignored `runs/live/`. No recurring automation has been created.

## First completed image results

- Qwen3-VL-4B taxonomy ranker: 0.38483 MAP@3 on the full 301-row development fold, versus the same-row prior at 0.49723. Rejected. NONE false-alarm rate was 85.15%; error-only MAP@3 was 0.36917. Fixing only the NONE rows would yield a diagnostic 0.58084, so error-code reasoning also needs improvement.
- Direct image classifier, epoch 1: 0.43355 on the same 301 rows. Not promoted; later epochs remain running under their existing timeout.
- Qwen3.5-9B pilot: launched at 05:57 UTC with a 1,800-second maximum and a budget check projecting 8.456 hours in the worst case, including the active classifier. It uses a balanced error-existence check, fitting-fold teacher explanations and batched offline inference. Quality pending.

## 9B pilot and output repair

The 48-row 9B pilot completed: MAP@3 0.55903 against the same-row prior at 0.53125. The paired question-bootstrap difference interval spans zero (−0.0745 to +0.1418), so this is not established improvement. Every response supplied fewer than three distinct codes; eight hit the generation limit. NONE false alarms fell to 46.67%, but some explanations contradicted elementary arithmetic.

A corrected full-301-row run launched at 06:21 UTC with a 45-minute timeout. It removes the single-root `ROOT:` prompt, requests code-only output and uses vLLM's documented regex constraint to require three official-code slots. Duplicate slots remain measured and repaired using fitting-only priors. This fixes formatting, not mathematical competence; quality remains unmeasured. GPU usage at admission was 3.14 hours; conservative worst-case total including the active classifier and margins was 8.9928 hours, below the 10-hour cap.

Implementation reference: https://docs.vllm.ai/en/v0.31.0/features/structured_outputs/

## Completed training and first public score (11:45 UTC)

The classifier completed all three epochs: held-out MAP@3 was 0.43355, 0.58638, then 0.59524 on the same 301 rows. Epoch 3 improved the same-row prior by 0.09801, with paired question-bootstrap interval [0.05684, 0.13951], excluding model-selection uncertainty. Rare classes remain weak (singleton rows 0.0, 2–5 support 0.16667); grade 10 MAP@3 is 0.41429. All 3,498 test predictions passed format/ID/code validation.

Submission 58126216 completed at Kaggle with **0.58216 public MAP@3**. This is the first actual leaderboard score; the approximately 0.82 objective remains unmet. An initial CLI request returned HTTP 400; the current official Python client successfully submitted and the submissions endpoint verified completion.

The corrected code-only 9B full-fold run scored **0.45515**. Formatting improved (zero truncations, only 12 duplicate-code repairs), but 53.5% of error rows were predicted NONE first. It is rejected and did not generate test predictions.

At 11:45 UTC, quota showed **6.00 hours used, 24.00 remaining** and no active GPU jobs. A larger 27B model run was then admitted with a 10,800-second outer timeout: worst-case total **9.35 hours**, including margins. It uses brief constrained evidence before three ranked codes, preserves the checkpoint's actual compressed-tensors format, and uses eager mode to avoid lengthy graph compilation. It validates all 301 rows first. Full-test generation requires validation MAP@3 >=0.65 AND a conservative runtime estimate fitting the remaining post-install budget. No quality claim exists for this candidate yet. No additional GPU job may bypass the cap guard.
