# Measured K-12 results — 8 November 2026

Best verified public MAP@3: **0.59800**, submission **58133839**, improved from **0.58216**. The requested approximately 0.82 public score and top-three final rank have not been achieved.

| Experiment | Evaluation | MAP@3 | Decision |
|---|---|---:|---|
| Original Qwen3-VL-4B LoRA | Public leaderboard, submission 58126216 | 0.58216 | Reference |
| Temperature 2 + 25% question/topic prior | Public leaderboard, submission 58133839 | **0.59800** | Best verified submission |
| Original image classifier | 301-row development fold | 0.59524 | Base checkpoint retained |
| Same model, cached batched features | Same 301 rows, batched numerical path | 0.59801 | Reference for output-adapter comparisons |
| Linear head refit | Same cached development features | 0.59025 | Rejected |
| Rank-32 residual adapter | Same cached development features | 0.57032 | Rejected: overfitting |
| Regularized rank-8 adapter | Same cached development features | 0.59911 | Rejected: gain uncertain and below +0.01 screen |
| Qwen3.5-27B constrained evidence ranker | 301-row development fold | 0.58970 | Rejected by predeclared 0.65 gate; no test inference |
| Frozen-feature nearest-neighbour memory blend | Same cached development features | 0.59911 | Rejected: insufficient improvement |
| Conditional-prior odds correction | Question-group cross-fitting inside development | 0.60742 | Tied fixed current blend; difference interval spans zero |
| Question-text logistic prior plus images | Question-group cross-fitting inside development | 0.58416 | Rejected: worse than fixed current blend |

The current fixed blend scored 0.60742 when selected/evaluated on the full development fold. Re-selecting its calibration separately inside five question-group folds scored 0.59801. These local figures are not public scores or untouched image-model OOF: the same development fold was already used for image-epoch and prompt selection. Question-bootstrap intervals do not include that selection uncertainty.

The text-prior blend's cross-fit difference from the fixed current blend was −0.02326, with paired question-bootstrap 95% interval [−0.04324, −0.00458]. The conditional-prior correction difference was zero, with interval [−0.02062, +0.01846]. No further public submission was made from either experiment. Their exact recipes and metrics are saved in `src/k12/text_prior.py`, `src/k12/prior_correction.py` and the corresponding local artifacts.

## Internal Qwen modifications

The output-adapter results do not support adding more output capacity. The larger adapter increased fitting MAP@3 to 0.95969 while development fell to 0.57032. Stronger regularization restored roughly baseline performance but did not establish improvement.

An internal-LoRA reading/root-explanation objective was implemented and functionally tested on the actual model. Single-device and two-device smoke runs completed training, checkpoint saving and image-only prediction. The two-device test used 16 training rows and four validation/test rows; its quality score is not meaningful. A causal-prefix check confirmed that appending teacher responses changed prefix class probabilities by at most 0.000773 on the checked example. Gold transcriptions/explanations are response targets, never classification inputs.

The full 1,199-row development training run was launched at 16:19 UTC and **cancelled while queued** at approximately 16:26 UTC. It produced no full-development model-quality result. Another account job, `shivateja005/gemma-4-k3-timing`, was using GPU L4 ×4 outside this project's reservation registry. The user explicitly chose **“Keep Gemma running; continue K-12 on CPU.”** Gemma was left running; no further K-12 GPU launches are allowed under that instruction.

## Why the target remains out of reach

Rare labels are a real weakness, but fixing only the ≤5-support classes would raise the original development score to at most 0.62846 if every other prediction stayed unchanged. Perfectly fixing all ≤20-support classes would reach 0.73643. These are non-deployable diagnostic bounds. Common-class reading and root-cause reasoning must improve too.

The frozen 27B model was not a sufficiently strong teacher. It took 2,171 seconds for 301 development images and scored 0.58970. Forty-four validation examples were missed by both the 4B and 27B models' top-three lists. Even choosing the better complete ranking per row using gold labels would yield only 0.74363. This is an oracle diagnostic for that selector, not a score for a trained ensemble or a bound on future training/re-ranking.

The next capability experiment is the prepared internal reading/root-explanation objective, followed by a fixed-recipe independent evaluation if GPU use is authorized again. Its smoke success does not predict a quality gain. External pseudo-labelling and larger models have not demonstrated enough accuracy here to justify assuming that more data or parameters will reach 0.82.

## Verification and quota

- Kaggle's submissions endpoint reconfirmed 0.59800 and 0.58216 with COMPLETE status.
- The best CSV has all 3,498 test IDs and valid distinct official-code rankings.
- All **21** project tests passed, including causal input contracts, partition separation, saved adapters and account-wide job discovery.
- A read-only account-wide preflight rejected the unregistered Gemma session. The launcher now discovers recently run account notebooks, checks active statuses and refuses unreserved concurrency. GPU launches are also disabled explicitly by the user's CPU-only policy.
- Account quota was 8.86 hours used at 16:29 UTC; this includes unrelated account usage. There were zero active K-12 GPU jobs after cancellation. The unrelated Gemma job can continue changing the account total; K-12 CPU experiments consume no Kaggle GPU quota.

Artifacts and complete status: `reports/status.json`. Competition data, images, model weights and per-student prediction logs remain private/ignored.
