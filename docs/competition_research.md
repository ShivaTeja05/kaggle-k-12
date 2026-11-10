# Competition research and entry strategy

Checked 8 November 2026. Budget: Kaggle free compute only, with a user-imposed maximum of 10 GPU quota hours out of 30, counting quota already consumed. A weekly reset does not renew this task budget. Approximately 0.82 public MAP@3 is the immediate milestone. This is a research record, not a claim of a winning model.

## What must be delivered

Identify the root major error in each handwritten working, or `NONE`. Inputs are the student photo, question, optional question diagram, answer key, grade and topic. Submit `submission.csv` with `trace_id,root_code`; each prediction contains up to three distinct, space-separated codes in confidence order. With one true label, MAP@3 rewards rank 1/2/3 by 1, 1/2 and 1/3. [Official overview and evaluation](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/overview).

The overview names 17 November 2026 and says deadlines are 23:59 UTC unless noted; the header tooltip instead displays 00:00 UTC that date. Treat that discrepancy conservatively: make the final selection by **16 November**, then recheck the live deadline. In November, London time equals UTC. Never use the last hours for an untested run.

## Rules that affect our implementation

- One Kaggle account and one team; maximum team size five.
- Five submissions per day; select at most two final entries.
- External models/data must satisfy accessibility, cost and licensing conditions.
- No human prediction or hand-labelling of validation/test records for submissions.
- Keep competition data secure and use it only for this competition. Do not identify students or redistribute their images.
- No private sharing outside the team. Public code sharing belongs on the competition forums/notebooks.
- Winners must supply reproducible training/inference code, environment and documentation; the stated winner licence is Apache 2.0.
- Eligibility includes account, age, jurisdiction and sanctions requirements; the participant must check their own eligibility.

These are implementation-relevant summaries. The binding text is the [official rules](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/rules), including Kaggle's foundational rules.

The sponsor explicitly disallows external APIs in the final solution and requires offline reproducibility. APIs can support training/enrichment. [Organizer API ruling](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/discussion/746450). A later reply permits commercial API pseudo-labels on train/unlabelled data for competition use. [Follow-up clarification](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/discussion/751440). We use no paid APIs and transmit no competition records to them.

The sponsor's later announcement permits models up to the Nemotron 550B reference size and places no compute restriction. Our budget still restricts us to free hardware. [Model/compute announcement](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/discussion/748667).

## Data and validation implications

The official data description specifies 1,500 labelled, 3,498 test and 50,000 unlabelled rows, with 1,000 public and 2,498 private test rows. Splits are student-disjoint, share questions and are stratified by root code and grade. Training transcriptions and annotations are absent at inference. There is a long tail of rare labels and annotation ambiguity. [Official data description](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/data).

Our downloaded-data audit (`reports/data_audit.json`) additionally finds:

The root-level train/test CSVs and canonical `competition_data/` copies differ in every `topic_name` field but match on IDs, labels and the other compared fields. Local data now uses the canonical copies, verified against the GPU job's training-file hash. Record this version distinction when comparing notebook results.

| Observation | Modelling consequence |
|---|---|
| 95.71% of test questions occur in train | Question priors and same-question demonstrations are useful; unseen-question stress tests should be reported separately. |
| 506/1,500 rows are NONE | Measure false alarms on correct work, not only average ranking. |
| 13 singleton classes | Ordinary closed-set OOF classifiers cannot learn those classes in their held-out fold. |
| NOT-02 and PRO-03 appear in taxonomy but not labelled train | Preserve the full 76-way output space; do not invent training examples or infer test labels. |
| 30 correct-final-answer rows have a root error | Correct final answer must not force NONE. |
| 19 incorrect-final-answer rows have NONE | Final-answer mismatch must not force a misconception. |
| Question diagrams occur in 17.32% of test rows | Include both question and working images. |

The root is the pedagogically central cause, not mechanically the first error. [Host's root-attribution clarification](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/discussion/754093).

Our folds use labels for balancing and group normalized identical question/transcription pairs. Student identifiers are unavailable, so local CV cannot reproduce student grouping exactly. Fold 0 is development; fold 4 should remain an audit fold until the recipe is frozen. Report known/unseen questions, grade, label support and NONE mistakes. Avoid fitting a large blend search to the same validation rows used to report gains.

## Public notebooks reviewed

Downloaded source and logs are retained locally under ignored `references/` directories. Public artifacts are references, not proof our implementation reaches their scores.

| Source | Observed method / evidence | What to investigate or fix |
|---|---|---|
| [Hanh Tran: K-12 Baseline](https://www.kaggle.com/code/honghanhhh/k-12-baseline-lb-0-597) | Latest downloaded source combines question/topic priors, VLM OCR, sparse text classification and top-K letter-probability reranking. | Candidate recall is a hard ceiling. Measure it before reranking; train/evaluate with comparable OCR. Avoid trusting log-probability truncation as calibrated evidence. |
| [Avik Das: Multimodal Neural & Empirical Bayes](https://www.kaggle.com/code/avikdas567/multimodal-neural-empirical-bayes-math-diagnosis) | Indexed notebook page reports public 0.50299. Source includes feature fusion, priors and tree models. | Some narrative scores differ from each other; prefer actual saved predictions and executed logs. A generic vision feature can miss symbolic reasoning. |
| [Samanyu: misconception vlm pseudo](https://www.kaggle.com/code/samanyu1808/misconception-vlm-pseudo) | Qwen3-VL-4B LoRA, transcription generation and 76-way head, with 563 pseudo-labelled rows in downloaded run. Its logs show head validation 0.5333 and blend 0.5548 on 225 rows. | Head initialization occurs before seeding. Training head sees gold transcripts while inference uses generated ones. Single random holdout and long generation make gains uncertain and expensive. |
| [autokagon: public fast GPU inference](https://www.kaggle.com/code/autokagon/k12-public-fast-gpu-inference) | Adapts Samanyu's frozen artifacts; documents batching, token-prefix checks and selective singleton scoring. | It is inference adaptation, not independent training evidence. Preserve attribution; speed work needs numerical rank checks. We did not adopt these third-party weights. |

The source of the joint transcription model is valuable evidence of a training/inference mismatch; the numerical impact is a hypothesis until an ablation measures it. Likewise, a fast notebook's existence does not establish a high private score.

The baseline page's rendered version 9 reports public 0.55366 and best version 11 at 0.59716; the API source and rendered page can expose different versions. Its executed logs show CUDA/library compatibility work, OOM warnings, and generation-length truncation. The displayed OCR similarity distance is 0.471, a rough `SequenceMatcher` statistic rather than a standard character error rate. The page has zero comments, so there is no participant comment thread to infer from. Our responses are explicit SDPA/FP16 settings, offline dependency checks, limited-resolution image tests, and avoiding mandatory transcript generation.

## Research and GitHub work

1. [CoLearn's task-specific experiments](https://colearnlab.com/training-llms-to-identify-student-misconceptions-from-their-written-work/) find that fine-tuning improves diagnosis and reduces false alarms. Their benchmark is not this competition's MAP@3 leaderboard; its accuracy numbers cannot be compared directly.
2. [MathCog, arXiv:2504.00843v2](https://arxiv.org/abs/2504.00843) studies diagnosis from handwritten work and reports difficulties with vague evidence and hallucinated inferences. Our implication: preserve visible work, avoid inventing omitted steps, and distinguish evidence extraction from diagnosis. This is a related task, not a comparable leaderboard benchmark.
3. [Eedi second-place solution](https://github.com/wangqihanginthesky/Eedi_kaggle/blob/main/Solution.md) uses retrieval, synthetic examples, hard negatives and reranking; randomized choices and order reversal address ranking sensitivity. Those ideas motivate experiments here, but Eedi's multiple-choice setting differs from handwritten workings and its larger models exceed our free-compute plan.
4. [MAP silver-medal repository](https://github.com/tossowski/MAP) documents multi-model training, preserving sequence length, augmentation and failed experiments. Transfer the validation discipline; do not transfer task-specific hard label exclusions to this taxonomy.
5. [Official Qwen3-VL repository](https://github.com/QwenLM/Qwen3-VL) provides model and fine-tuning references under Apache 2.0. We selected its official Kaggle `4b-instruct/1` variant, whose metadata confirms Apache 2.0 and fine-tuning support. [Transformers Qwen3-VL documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) supplies the multimodal API.

No directly matching competition GitHub solution was verified in the searches performed. We found the relevant implementation evidence primarily in Kaggle notebooks and related competition repositories. Do not describe the search as exhaustive.

## Top-three strategy under free compute

The public leaderboard read on 8 November showed 0.84450, 0.84200 and 0.83916 at ranks 1–3. These are a dated public benchmark, not final private targets. [Live leaderboard](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/leaderboard).

The first CPU prior scores 0.48533 OOF (question-bootstrap interval approximately 0.45773–0.51284). It lacks student-work evidence and is not competitive for a prize. Its top-3 recall is 0.67267 and singleton MAP@3 is zero. This establishes why more than metadata is necessary.

The next experiments are deliberately different:

- Direct Qwen3-VL image classifier with language LoRA, root-code head and auxiliary answer-correctness supervision. Train and validation consume identical input fields. No transcription generation cost or gold-transcription dependency. Seed before every head initialization and save exact environment/config hashes.
- Taxonomy-conditioned image diagnosis covering all codes, with fitting-fold demonstrations only. This can propose rare classes that a learned classifier misses. Its output is a ranking, not calibrated probabilities.
- OOF-calibrated blending, only after independent models show complementary errors. Reserve the audit fold and retain an unblended reference.
- If results justify it, use a second open model or seed and confidence/agreement-filtered pseudo-labels from the supplied unlabelled pool. Never pseudo-label held-out rows and feed them back into their own validation training; regenerate teachers per fold or hold a truly untouched audit set.

The classifier and generative ranker are experiments, not established improvements. Promotion requires genuine held-out image results, rare-class analysis and a complete offline run. Additional GPU jobs must fit the account's remaining free quota; do not launch paid infrastructure or call paid APIs.

Execution findings on 8 November: the first classifier smoke run exposed an incompatible optional `torchao` dispatcher in Kaggle's PEFT environment. Disabling that unused dispatcher for unquantized FP16 weights allowed the smoke run to complete training and inference. It trained eight rows, predicted four validation and four test rows, and peaked at 9.64 GB; its zero score on four validation rows is not a performance estimate. The full three-epoch classifier was then launched privately. The first generative ranker run failed before its first prediction with a single-T4 OOM on the taxonomy prompt. Version 2 balances the model over both T4s and requests final-token logits only; its quality and complete execution remain to be verified. These job states are tracked in `reports/status.json`.

## Dated execution plan

| London date | Checkpoint |
|---|---|
| 8 Nov | Rules/research audit, real CPU baseline, private GPU smoke test, launch first image experiment if smoke passes. |
| 9–10 Nov | Compare image classifier and taxonomy ranker on development images; examine error slices and runtime. |
| 11–12 Nov | Spend remaining free quota on the most promising independent model or targeted pseudo-label experiment. |
| 13–14 Nov | Freeze prompt/model settings, evaluate reserved audit fold, measure ensemble gains with paired question bootstrap. |
| 15 Nov | Full-training refit of selected recipe, offline inference, exact submission validation. |
| 16 Nov | Select two final entries and retain complete reproduction artifacts; internal deadline. |

This schedule is a plan, not an automation or promise of unattended daily execution.

## If we miss top three: diagnose rather than invent a story

| Evidence | Likely cause | Test / corrective experiment |
|---|---|---|
| Gold-text model much stronger than image/generated-text model | Handwriting/diagram reading bottleneck | Compare on the same held-out labelled images; measure operator/number transcription errors. |
| NONE false alarms dominate | Over-diagnosis or bad calibration | Tune a soft NONE offset on development data; check audit MAP@3 and false-negative rate. Do not force answer-match rules. |
| True code missing from candidate set | Retrieval ceiling | Report recall@K by support band; expand taxonomy coverage before improving reranker. |
| Rare labels fail despite good common-label accuracy | Sparse supervision | Try definitions, fitting-fold demonstrations, filtered synthetic/pseudo examples. |
| Development gains disappear on audit | Model/prompt/blend overfitting | Reduce search, freeze recipe and use the untouched audit result. |
| Public improves but private drops | Leaderboard overfitting, split variance or distribution shift | Compare preselected offline evidence and final private results; avoid a causal claim from public score alone. |
| Crossed-out or ambiguous work triggers wrong roots | Evidence interpretation and annotation ambiguity | Study existing training annotations and held-out confusion, without creating human test labels. |
| Run exceeds notebook time/VRAM | Resolution, token length or repeated inference | Profile batches, cache within the private run, reduce generation, compare quality before accepting speed changes. |

We cannot establish the actual reason for a final placing before final results exist. Keep run manifests, full OOF probabilities, model-selection history and scored submissions so the later analysis has evidence.
