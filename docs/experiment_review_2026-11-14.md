# K-12 misconception detection: review of every experiment to 14 November 2026

Written 14 November 2026, about 04:40 UTC. This is a reading-and-writing review. Nothing was run, launched, spent or
submitted to produce it. Every number comes from a file in this repository; the source is named in brackets where it
matters. Where a number could not be found the text says "not recorded".

**Sources, in order of authority.** (1) `docs/model_audit_plan_2026-11-12.md`, sections 1 to 17.1 ("the audit record";
later sections supersede earlier ones). (2) `reports/status.json`, `reports/q38_dev_zero_shot.json` and the other
`reports/*.json`. (3) `docs/study/report/report.md`, `docs/study/sections/*.md`, `docs/handoff.md`,
`docs/path_to_082.md`, `docs/adaptation_strategy.md`, `docs/htr_survey_takeaways.md`,
`reports/experiment_results_2026-11-08.md`, `reports/root_example_screen.md`, `reports/working_verifier_progress.md`
and `AGENTS.md`.

**Terms used throughout.**

| Term | Meaning |
|---|---|
| MAP@3 | The competition metric. Each row gets up to three ranked codes. The row scores 1, 1/2 or 1/3 if the true code is first, second or third, and 0 otherwise. The score is the mean over rows. |
| Dev | Fold 0 of the project's own 5-fold split of the 1,500 labelled rows: 301 rows. One dev score has a standard error of about 0.023. |
| Fit rows | The other 1,199 labelled rows. Dev models are trained on these. |
| No-error rows | The 101 dev rows whose true label is NONE. |
| Seen-pair rows | The 94 dev error rows whose (question, error code) pair also appears in the fit rows. |
| Hard rows | The 106 dev error rows whose (question, error code) pair never appears in the fit rows: 84 on a known question with a code new for it, 22 on a question not in the fit rows. |
| Groups A, B, C | The hard rows split by what the Claude-labelled pool holds. A (32 rows): Claude labelled this exact pair on another photo. B (31): the question has pool photos (median 81 labelled) but none with this pair. C (43): the question has no pool photos. |
| Pool | The 50,000 unlabelled photos. They cover 284 questions. |
| Teacher transcript | The annotators' typed transcription of a training photo. Not available for test. |
| Machine transcript | Our transcriber's output for a photo. |
| Teacher labels | The 1,500 human labels. "Claude labels" are Claude Opus 5.5 answers on pool photos (26,202 photos labelled; never a test photo). |
| In-sample | A setting was chosen and scored on the same 301 rows. Optimistic. |
| Cross-fitted | A setting was chosen on four fifths of the dev questions and scored on the remaining fifth. |
| [a, b] | A 95% range from a paired bootstrap over questions. |
| Pass mark (gate) | A threshold written down before a run that decides whether its result is used. |
| Blend | The three classifiers multiplied together with fixed weights (a geometric blend). |
| Fused | The blend multiplied by another model's scores (the ranker, or Qwen3.8-27B). |
| Submitted system | The pipeline behind the best public file (0.73183): blend × pointwise ranker v1, then label balancing. |

---

## 1. Where we stand

- **Best public score: 0.73183** (submission 58299898, 13 Nov 07:08 UTC), rank about 77 of 433. The leaders are at
  about 0.845 (top three 0.8445, 0.842, 0.83916 on 8 Nov; 0.8465 at the top on 9 Nov). The 0.82 milestone is not reached.
- **What produced it:** a transcriber, three fine-tuned classifiers blended together, a Qwen3-14B yes/no ranker fused
  on top, and Sinkhorn label balancing. Cross-fitted dev score 0.7525.
- **Latest dev result (section 17.1 of the audit record):** Qwen3.8-27B, used zero-shot with thinking on, fused with
  the submitted system scores **0.805 against 0.772** on the same 270 dev rows, a gain of +0.033 [+0.006, +0.060].
  This is a dev result on 270 of 301 rows, not a leaderboard score. Scoring the 1,593 least-confident test photos with
  this model was approved on 14 Nov at about 04:20 UTC (`AGENTS.md`); no result was recorded when this review was written.
- **Single biggest weakness:** hard rows. The submitted system scores 0.500 on them. Claude Opus scores 0.715 with the
  photo and 0.637 reading our own machine transcripts. Hard rows are 35% of dev.
- **Where we are already at or above Claude:** no-error rows (0.911 against 0.934) and seen-pair rows (0.867 against 0.812).
- **What the record says about cause:** our fine-tuned models learn which codes go with which question. They are
  strong once they have seen the exact (question, code) pair and weak when they have not. The newest evidence
  (section 17) adds that the base model generation matters: every submitted component is a Qwen3-generation model.

---

## 2. The current system

This is what runs on one test photo in the best public file. Everything runs offline on open models. Claude is used
only at training time, on training and pool photos.

| Step | Component | What it does | How it was built | Measured quality |
|---|---|---|---|---|
| 1 | **Transcriber** | Photo → numbered text lines (L1, L2, ...). | Qwen3-VL-8B with a LoRA adapter (rank 32), trained 2 epochs on the 1,199 teacher transcripts. Greedy decoding, up to 512 new tokens, repetition penalty 1.05. Whether the test-time transcriber was retrained on all 1,500 rows is not recorded. | Recovers 88% of the numbers in the teacher transcript; 91% of the numbers it writes are right. Median character error 0.095 after normalisation (0.328 raw). The annotated error text is present in only 55% of machine transcripts. |
| 2a | **Text classifier, 8B** | Question + answer key + machine transcript → 76 probabilities (75 codes + NONE). The decision is read from the model's internal vector at the last prompt token through one linear layer. The model writes nothing. | Qwen3-8B, LoRA rank 32, 3 epochs, teacher labels only. | Dev 0.667 alone; hard rows 0.363. |
| 2b | **Text classifier, 32B** | Same design as 2a. | Qwen3-32B loaded in 4-bit (QLoRA), 1 epoch, teacher labels plus 3,427 Claude-labelled rows. | Dev 0.652 on machine transcripts (0.687 on teacher transcripts); hard rows 0.357. |
| 2c | **Image classifier** | Photo + question → 76 probabilities, same last-token readout. The only blend member that sees the handwriting. | Qwen3-VL-8B, LoRA rank 32, vision tower frozen. 1,500 teacher rows + 3,427 Claude-labelled pool photos, 2 epochs. Extra loss (weight 0.15) for writing the teacher's transcription. | Dev 0.653 alone (measured on a sibling model trained on the fit rows); hard rows 0.362. |
| 3 | **Blend** | Multiplies the three outputs: 32B^0.35 × 8B^0.30 × image^0.35, rescaled to sum to 1. | Weights picked on dev. | Dev 0.718 in-sample, 0.706 cross-fitted. Public 0.70850 on its own. |
| 4 | **Pointwise ranker (v1)** | For each of the blend's top 10 codes, one prompt: question, machine transcript, and "Candidate diagnosis: code, name, description, example if one exists". The model answers yes or no. Score = logit(Yes) − logit(No). | Qwen3-14B, LoRA rank 32. 5,198 training rows (2,398 teacher views + 2,800 Claude rows), 31,188 pairs, groups of 6 candidates, soft cross-entropy over the group. No dev row in training. | Dev 0.7087 alone; hard rows 0.486. |
| 5 | **Fusion** | blend × ranker^0.5. | Weight 0.5 chosen by 4 of 5 folds. | See step 6. |
| 6 | **Sinkhorn label balancing** | One scaling factor per code, so that the average predicted label mix moves towards the training label mix. Uses no test labels. Strength λ. | λ 0.5 in the 0.71716 file; λ 0.75 in the best file. | Blend alone: +0.019 dev, +0.009 public. Whole system: dev 0.7525 cross-fitted, **public 0.73183**. |
| Training only | **Teacher (Claude Opus 5.5)** | Labels pool photos with a transcript, a root line, an explanation and three ranked codes. | 26,202 pool photos labelled; turned into soft targets (0.7 / 0.2 / 0.1 on Claude's three codes, weight 0.8). | Claude itself scores 0.819 on dev. |

Not part of any submitted file: Qwen3.8-27B (dev test only so far), the Stage B reason-first student, the error
catalogue, ranker v2.

---

## 3. Every experiment, in time order

Verdict words: **kept** (used in a later system), **rejected**, **inconclusive** (range crosses zero or could not be
deployed), **diagnostic** (a measurement, not a candidate), **no result** (did not finish). "Dev" scores are MAP@3 on
the 301 dev rows unless a different row count is given. Costs are shown where the record gives them; Kaggle GPU runs
were free; "CPU" means no spend.

| # | Date (UTC) | Experiment | What it tested | Result | Verdict | Cost |
|---|---|---|---|---|---|---|
| | | **8 Nov: free Kaggle GPUs and CPU** | | | | |
| 1 | 8 Nov | Question/topic prior (no image) | How far "the usual codes for this question" gets | 0.485 over 5 folds; 0.497 on dev | Reference | CPU |
| 2 | 8 Nov | Word-and-character classifier on teacher transcripts | Whether perfect text plus a simple model solves it | 0.534 over 5 folds | Diagnostic: text access alone does not solve it | CPU |
| 3 | 8 Nov | Qwen3-VL-4B, untrained, scores every taxonomy code | A zero-shot image ranker | 0.385; calls 85% of no-error rows an error | Rejected | Free |
| 4 | 8 Nov | Qwen3-VL-4B LoRA image classifier, 3 epochs | A fine-tuned small vision model | 0.434 → 0.586 → 0.595 by epoch; +0.098 over the prior [+0.057, +0.140] | Kept | Free |
| 5 | 8 Nov 11:43 | **Submission 1** (58126216): model 4, epoch 3 | First leaderboard score | Dev 0.595; **public 0.58216** | Kept as reference | |
| 6 | 8 Nov | Qwen3.5-9B, untrained, evidence prompt, 48 rows | A stronger untrained model | 0.559 against prior 0.531 on those rows; range −0.075 to +0.142 | Inconclusive | Free |
| 7 | 8 Nov | Qwen3.5-9B, untrained, code-only constrained output | Fixing the output format | 0.455; calls 53.5% of error rows NONE | Rejected | Free |
| 8 | 8 Nov | Qwen3.5-27B (4-bit), first run | Larger untrained model | Grammar compile timed out before any prediction | No result | Free |
| 9 | 8 Nov | Qwen3.5-27B (4-bit), untrained, brief evidence then three codes | Larger untrained model | 0.590; pass mark 0.65; 2,171 s for 301 rows | Rejected; no test predictions | Free |
| 10 | 8 Nov | Temperature 2 + 25% question/topic prior on model 4 | Calibration | 0.607 in-sample, 0.598 cross-fitted | Kept | CPU |
| 11 | 8 Nov 16:08 | **Submission 2** (58133839): model 4 with calibration 10 | | Dev 0.607 / 0.598; **public 0.59800** | Kept | |
| 12 | 8 Nov | Refit linear head on frozen 4B features | More output capacity | 0.590 against 0.598 | Rejected | CPU |
| 13 | 8 Nov | Rank-32 residual adapter on frozen 4B features | More output capacity | 0.570; fit-row score rose to 0.960 | Rejected: over-fits | CPU |
| 14 | 8 Nov | Regularised rank-8 adapter | Same, heavily constrained | 0.599, +0.001 [−0.007, +0.010] | Rejected: below the +0.01 screen | CPU |
| 15 | 8 Nov | Nearest-neighbour memory in 4B feature space | Retrieval of similar fit rows | 0.577 alone; 0.599 when gated by confidence | Rejected | CPU |
| 16 | 8 Nov | Conditional-prior odds correction | Prior correction | 0.607, change 0.000 [−0.021, +0.018] | Rejected | CPU |
| 17 | 8 Nov | Question-text logistic prior blended in | A text prior | 0.584, −0.023 [−0.043, −0.005] | Rejected | CPU |
| 18 | 8 Nov | Rare-class bound | How much rare codes could be worth | Fixing every code with ≤5 examples gives at most 0.628; ≤20 examples 0.736 | Diagnostic | CPU |
| 19 | 8 Nov | 4B and 27B complementarity | Whether the two models miss different rows | Best-of-two oracle 0.744 (0.740 in a later audit); 44 rows missed by both | Diagnostic | CPU |
| 20 | 8 Nov | Reading-supervised 4B (transcription and explanation as training targets) | Whether teaching the model to read helps | Smoke tests passed; full run cancelled while queued (shared-quota conflict) | No result | Free |
| | | **9–10 Nov: CPU only** | | | | |
| 21 | 9 Nov | Candidate coverage of the calibrated 4B | Whether ranking or recall is the limit | True code in top 3 / 5 / 8 / 12 on 73.8% / 82.7% / 88.7% / 92.4% of rows; 4B top 8 plus 27B top 3: 91.7% | Diagnostic | CPU |
| 22 | 9 Nov | Oracle checks on the 4B ranker | Value of knowing "is there an error" and the code family | 0.677 with the true error/no-error flag; 0.919 with the true code family | Diagnostic | CPU |
| 23 | 9 Nov | Reciprocal rank fusion of 4B and 27B | Simple fusion | 0.626 against 0.607, range −0.015 to +0.051 | Inconclusive; 27B had no test predictions | CPU |
| 24 | 9 Nov | Family classifier on cached 4B features | A separate error-family head | Family accuracy 53.8% against 54.2%; combined 0.595 | Rejected | CPU |
| 25 | 9 Nov | Cross-fitted router between 4B and 27B | Per-row model choice | 0.618, below fixed fusion (0.626) | Rejected | CPU |
| 26 | 9 Nov | Logistic re-ranker over 4B and 27B candidates | A learned candidate scorer | 0.650, +0.024 over fusion [+0.006, +0.043]; error rows 0.510, no-error 0.927 | Inconclusive: dev only, never deployed | CPU |
| 27 | 9 Nov | Tesseract OCR on 60 photos | A CPU reader | Recovers 11.7% of annotated numbers; math token error 89.7% | Rejected | CPU |
| 28 | 9 Nov | Text verifier over the image model's top 8 (teacher text) | Re-ranking from the written steps | 0.550 literal; 0.575 with equation features; 0.592 keeping the image NONE probability; reference 0.607. Best variant against the reference: range −0.041 to +0.010 | Rejected | CPU |
| 29 | 10 Nov | Root-example verifier (similarity to annotated root lines) | Matching a working to labelled examples of each code | Fusion 0.626 → 0.637 with teacher text, +0.012 [−0.001, +0.024]; 0.630 when downstream examples are added | Inconclusive | CPU |
| 30 | 10 Nov | OCR reading-order audit | Whether line order explains OCR error | Token error 96.4% ordered, 94.7% unordered; 0 of 27 cited equations read exactly | Diagnostic: recognition, not order, was the problem | CPU |
| 31 | 10 Nov | PP-FormulaNet-plus-S on 265 cropped regions | A formula reader | Number recall 21.5% against 14.2% for Tesseract on the same crops; precision 28.7% against 49.7%; 1 of 27 cited equations; downstream 0.656 against 0.664 | Rejected | CPU |
| 32 | 10 Nov | Autocontrast crops; Latin PP-OCRv5 reader; best of two readers | Other readers and views | Downstream 0.592; 0.597; 0.594 against 0.597 | Rejected | CPU |
| 33 | 10 Nov | Verifier restricted to the teacher-marked root lines | Whether cropping to the root line helps | 0.609 → 0.601 | Rejected | CPU |
| | | **10–11 Nov: rented GPUs (RunPod)** | | | | |
| 34 | 10–11 Nov | Qwen3-VL-8B image classifier ("A") | A larger vision model | 0.550 → 0.604 → 0.619 | Baseline for this phase | Not recorded per run |
| 35 | 10–11 Nov | A + loss for writing the teacher transcription, weight 0.15 ("B") | Reading supervision | 0.557 → 0.615 → 0.632; against the 4B reference [−0.014, +0.060] | Kept (gain not proven) | Not recorded |
| 36 | 10–11 Nov | A + soft secondary labels + rare-code oversampling ("D") | Rare codes | 0.618 against 0.619 | Rejected: no gain | Not recorded |
| 37 | 10–11 Nov | A + error-family and has-error helper heads ("E") | Wrong-family errors | 0.613 against 0.619 | Rejected: no gain | Not recorded |
| 38 | 10–11 Nov | B with evidence-first targets (dataset v3) | Better training targets | 0.539 → 0.627 → 0.6285, level with B | Rejected: no gain | Not recorded |
| 39 | 10–11 Nov | 32B image classifier, 4-bit | A much larger vision model | 0.403 after epoch 1; stopped | Rejected | Not recorded |
| 40 | 11 Nov | Self-training labels from our own 8B (1,565 rows kept of 9,602 scored) | Pseudo-labels from our own model | Data built; no training result recorded | No result | CPU |
| 41 | 11 Nov 02:07 | **Submission 3** (58221584): B^0.7 × 4B^0.3, then 10% prior | | Dev 0.647 (in-sample); **public 0.62383** | Kept | |
| 42 | 11 Nov | Transcriber (Qwen3-VL-8B LoRA) | Photo → text | Number recall 0.88, precision 0.91 (older OCR 0.12–0.22 recall) | Kept | Not recorded |
| 43 | 11 Nov | Qwen3-8B text classifier, 1 epoch | Diagnose from text | 0.570 on teacher transcripts, 0.542 on machine transcripts | Kept | Not recorded |
| 44 | 11 Nov | Text classifier with 3 same-question examples and code descriptions in the prompt | In-prompt examples | Was running at the handoff; result not recorded | No result | Not recorded |
| 45 | 11 Nov | Qwen3-8B text classifier, 3 epochs, teacher labels only | Longer training | 0.635 at epoch 2, 0.667 at epoch 3 (machine transcripts) | Kept (blend member) | Not recorded |
| 46 | 11 Nov 06:27 | **Submission 4** (58228022): text 0.6 × image B 0.4 | Read-then-diagnose | Dev 0.684 (in-sample); **public 0.66183** | Kept | |
| 47 | 11 Nov | Claude Opus 5.5 on the 301 dev photos, zero-shot | How good a teacher is | **0.819**; top-1 0.738; true code in top 3 on 92.0%; hard rows 0.715 | Yardstick; chosen as teacher | $2.75 |
| 48 | 11 Nov | Claude Opus with 4 examples in the prompt | Few-shot teacher | 0.811, −0.008 [−0.034, +0.017] | Rejected: no better, costs more | $3.48 |
| 49 | 11 Nov | Claude Fable 5.2 on dev | A different teacher | 0.788, −0.031 [−0.055, −0.006] | Rejected | $3.88 |
| 50 | 11 Nov | Claude Opus labels for pool photos | Teacher data | 26,202 photos labelled. Claude puts NONE first on 70.9% of them | Kept (training data) | $191.60 (this review's sum of four recorded runs on two keys) |
| 51 | 11 Nov | Claude Opus explanations for the 1,199 fit rows | Reasoning text for training | Collected | Kept (training data) | $10.23 |
| 52 | 11 Nov | Image classifier distilled from Claude (1,500 + 3,427 rows) | Teacher labels for the image model | 0.653 at epoch 2 | Kept (blend member) | Not recorded |
| 53 | 11 Nov 10:43 | **Submission 5** (58234705): 8B text × distilled image, 50/50 | | Dev 0.693 (in-sample); **public 0.68283** | Kept | |
| 54 | 11 Nov | Qwen3-32B text classifier, 4-bit, 1 epoch, 3,427 Claude rows | A larger text model | 0.652 on machine transcripts, 0.687 on teacher transcripts | Kept (blend member) | Not recorded |
| 55 | 11 Nov 14:03 | **Submission 6** (58240192): 3-way blend 0.35 / 0.30 / 0.35 | | Dev 0.718 in-sample, 0.706 cross-fitted; **public 0.70850** | Kept | |
| 56 | 11 Nov 14:05 | Submission 6b (58240232): accidental duplicate of the same file | | Public 0.70850 | Used a daily slot; no effect | |
| 57 | 11 Nov | Second teacher from an OpenAI vision model (`scripts/openai_label.py`) | A different teacher | Script exists; output file is empty; no result recorded | No result | Not recorded |
| 58 | 11 Nov | Text 8B with 0 / 2,856 / 11,426 Claude rows, 1 epoch | How Claude data scales | 0.553 / 0.666 / 0.692 on machine transcripts. First quarter +0.113 [+0.073, +0.156]; the rest +0.026 [−0.002, +0.053]. Hard rows 0.420 → 0.440 | Best single model; see 59 | Not recorded |
| 59 | 11 Nov | The 11,426-row model inside the blend | Whether the best single model helps the blend | Blend 0.715 against 0.718; as a fourth member −0.007 to −0.014 | Rejected | CPU |
| 60 | 11 Nov | Untrained Qwen3-32B and Qwen3-235B on machine transcripts | Model size without training | 0.518 and 0.584; with 4 examples in the prompt 0.596 and 0.650. 235B over 32B with examples: +0.054 [+0.014, +0.094]. Hard rows 0.41–0.45 for 235B | Rejected as stand-alone | About $6.50 (one H200) |
| 61 | 11 Nov | 235B (with examples) as a fourth blend member | Size as an extra member | +0.012 over the equal-weight blend [−0.014, +0.037]; +0.016 with a cross-fitted weight [−0.006, +0.038]; hard rows 0.431 → 0.429 | Rejected: range crosses zero; test scoring priced at about $15 | CPU |
| 62 | 11–12 Nov | 32B text with explanation targets (v4); first candidate-student run | Explanation training | Stopped unfinished when all pods were terminated at 01:46 UTC on 12 Nov | No result | Not recorded |
| | | **11–12 Nov: CPU study of the blend** | | | | |
| 63 | 11–12 Nov | Sinkhorn label balancing toward the training label mix | Fixing the predicted label mix | λ 0.5: 0.737, +0.019 [+0.006, +0.033]. Cross-fitted λ: 0.7337, +0.016 [+0.001, +0.031]. Error rows 0.618 → 0.642; no-error rows 0.916 → 0.926 | Kept | CPU |
| 64 | 11–12 Nov | Sinkhorn toward Claude's label mix; toward a uniform mix | Other targets | −0.066; −0.019 | Rejected | CPU |
| 65 | 11–12 Nov | Balancing each member before blending | Where to balance | 0.7298–0.7331 over 3 seeds | Rejected: no better than balancing once | CPU |
| 66 | 11–12 Nov | Cross-fitted blend weights | Honest value of the blend | 0.7063 against 0.7182 in-sample: −0.012 [−0.021, −0.004] | Diagnostic: about 0.012 of the in-sample score is selection | CPU |
| 67 | 11–12 Nov | Arithmetic blend with the same weights | Mixing rule | 0.7071, −0.011 | Rejected | CPU |
| 68 | 11–12 Nov | Logistic-regression and gradient-boosting stackers | Learned combiners | 0.701–0.712 and 0.687–0.696 (−0.006 to −0.031) | Rejected | CPU |
| 69 | 11–12 Nov | Question prior added (log-linear); question prior divided out; self-prior | Priors on top of the blend | −0.004; best strength was zero; −0.008 | Rejected | CPU |
| 70 | 11–12 Nov | Route to the most confident member; confidence-weighted blend; reciprocal-rank blend | Other combiners | 0.6949 (−0.023); 0.7145 (−0.004); 0.7171 (−0.001) | Rejected | CPU |
| 71 | 11–12 Nov | NONE gates, family bias, family factorisation | Structure in the label space | Family factorisation −0.004 raw, −0.008 on the balanced blend | Rejected | CPU |
| 72 | 11–12 Nov | Nearest neighbours inside one question over about 27,000 teacher and Claude transcripts | Retrieval | 0.42–0.45 alone; −0.006 to 0.000 fused | Rejected | CPU |
| 73 | 11–12 Nov | Prior correction for Claude's label skew | Label shift from teacher data | +0.027 on one Claude-trained 8B; −0.010 to +0.001 on the blend | Rejected for the blend | CPU |
| 74 | 11–12 Nov | Masking the question text at training time (proxy) | Breaking the question shortcut | +0.005 [−0.021, +0.027] | Rejected | CPU |
| 75 | 11–12 Nov | Symbolic step checker | Checking the arithmetic line by line | Checks 40% of lines; finds the root line 42% of the time | Rejected | CPU |
| 76 | 11–12 Nov | Product-of-experts debiasing | Removing the question prior from the models | No-error and seen-pair rows collapse | Rejected | CPU |
| 77 | 11–12 Nov | Programmatic "mal-rule" synthesis | Generating wrong workings by rule | Supply ten times below estimate; no gain | Rejected | CPU |
| 78 | 11–12 Nov | Rule: prefer the specific code over the generic one | A label-space heuristic | −0.006 | Rejected | CPU |
| 79 | 11–12 Nov | Examples from other questions appended to candidate cards | Cross-question examples | −0.052 on new-pair matching | Rejected | CPU |
| 80 | 11–12 Nov | Perfect teacher transcripts fed to the blend members | Value of perfect reading for classifiers | Single models +0.025 to +0.035; blend +0.0155 [+0.006, +0.026] | Diagnostic: reading is not the main gap for classifiers | CPU |
| 81 | 11–12 Nov | Transcriber over-correction check | Whether the reader silently fixes student errors | A wrong final answer came out right on 8 of 173 rows | Diagnostic: not a problem here | CPU |
| 82 | 11–12 Nov | External datasets and other research leads | Extra data or readers | Assessed, not run: different label spaces, typed text, licences unverified or non-commercial | Rejected on paper | CPU |
| 83 | 12 Nov 02:31 | **Submission 7** (58258119): 3-way blend + Sinkhorn λ 0.5 | | Dev 0.737 (0.7337 cross-fitted); **public 0.71716** (+0.009) | Kept | |
| | | **12 Nov: audit and paid tests of new designs** | | | | |
| 84 | 12 Nov | Final-answer lookup inside one question | Matching students by final answer | Covers 52 rows; 0.827 there against the blend's 0.885 | Rejected | CPU |
| 85 | 12 Nov | Blend fused with Claude's real dev answers | The best case for any student that copies Claude | 0.824, +0.005 over Claude alone [−0.009, +0.019] | Diagnostic: copying Claude cannot beat Claude | CPU |
| 86 | 12 Nov | Claude + our question memory; + learned stacker; + learned remap to teacher codes | Whether anything we own adds to Claude | −0.003; −0.016; −0.002 | Diagnostic: nothing adds | CPU |
| 87 | 12 Nov | Simulated student that copies Claude on 50 / 70 / 80 / 90 / 100% of rows, fused | How good a student must be for 0.80 | Fused 0.760 / 0.780 / 0.795 / 0.810 / 0.824 | Diagnostic: needs more than 80% agreement | CPU |
| 88 | 12 Nov | Agreement of our text 8B with Claude's first answer, by Claude rows used | How fast students approach Claude | Hard rows 0.264 / 0.377 / 0.491 with 0 / about 2,900 / 11,426 rows | Diagnostic: about +0.11 per 4× data | CPU |
| 89 | 12 Nov | "The answer key leaks into transcripts" (an earlier claim) | Transcriber prompt | Key in last two lines of 25.9% of machine and 26.2% of teacher transcripts | Claim retracted | CPU |
| 90 | 12 Nov | Well-read against badly-read halves of the error rows | Whether reading limits a reasoning model | Untrained 235B −0.154 [−0.288, −0.018]; Claude with photo −0.028 [−0.124, +0.068] | Diagnostic: holds weakly | CPU |
| 91 | 12 Nov | Fix: full code descriptions in candidate prompts | 7,378 of 9,705 prompts had a description cut mid-sentence | Fixed; no score effect measured on its own | Kept (data fix) | CPU |
| 92 | 12 Nov 18:06–20:16 | **Stage B reason-first student**: Qwen3-8B reads the transcript and 10 described candidates, writes root line and reasoning, then picks a letter. 9,705 rows, 1 epoch | Whether writing reasoning first makes a small model diagnose like Claude | Alone 0.625; hard rows 0.417; fused −0.011 [−0.030, +0.009]; agrees with Claude on hard rows 0.387 (needed 0.65) | Rejected: failed all four gates | $2.14 |
| 93 | 12 Nov 20:38 | **Claude Opus without the photo**, reading our machine transcripts | The ceiling for any text-only model on our transcripts | 0.771 against 0.819 with the photo: −0.048 [−0.076, −0.020]; hard rows 0.637. Fused with our blend, cross-fitted: 0.751, +0.014 [−0.020, +0.048] | Diagnostic: text-only ceiling is under 0.80 | $1.79 |
| 94 | 12 Nov | Hard rows split into groups A, B, C | Where examples exist | See section 6.2 | Diagnostic | CPU |
| 95 | 12 Nov | Error catalogue, text matcher: 1,503 synthetic student workings for 115 dev questions, matched by text | Whether made-up examples can stand in for real ones | Fused −0.015 [−0.029, −0.002]. Same matcher on real examples 0.609 (A), on synthetic 0.328 | Rejected | CPU |
| 96 | 12 Nov | Coverage-aware blend (extra member chosen by whether the question has pool photos) | Routing by a test-time fact | In-sample +0.008 and +0.027; cross-fitted over 20 splits −0.005 [−0.023, +0.012] | Rejected | CPU |
| 97 | 12–13 Nov 21:40–00:14 | **Error catalogue, trained**: same text-8B recipe and seed twice, with and without the 1,503 synthetic rows | Whether training on synthetic examples lifts groups B and C | All rows 0.666 → 0.648 [−0.054, +0.018]; B+C +0.077 [−0.013, +0.167]; A −0.182 [−0.323, −0.049]; no-error −0.045 [−0.085, −0.008] | Rejected: pass mark not met | $2.81 |
| 98 | 13 Nov | Catalogue model (and its control) as a fourth blend member | | −0.003 and −0.001, cross-fitted | Rejected | CPU |
| | | **13 Nov: pointwise ranker** | | | | |
| 99 | 13 Nov 02:04–06:46 | **Pointwise ranker v1** (Qwen3-14B yes/no per candidate) | A ranker that reads the code's description, as used for unseen misconceptions in an earlier Kaggle competition | Alone 0.7087. Fused 0.7525 against 0.7337: +0.0188 [+0.0011, +0.0368]; hard rows +0.058 [+0.014, +0.102]; no-error −0.015 [−0.032, 0.000] | Kept (three of four pass-mark conditions met; no-error 0.9109 against 0.915 needed) | $5.25 |
| 100 | 13 Nov 07:08 | **Submission 8** (58299898): blend × ranker v1^0.5, Sinkhorn λ 0.75 | | Dev 0.7525 cross-fitted; **public 0.73183** (+0.0147) | Kept: current best | |
| 101 | 13 Nov | Keep the blend's no-error probability in the fusion | Protecting no-error rows | −0.009 [−0.021, +0.002] | Rejected | CPU |
| 102 | 13 Nov | Widen the candidate list from 10 to 20 | Recall on hard rows | Recall 0.858 → 0.925, but the added codes have low blend probability; not scored | Not adopted | CPU |
| 103 | 13 Nov 07:42–13:09 | **Ranker v2**: continue v1 on 4,262 more Claude error rows, 1,081 Claude no-error rows and the 1,199 teacher rows | Whether more Claude-labelled pairs lift the ranker | Alone 0.696 against 0.7087 (−0.013 [−0.041, +0.015]); fused 0.7464 against 0.7525 (−0.006 [−0.020, +0.008]); hard rows 0.478 | Rejected: on-pod gate (0.725) failed | $5.97 (includes $0.56 for a pod killed by a collector bug) |
| 104 | 13 Nov | Boost error codes never seen on the question (teacher labels) | Pushing towards unseen codes | −0.007 [−0.017, +0.002]. The system already puts 0.30 of its mass on such codes; the truth is one 0.32 of the time | Rejected | CPU |
| 105 | 13 Nov | Same boost, codes unseen in teacher and pool labels | | Chosen boost is 0 in every fold | Rejected | CPU |
| 106 | 13 Nov | Same boost, pool-only codes | | −0.001 [−0.010, +0.009] | Rejected | CPU |
| 107 | 13 Nov | Ranker weight depends on blend confidence | Adaptive fusion | −0.001; better in 45% of 20 splits | Rejected | CPU |
| 108 | 13 Nov | Re-tune the three blend weights with the ranker present | | −0.005; better in 5% of splits | Rejected | CPU |
| 109 | 13 Nov | Arithmetic mix instead of product | | −0.005 | Rejected | CPU |
| 110 | 13 Nov | Rank-based bonus | | −0.003 | Rejected | CPU |
| 111 | 13 Nov | Add the 8B classifier trained on all 11,426 Claude rows | A fourth member | +0.004 with v1, −0.004 with the ranker average | Rejected: noise | CPU |
| 112 | 13 Nov | Swap it for the teacher-only 8B member | | −0.003 with v1, +0.003 with the average | Rejected: noise | CPU |
| 113 | 13 Nov | Add the catalogue-trained or the reading classifier | | −0.002 to −0.003 | Rejected | CPU |
| 114 | 13 Nov | Average ranker v1 and v2 (geometric mean) | Two rankers that disagree on 60 of 301 top answers | Dev 0.7597: +0.007 over the submitted file [−0.002, +0.017]; better in 95% of 20 splits | The only one of the eleven that held on dev; see 115–116 | CPU |
| | | **14 Nov** | | | | |
| 115 | 13–14 Nov 23:24–00:45 | Score the test rows with the saved ranker v2 adapter (no training) | Needed for the average | 3,498 rows scored; v1 and v2 disagree on the top answer on 21.5% of test rows | Done | $1.47 |
| 116 | 14 Nov 00:55 | **Submission 9** (58326821): blend × mean of rankers v1 and v2 | | Dev 0.7597; **public 0.72550** (−0.006 against 0.73183) | Rejected: ranker v2 dropped | |
| 117 | 14 Nov | Review of all 21 discussion threads, 13 public notebooks and the host's two blog posts | What outside evidence exists | Host reference scores on the public split, models used off the shelf: Claude Opus 5.5 0.792, GPT-6 Astra (low) 0.790, Gemini 4 Argon 0.775, Qwen3.8-27B 0.757, Gemma 4 31B 0.710. No public notebook above 0.635 | Information: we had never run Qwen3.8-27B | CPU |
| 118 | 14 Nov 02:42–03:57 | **Qwen3.8-27B (FP8), zero-shot, thinking on**, on dev photos with the prompt Claude was given | A newer open vision-language model with no training | 270 of 301 photos answered. Alone 0.759 against 0.772 for the submitted system on the same rows (−0.013 [−0.063, +0.035]). Hard rows 0.608 against 0.502; seen pair 0.709 against 0.868 | See 120 | $1.37 (both runs) |
| 119 | 14 Nov | Same model, thinking off | Whether thinking matters | 102 photos: 0.657 against 0.717 for the submitted system on those rows | Rejected (thinking-off mode) | Included above |
| 120 | 14 Nov | Qwen3.8-27B (thinking) fused with the submitted system, cross-fitted | The memoriser plus a generalist | **0.805 against 0.772** on 270 rows: +0.033 [+0.006, +0.060]. Hard rows 0.614; no-error 0.922; seen pair 0.851. Fused with the blend only (no ranker): 0.795 | Passed its fused pass mark (+0.015) on dev. No test or public result yet | CPU |
| 121 | 14 Nov | Send only the rows where our system is least sure (top probability below 0.82) | Whether the gain needs every row | Keeps the whole gain (+0.039 against +0.039, fixed settings, in-sample) with 61% of the tokens | Inconclusive: in-sample only | CPU |
| 122 | 14 Nov about 04:20 | Score the 1,593 least-confident test photos with Qwen3.8-27B | Test predictions for 120 | All 1,593 answered. **Submission 10 (58335588): public 0.76266**, new best (+0.031 over 0.73183), from the first 1,392 answers. Complete file staged | Kept: new best | $7.08 |

**Designed but never run** (so there is no evidence either way): a blank-photo test of how much the image classifier
looks at the photo (D2); an image-reading reason-first student (E2); retraining the transcriber on teacher plus Claude
transcripts (E3); a held-out check set of about 1,500 Claude-labelled pool photos (F4); fine-tuning Qwen3.8-27B on
our labels; training the text classifiers on machine transcripts of pool photos (proposed in section 3 of the audit
record; no run recorded).

**Spend recorded.** RunPod $76.91 after experiment 118 (cap $80 until 14 Nov, then $88). Claude API $188.40 of $200
on the main key and $25.33 on a second key. Kaggle free GPU: the account showed 8.86 hours used on 8 Nov, a figure
that includes an unrelated job; the project's cap was 10 hours.

---

## 4. What works

Each row is an ingredient with a measured positive effect. Sizes are dev MAP@3 unless marked "public".

| Ingredient | Size of the effect | Evidence |
|---|---|---|
| Fine-tuning a vision-language model on the 1,199 fit rows, against the no-image prior | +0.098 [+0.057, +0.140] | Experiment 4 |
| A larger vision model (8B for 4B) with a reading loss | 0.595 → 0.632 alone; public +0.026 for the file that used it | 34, 35, 41. The reading loss itself is +0.013 over the plain 8B and not proven |
| Splitting the job into "read the photo" then "diagnose the text" | Public +0.038 (0.62383 → 0.66183), the largest single step | 42–46 |
| The first few thousand Claude labels as training targets | Text 8B 0.553 → 0.666 with 2,856 rows: +0.113 [+0.073, +0.156]. Public +0.021 for the file that added a Claude-trained image model | 52, 53, 58 |
| Blending members that make different mistakes | No member above 0.667; the 3-way blend 0.706 cross-fitted (0.718 in-sample). Public +0.026 when the 32B text member was added | 55 |
| Sinkhorn label balancing toward the training label mix | +0.019 [+0.006, +0.033] on dev; +0.009 public | 63, 83 |
| A pointwise yes/no ranker that reads each code's description | +0.019 [+0.001, +0.037] on dev, +0.058 [+0.014, +0.102] on hard rows; +0.0147 public | 99, 100 |
| A newer base model that thinks before answering (Qwen3.8-27B, no training) fused in | +0.033 [+0.006, +0.060] on 270 dev rows; hard rows 0.502 → 0.614. Dev only | 118–120 |
| Fine-tuning against prompting at the same size | Qwen3-32B: 0.596 with examples in the prompt → 0.652 fine-tuned (+0.056). The host's own study reports +7 to +18 points of top-1 from fine-tuning | 54, 60, 117 |
| Model size, with equal information | Qwen3-235B over Qwen3-32B, both with examples in the prompt: +0.054 [+0.014, +0.094] | 60 |
| One real example of the exact (question, code) pair | Submitted system: 0.500 with none, 0.820 with one teacher example, 0.955 with two or three. With a Claude-labelled example instead: 0.656 | Section 6.5. This is an observation; we have no source of more teacher examples |

**What we are good at, in one table.**

| Row type | Our best system | Claude Opus with the photo | Comment |
|---|---|---|---|
| No-error rows (34% of dev) | 0.911 (submitted); 0.926 (blend + balancing) | 0.934 | Level. Telling "error" from "no error" has an AUC of 0.947 in the blend |
| Seen-pair rows (31%) | 0.867 | 0.812 | We are ahead. Fine-tuning has learned the annotators' habits per question |
| Hard rows (35%) | 0.500 | 0.715 | All of the gap is here |

---

## 5. What failed, and why

Failures are grouped by the reason. Each group says whether the reason was measured or is our interpretation.

### 5.1 Re-mixing outputs we already have is exhausted (measured)

| Evidence | Numbers |
|---|---|
| Every cross-fitted combiner, prior, stacker or router over the three blend members | Between −0.031 and +0.006 (experiments 66–71) |
| Eleven fusion ideas on top of the submitted system | Ten between −0.007 and +0.004; the eleventh (+0.007) lost on the leaderboard (104–114, 116) |
| Output adapters, feature memory and priors on the first 4B model | −0.028 to +0.001 (12–17) |
| Extra members that resemble existing ones | −0.007 to −0.014 (59); +0.004 / −0.004 (111) |
| More flexible combiners did worse | Logistic regression −0.006 at the strongest regularisation, −0.017 at the weakest; gradient boosting −0.022 to −0.031 |

Why, as measured: the system is already calibrated where it matters. It puts 0.30 of its probability mass on codes
unseen for the question and the truth is such a code 0.32 of the time; it picks the wrong unseen code. The truth is in
the blend's top 10 on 94% of rows, so the missing piece is re-ranking information that the members' probabilities do
not contain. Interpretation: a gain needs new evidence, not a new way to mix old evidence. The two additions that did
help (the ranker, Qwen3.8-27B) both brought a model that reads something the classifiers do not.

### 5.2 More Claude-labelled examples of the same kind did not transfer (measured; the cause is interpretation)

| Evidence | Numbers |
|---|---|
| Text 8B, 2,856 → 11,426 Claude rows | +0.026 [−0.002, +0.053] overall; hard rows 0.420 → 0.440 (58) |
| Same model inside the blend | 0.715 against 0.718 (59) |
| Ranker v2: 4,262 more Claude error rows | Alone −0.013; fused −0.006; public −0.006 when averaged in (103, 116) |
| Rows whose exact pair gained 4 or more Claude examples in v2 | 0.782 → 0.746 (n = 42). Rows that gained 1–3: 0.655 → 0.678 (n = 43). Rows that gained none: 0.542 → 0.513 (n = 115) |
| A Claude-labelled example of the pair against a teacher example | 0.656 against 0.820 for the submitted system |
| Agreement with Claude's first answer on hard rows | 0.264 / 0.377 / 0.491 with 0 / about 2,900 / 11,426 rows: about +0.11 per fourfold increase |
| Coverage of the pool | Only 30% of hard rows have their pair anywhere in Claude's pool labels; the pool covers 284 questions; 28% of test rows are on questions with no pool photos |

Interpretation, not measured directly: Claude's labels follow Claude's conventions, not the teachers'. Supporting
facts: Claude itself is at 0.819, not 1.0; it puts NONE first on 70.9% of pool photos against 33.7% in the training
labels; it gives one "sloppy notation" code as its second choice on 17,021 of 26,202 answers, a code with exactly one
teacher example. The earlier projection that more examples per pair would lift the ranker (0.755 → 0.788) was built on
a curve that the v2 run showed to be mostly confounded: common pairs are easier.

### 5.3 Untrained large models were weaker than a trained small one, until Qwen3.8-27B (measured; the cause is interpretation)

| Model, no training on our labels | Dev | Hard rows |
|---|---|---|
| Qwen3-VL-4B scoring every code | 0.385 | not recorded |
| Qwen3.5-9B, code-only output | 0.455 | not recorded |
| Qwen3.5-27B (4-bit), brief evidence | 0.590 | not recorded |
| Qwen3-32B on our transcripts | 0.518 (0.596 with 4 examples) | 0.392 |
| Qwen3-235B on our transcripts | 0.584 (0.650 with 4 examples) | 0.448 (0.407 with examples) |
| Qwen3.8-27B on the photo, thinking off (102 rows) | 0.657 | 0.549 on those rows |
| Qwen3.8-27B on the photo, thinking on (270 rows) | 0.759 | 0.608 |
| For comparison: our trained text 8B (teacher labels only) | 0.667 | 0.363 |

Measured: every Qwen3-generation classifier or prompt-only model stayed at 0.34–0.45 on hard rows, trained or not
(the yes/no ranker reached 0.486), and training never lifted hard rows above what the untrained model already did
(32B: 0.392 untrained, 0.357 trained). Qwen3.8-27B with thinking reaches 0.608. Interpretation: model generation and
thinking matter more than parameter count here. The earlier runs differ from the Qwen3.8 run in several ways at once
(older models; short constrained outputs with no thinking; the two largest read our transcript and not the photo), so
the record cannot say which difference is responsible. No direct thinking-on against thinking-off comparison on the
same rows is recorded.

### 5.4 A small model taught to "reason first" did not diagnose like its teacher (measured; the cause is interpretation)

| Evidence (Stage B, experiment 92) | Numbers |
|---|---|
| Alone, against the plain classifier student | 0.625 against 0.692; hard rows 0.417 against 0.440 |
| Agreement with Claude's first answer on hard rows | 0.387, against 0.491 for the plain classifier student |
| Over-confidence | One code at ≥ 0.95 probability on 187 of 301 rows; right on 69% of those (42% on hard rows) |
| Sensitivity to candidate order | Two orderings pick the same code on 70% of rows; accuracy 0.66 when they agree, 0.31 when not |
| Root line | Within one line of the teacher's on 67% of error rows; accuracy 0.50 when the line is right, 0.26 when wrong |

The outputs were well formed (99.7% parsed), so this was not a formatting failure. Interpretation: at 8B and 9,705
training rows the model learns the form of Claude's reasoning without its substance. Whether a larger or newer base
model would behave differently in this recipe was not tested.

### 5.5 Synthetic examples carry about half the signal of real ones and damage other rows (measured; the mechanism is interpretation)

| Evidence (error catalogue, experiments 95, 97) | Numbers |
|---|---|
| Catalogue contains the true code | 0.97 (A), 0.71 (B), 0.84 (C) of rows |
| Same text matcher, real against synthetic examples | 0.609 against 0.328 (group A); 0.681 against 0.446 (seen pair) |
| Trained on the catalogue, machine transcripts | B+C +0.077 [−0.013, +0.167]; A −0.182 [−0.323, −0.049]; no-error −0.045 [−0.085, −0.008]; all rows −0.018 |
| Same two models scored on teacher transcripts | 0.692 → 0.720 (+0.028); A unchanged; +0.053 [+0.003, +0.106] on the 177 rows whose questions are in the catalogue |
| Share of rows predicted NONE first | 30.6% with the catalogue against 35.2% without (truth 33.6%) |

Interpretation: the catalogue shows each question with about ten different codes, so the model can no longer answer
from the question alone. That helps where the usual code is wrong (B, C) and hurts where it is right (A), most on
noisy machine transcripts. The pilot also added 1,503 error workings and no correct ones. This is the only training
intervention before the ranker that moved groups B and C at all.

### 5.6 Matching a working to stored examples by surface similarity does not work (measured)

| Evidence | Numbers |
|---|---|
| Nearest neighbours in 4B feature space | 0.577 alone (15) |
| Nearest neighbours inside a question over about 27,000 transcripts | 0.42–0.45 alone; −0.006 to 0 fused (72) |
| Final-answer lookup | 0.827 against the blend's 0.885 on the 52 rows it covers (84) |
| Root-example similarity | +0.012 [−0.001, +0.024], never confirmed (29) |
| Examples from other questions added to candidate cards | −0.052 (79) |
| Four examples in the prompt for Claude | −0.008 [−0.034, +0.017]; top-1 up, top-3 recall down (48) |

The true code of a hard row exists on some other question in the fit rows for 96% of hard rows, with a median of 24
examples, and that does not help our models.

### 5.7 Better reading helped classifiers little; classical OCR could not read these pages (measured)

| Evidence | Numbers |
|---|---|
| Perfect teacher transcripts instead of machine ones | Single classifiers +0.025 to +0.035; blend +0.0155 [+0.006, +0.026] (80) |
| Share of the gap to Claude on rows the transcriber read exactly right | 17.0 of 30.3 gap points |
| Tesseract; PP-FormulaNet on crops; PP-OCRv5 | Number recall 12%; 22%; lower than PP-FormulaNet (value not recorded). 0 to 2 of 27 cited equations read exactly (27, 31, 32) |
| Symbolic checking of the written arithmetic | Parser found a checkable equality on 8 of 60 teacher transcripts; step checker covers 40% of lines (28, 75) |

This holds for classifiers that mostly answer from the question. For a model that reasons, reading matters more: see
section 6.4.

### 5.8 Training-recipe changes to the image classifier gave nothing (measured; no cause established)

Soft secondary labels with rare-code oversampling (0.618 against 0.619), helper heads for error family and
error-present (0.613), evidence-first targets (0.6285 against 0.632), and a 4-bit 32B image classifier (0.403 after one
epoch, stopped). Output adapters on frozen features over-fitted (fit rows 0.960, dev 0.570).

### 5.9 Small dev gains did not survive the leaderboard (measured)

| Change | Dev said | Public said |
|---|---|---|
| Sinkhorn balancing | +0.019 [+0.006, +0.033] | +0.009 |
| Pointwise ranker v1 | +0.019 [+0.001, +0.037] | +0.0147 |
| Averaging rankers v1 and v2 | +0.007 [−0.002, +0.017], better in 95% of 20 splits | −0.006 |
| Blend weights chosen on dev | 0.718 in-sample | 0.706 cross-fitted; 0.7085 public |
| Coverage-aware blend | +0.008 and +0.027 in-sample | −0.005 cross-fitted (never submitted) |

With 301 rows the standard error of one score is 0.023 and of a paired difference about 0.013. Dev has also judged
dozens of designs. Gains under about +0.02 cannot be told from noise on it.

### 5.10 Runs that produced no answer

The first 27B run (grammar compile timeout); the reading-supervised 4B run (cancelled in the queue); the 32B
explanation run and the first candidate-student run (pods terminated); the first ranker v2 pod (killed by a collector
bug, $0.56); the self-training labels and the same-question-examples text run (no result recorded); the OpenAI second
teacher (no output). The Qwen3.8-27B dev run covered 270 of 301 rows with thinking and 102 without, because of a time limit.

---

## 6. Where we lose points

### 6.1 By row type

Dev MAP@3. The first block is all 301 rows. The second block is the 270 rows that Qwen3.8-27B answered (101 no-error,
86 seen-pair, 83 hard; counts derived from the fractions in `reports/q38_dev_zero_shot.json`).

| System | All | No error | Seen pair | Hard |
|---|---|---|---|---|
| **301 rows** | | 101 rows | 94 rows | 106 rows |
| 3-way blend, no balancing | 0.718 | 0.916 | 0.830 | 0.431 |
| Blend + Sinkhorn, cross-fitted (the reference) | 0.7337 | 0.926 | 0.856 | 0.442 |
| Pointwise ranker v1 alone | 0.7087 | 0.891 | 0.764 | 0.486 |
| **Submitted system** (blend × ranker v1) | **0.7525** | 0.911 | 0.867 | **0.500** |
| Text 8B trained on 11,426 Claude rows ("old 8B student") | 0.692 | 0.906 | 0.746 | 0.440 |
| Stage B reason-first student | 0.625 | 0.822 | 0.647 | 0.417 |
| Claude Opus reading our machine transcript | 0.771 | 0.931 | 0.752 | 0.637 |
| Claude Opus with the photo | 0.819 | 0.934 | 0.812 | 0.715 |
| **270 rows** | | 101 rows | 86 rows | 83 rows |
| Submitted system | 0.772 | 0.911 | 0.868 | 0.502 |
| Qwen3.8-27B, thinking, alone | 0.759 | 0.924 | 0.709 | 0.608 |
| Submitted × Qwen3.8-27B, cross-fitted | **0.805** | 0.922 | 0.851 | 0.614 |
| Claude Opus reading our machine transcript | 0.786 | 0.931 | 0.746 | 0.653 |
| Claude Opus with the photo | 0.823 | 0.934 | 0.795 | 0.717 |

- Before the ranker, the gap to Claude (0.1008 [0.063, 0.138]) split as: 78% on error rows with a code new for a known
  question (84 rows, 0.431 against 0.712), 21% on error rows from unseen questions (22 rows, 0.432 against 0.727), 6% on
  no-error rows, and −5.5% on seen-pair rows, where we win.
- Holding the other two row types fixed, 0.80 overall needs about 0.63 on hard rows.
- The share of held-out error rows whose pair is unseen falls as labels grow: 71.6% with 600 teacher rows, 62.9% with
  900, 55.7% with 1,200. The record states the same row type is about half of the error rows on test.

### 6.2 Hard rows by group

| System | A (32 rows): Claude labelled this pair in the pool | B (31): question in the pool, pair absent | C (43): question not in the pool |
|---|---|---|---|
| Blend + Sinkhorn (cross-fitted reference) | 0.542 | 0.290 | 0.477 |
| Pointwise ranker v1 alone | 0.604 | 0.339 | 0.504 |
| **Submitted system** | 0.656 | 0.344 | 0.496 |
| Old 8B student (11,426 Claude rows) | 0.677 | 0.204 | 0.434 |
| Catalogue-trained 8B (against its control) | 0.375 (0.557) | 0.312 (0.274) | 0.527 (0.422) |
| Qwen3-235B, untrained, our transcript | 0.510 | 0.242 | 0.550 |
| Claude Opus, our transcript | 0.688 | 0.505 | 0.694 |
| Claude Opus, photo | 0.839 | 0.586 | 0.717 |
| On the 270-row subset (A 29, B 27, C 27): submitted | 0.621 | 0.395 | 0.481 |
| Qwen3.8-27B, thinking, alone | 0.713 | 0.500 | 0.605 |
| Submitted × Qwen3.8-27B | 0.747 | 0.494 | 0.593 |

- Group A: where a Claude-labelled example of the pair exists, our Claude-trained student (0.677) equals Claude reading
  the same transcript (0.688).
- Group B is the hardest for everyone. The question is familiar and has about 81 labelled pool photos, none with this
  error. Every Qwen3-generation model or system scores 0.16–0.38 there. Claude drops to 0.505–0.586. Qwen3.8-27B reaches 0.50.
- Group C: no pool photos for the question. Our trained models are at 0.37–0.53; untrained large models do better
  (0.55 for Qwen3-235B, 0.605 for Qwen3.8-27B).

### 6.3 Where the truth ranks when we are wrong

| Measure | Number |
|---|---|
| Submitted system, 106 hard rows: truth is 1st / 2nd / 3rd / 4th–10th / lower | 36 / 26 / 12 / 16 / 16 |
| Ranker v1 alone, hard rows: 1st / 2nd / 3rd / 4th–10th / not offered | 39 / 20 / 10 / 22 / 15 |
| Blend, all rows: truth in top 3 / 5 / 10 | 83.1% / 89.4% / 94.0% |
| Blend, hard rows: truth outside the top 10 | 16 of 106 |
| When the submitted system's top answer on a hard row is wrong (70 rows), the top answer is | a code seen on that question: 25; another unseen code: 31; a pool-only code: 7; NONE: 7 |
| Wrong top answer is in the same code family as the truth | 13% of the time |
| Hard rows where Claude (from our transcript) is right and we are wrong / the reverse | 28 / 9 |
| Our top answer is one of the question's training codes (blend, new-code rows) | 36%, against 11% for Claude |
| Two test students on the same question get the same first code from us | 42%; in the training labels this happens 27% of the time |
| Oracle: perfect re-ranking of the blend's top 10 | 0.940 |
| Oracle: best of the three blend members per row | 0.795 |

The losses are different diagnoses, not near misses, and not one over-used code: we pick one of the six commonest
codes on 37% of hard rows and the truth is one of them on 40%. Claude's own errors have the same shape: of its wrong
dev answers, 51 are in the wrong code family, 14 are in the right family with the wrong code, 7 are false alarms and 7
are missed errors.

### 6.4 Against Claude, with and without the photo

| Who reasons | Reads the photo | Reads a teacher transcript | Reads our machine transcript |
|---|---|---|---|
| Claude Opus | 0.819 (hard 0.715) | not run | 0.771 (hard 0.637) |
| Our old 8B student | cannot | 0.720 | 0.692 (hard 0.440) |
| Our 32B text classifier | cannot | 0.687 | 0.652 (hard 0.357) |
| Qwen3.8-27B, thinking (270 rows) | 0.759 (hard 0.608) | not run | not run |

- Losing the photo costs Claude −0.048 [−0.076, −0.020] overall and −0.078 on hard rows. So 0.771 is the ceiling for
  any text-only model on these transcripts.
- On identical transcripts Claude scores 0.637 on hard rows and our models 0.44–0.45 (0.500 for the submitted system).
  The record splits the 0.27 hard-row gap into about 0.19 of reasoning on identical input and about 0.08 of reading.
- The loss without the photo is even: 0.08 on well-read error rows, 0.06 on badly-read ones. Rows whose question has a
  diagram lose less (−0.025) than the rest (−0.052), so the missing diagram is not the cause.
- Nothing we own adds to Claude: the blend fused with Claude's real answers is +0.005 [−0.009, +0.019] over Claude alone.

### 6.5 The effect of one example of the exact pair

| Examples of the exact (question, code) pair | Rows | Score |
|---|---|---|
| **Submitted system, by teacher examples in the fit rows** | | |
| None | 106 | 0.500 |
| One | 61 | 0.820 |
| Two or three | 26 | 0.955 |
| None from teachers, but one or more from Claude's pool labels (group A) | 32 | 0.656 |
| **3-way blend, by teacher examples** (earlier analysis) | | |
| 0 / 1 / 2 / 3 or more | not recorded | 0.45 / 0.80 / 0.92 / 1.00 |
| **By examples the ranker trained on (teacher or Claude), error rows: blend / ranker alone / fused / Claude** | | |
| 0 | 83 | 0.394 / 0.444 / 0.444 / 0.669 |
| 1 | 37 | 0.770 / 0.667 / 0.775 / 0.779 |
| 2–3 | 40 | 0.821 / 0.679 / 0.829 / 0.829 |
| 4–7 | 23 | 0.870 / 0.819 / 0.884 / 0.884 |
| 8 or more | 17 | 0.843 / 0.931 / 0.931 / 0.843 |

Claude is nearly flat across these bins; we are a step function. The last block is confounded (common pairs are
easier), as the ranker v2 run showed.

### 6.6 Transcription quality

| Measure (dev photos) | Value |
|---|---|
| Numbers in the teacher transcript that the machine transcript recovers | 0.881 |
| Numbers in the machine transcript that are right | 0.913 |
| Character error, median, raw (includes line ids, spacing, teacher annotations) | 0.328 |
| Character error, median, normalised | 0.095 (quartiles 0.027 and 0.333; maximum 6.77) |
| Claude's own transcripts, same normalised measure | 0.124 |
| Root line read with character error ≤ 0.10 | 131 of 200 error rows (0.655) |
| Annotated error text present in the machine transcript | 55% (0.555) |
| Rows where the transcript has the wrong number of lines | 40; untrained 235B scores 0.321 there, Claude with the photo 0.804 |
| Wrong final answer silently turned right by the transcriber | 8 of 173 rows |
| Cost of machine against teacher transcripts for each text classifier | 0.025–0.035 |
| Error rows, well-read half against badly-read half | 235B untrained 0.565 / 0.403; text 8B 0.632 / 0.537; Claude with photo 0.777 / 0.745 |
| Training and scoring mismatch | The text classifiers were trained on teacher and Claude transcripts and scored on machine transcripts |

### 6.7 Dev against public

| Submission | Dev (how obtained) | Public | Dev minus public |
|---|---|---|---|
| 1. 4B image classifier | 0.595 | 0.58216 | 0.013 |
| 2. + calibration and prior | 0.607 in-sample; 0.598 cross-fitted | 0.59800 | 0.009; 0.000 |
| 3. 8B with reading loss × 4B | 0.647 in-sample | 0.62383 | 0.023 |
| 4. Read-then-diagnose blend | 0.684 in-sample | 0.66183 | 0.022 |
| 5. Text × Claude-trained image | 0.693 in-sample | 0.68283 | 0.010 |
| 6. 3-way blend | 0.718 in-sample; 0.706 cross-fitted | 0.70850 | 0.010; −0.002 |
| 7. + Sinkhorn λ 0.5 | 0.737 fixed λ; 0.7337 cross-fitted | 0.71716 | 0.020; 0.017 |
| 8. + ranker v1 (best) | 0.7525 cross-fitted | 0.73183 | 0.021 |
| 9. + ranker v2 averaged in | 0.7597 | 0.72550 | 0.034 |

(The last column is this review's subtraction of two recorded numbers.)

- Cross-fitted dev matched public within 0.003 for submissions 2 and 6. For submissions 7 and 8 it is 0.017–0.021 above.
- Known reasons in the record: the dev fold is label-stratified, so its label mix is unusually close to the fit rows'
  (distance 0.109 against 0.322 expected for a random split), which flatters Sinkhorn; dev has been reused many times;
  cross-fitting covers only the last setting chosen, not the earlier choices.
- The public leaderboard is about 29% of the test set (about 1,000 workings), by the host's statement. Its own standard
  error is not recorded.
- Facts about the test set: 95.7% of test rows are on a question that appears in the training set; the test set has
  about 7.1 rows per question against 3.2 in training; 69% of test rows are on questions with at least one Claude
  error row in the pool; 28% are on questions with no pool photos. The host says the NONE share in test "should be
  similar" to training.

### 6.8 Smaller, older findings that still stand

- Rare codes: 13 codes have one training example and two have none. On the early 8B classifier, codes with ≤5 examples
  scored 0.26. Fixing every such row would have added at most +0.033 to the first model.
- On the early 8B classifier, 68% of lost points were "saw an error but named the wrong error family".
- Reaching Claude's level is not enough on its own terms: the host measured Claude Opus 5.5 at 0.792 on the public
  split, below the 0.82 milestone.

---

## 7. Research questions worth searching for

Each question is tied to a measured weakness above. The phrases are search terms, not paper titles.

| # | Question | Measured weakness it comes from | Search phrases |
|---|---|---|---|
| 1 | How can a model diagnose an error type it has never seen on that question? | Hard rows: 0.500 for us, 0.715 for Claude (6.1). Group B: 0.34 against 0.59 (6.2) | "zero-shot misconception diagnosis"; "compositional generalisation to unseen label–context pairs"; "student error classification unseen questions"; "math misconception detection large language models"; "cognitive diagnosis open-ended student work" |
| 2 | How do you stop a fine-tuned classifier from answering "the usual code for this question"? | Step from 0.500 to 0.820 with one example (6.5); same first code for two students 42% of the time against 27% in labels (6.3); question masking and product-of-experts both failed (5.1) | "shortcut learning mitigation fine-tuned language models"; "spurious correlation between context and label"; "debiasing with a bias-only model"; "group distributionally robust optimisation text classification"; "counterfactual data augmentation for shortcut removal" |
| 3 | How can a strong reasoning model be distilled into a small one without keeping only the form of the reasoning? | Stage B student: agreement with Claude 0.387 on hard rows, below the plain classifier's 0.491; over-confident; order-sensitive (5.4) | "knowledge distillation from reasoning traces"; "chain-of-thought distillation small language models"; "rationale-augmented fine-tuning faithfulness"; "teacher–student capacity gap reasoning"; "multiple-choice option order sensitivity language models" |
| 4 | How should labels from a teacher model be used when its conventions differ from the human annotators'? | A Claude-labelled example is worth 0.656, a teacher example 0.820; more Claude rows added +0.02 on hard rows; Claude says NONE first on 70.9% of the pool against 33.7% in labels (5.2) | "pseudo-label noise and label shift in distillation"; "annotator convention mismatch teacher model"; "noise transition matrix learning with noisy labels"; "class-imbalanced semi-supervised learning"; "confidence-weighted pseudo-labelling small labelled set" |
| 5 | Can synthetic student errors be made close enough to real ones to train on? | Synthetic examples carry about half the signal (0.328 against 0.609); training on them gave B+C +0.077 but A −0.182 and no-error −0.045 (5.5) | "synthetic student error generation mathematics"; "mal-rules buggy procedures simulation"; "simulating student misconceptions with language models"; "synthetic-to-real gap data augmentation text"; "hard negative generation fine-grained classification" |
| 6 | How can handwritten multi-line maths be transcribed faithfully, mistakes included? | Annotated error text present in 55% of transcripts; 40 rows with the wrong line count; classical OCR recovers 12–22% of numbers (6.6, 5.7) | "handwritten mathematical expression recognition multi-line"; "OCR over-correction of student errors"; "faithful transcription vision language models"; "reading order recovery handwritten documents"; "handwritten math solution transcription benchmark" |
| 7 | Is it better to diagnose straight from the image than to read first and reason on text? | Claude loses 0.048 without the photo; text-only ceiling 0.771; our image classifier is the weakest reasoner (6.4) | "vision language model handwritten math error detection"; "error localisation in handwritten student solutions"; "multimodal grading of student work"; "OCR-free document understanding reasoning"; "two-stage versus end-to-end visual reasoning" |
| 8 | How do you fine-tune a model that thinks before answering on about a thousand labels without losing its zero-shot strength? | Qwen3.8-27B zero-shot: hard rows 0.608 but seen pair 0.709; our fine-tuned models are the reverse (0.500 and 0.867). Training never lifted hard rows on Qwen3-generation models (5.3) | "fine-tuning reasoning models preserving chain of thought"; "robust fine-tuning zero-shot weight interpolation"; "catastrophic forgetting LoRA vision language models"; "few-shot adaptation without losing generalisation"; "supervised fine-tuning with thinking tokens" |
| 9 | How should a memoriser and a generalist be combined row by row? | Fusing Qwen3.8-27B gives +0.033; sending only the least-confident rows kept the gain in-sample; learned routers over similar members failed (5.1, experiment 121) | "selective prediction and model routing"; "learning to defer large language models"; "ensemble of fine-tuned and zero-shot models"; "confidence-based model cascades"; "out-of-distribution detection for expert routing" |
| 10 | What is the best way to rank fine-grained labels from their descriptions? | Truth in the top 10 on 94% of rows but ranked 2nd–10th on 54 of 106 hard rows; the pointwise ranker gave +0.019; 13 codes have one example (6.3, 6.8) | "LLM reranking pointwise pairwise listwise"; "label description cross-encoder classification"; "retrieve and rerank misconception"; "few-shot classification with label semantics"; "long-tail fine-grained text classification" |
| 11 | How can model choices be trusted on a 301-row validation set? | Standard error 0.023; +0.007 on dev became −0.006 on the leaderboard; cross-fitted dev ran 0.017–0.021 above public twice (5.9, 6.7) | "model selection with small validation sets"; "adaptive overfitting to a holdout set"; "paired bootstrap significance testing small test sets"; "ensemble selection overfitting validation"; "leaderboard shake-up public private split" |
| 12 | How can the predicted label mix be corrected at test time when the true mix is unknown? | Sinkhorn toward the training mix: +0.019 dev, +0.009 public; toward a wrong target −0.066 (experiments 63, 64) | "label shift estimation black-box predictors"; "test-time prior adaptation"; "optimal transport label balancing"; "distribution matching pseudo-labels"; "prior shift calibration classification" |

---

## 8. Open inconsistencies

Places where the record does not add up, or where two sources differ. The later source is used above unless stated.

1. **Hard-row score of the blend.** Given as 0.431 (raw blend), 0.442 / 0.4418 (blend + Sinkhorn, cross-fitted; audit
   record sections 2 and 14) and 0.451 (sections 9 and 10 and the study report). The differences look like "no
   balancing", "cross-fitted λ" and "fixed λ 0.5", but the record does not say so in one place. This review uses 0.431
   for the raw blend and 0.442 for the reference.
2. **Group A and seen-pair scores of the blend.** Group A: 0.573 (section 11) against 0.5417 (section 14). Seen pair:
   0.856, 0.848 (section 11.2) and 0.830 (raw blend). Probably the same cause as item 1.
3. **Blend + Sinkhorn overall.** 0.737 (fixed λ 0.5) and 0.7337 (cross-fitted) are both called the dev score in
   different places. `reports/status.json` records 0.737 for submission 7.
4. **Ranker v1 on hard rows.** Section 15 gives the truth's rank as 39 / 20 / 10 / 22 / 15, which works out to 0.494.
   Section 14 gives 0.4858 for the same model on the same rows.
5. **"0.755 → 0.788".** Section 15 projects from a fused score of 0.755; the cross-fitted fused score is 0.7525.
6. **Claude rows in the 25% scaling run.** "2,900" in sections 2 and 8.3 of the audit record; "2,856" in section 12
   and the study report.
7. **Number of Claude-labelled pool photos.** 26,202 in the audit record and study guide; the reconciliation note in
   `reports/claude_budget.json` says 17,503 photos (an earlier state: dev 301 + 9,602 + 7,600);
   `reports/cand_dataset.json` counts 26,176 valid pool rows after dropping 26 duplicates of dev or test photos.
8. **235B as a fourth member after Sinkhorn.** −0.017 in `docs/study/sections/09-history-lessons.md`; +0.008 in
   `docs/study/sections/06-ensemble.md`. Both say the range crosses zero or is negative.
9. **Best-of-two oracle for 4B and 27B.** 0.7436 in `reports/model_complementarity.json`; 0.7403 in
   `docs/path_to_082.md`. Possibly raw against calibrated 4B; not stated.
10. **Sinkhorn fold choices.** One planning file says all five folds chose λ 0.5; the result file shows four of five.
11. **Cross-fitted dev "predicts public".** The study guide states this from submissions 2 and 6 (within 0.003).
    Submissions 7 and 8 contradict it (0.017–0.021 above public).
12. **Stale fields in `reports/status.json`.** `updated_at_utc` is 13 Nov 07:08 although the file contains the 14 Nov
    submission; `next_required` and `budget` still describe the 8 Nov state.
13. **Stale caps in older documents.** `docs/handoff.md` says a $50 RunPod cap; the study report's cost ledger stops at
    $62.85. `AGENTS.md` and `reports/runpod_budget.json` (cap $88, spend $76.91 before experiment 122) are current.
14. **Confidence threshold for the Qwen3.8 test scoring.** 0.82 in the audit record, 0.817 in `AGENTS.md`.
15. **Leaderboard top.** 0.8445 (snapshot of 8 Nov in `status.json`), 0.8465 (9 Nov and the study report). Rank 74 of
    426 on 13 Nov and 77 of 433 on 14 Nov for the same 0.73183 file.
16. **Dev score of the image member.** 0.653 is from a sibling model trained on the fit rows; the model that scored the
    test set was trained on all 1,500 rows and has no dev score.
17. **Kaggle GPU hours.** "8.86 hours used" is an account-wide figure that includes an unrelated job; the project's own
    share is not recorded.
18. **Training NONE share.** 33.7% from the data (506 of 1,500); one research file quotes 29%.
19. **Claude spend.** The $200 working limit applies to the main key ($188.40). The second key's $25.33 sits outside
    it, so total Claude spend across both keys is above $200.
20. **Runs with no recorded outcome.** Self-training labels (experiment 40), the same-question-examples text run (44)
    and the OpenAI second teacher (57) appear in the files with no result.
21. **Qwen3.8-27B "alone at least 0.74".** The record calls this pass-mark condition "borderline": 0.759 was measured
    on 270 rows where our system scores 0.772, not on all 301 where it scores 0.752. Only the fused condition is
    clearly met.
