# K-12: what is wrong inside our models, and the research plan toward 0.80

12 Nov 2026. CPU only, nothing launched, nothing spent, nothing submitted. Scores are MAP@3 on the 301 development rows
(fold 0) unless stated. Earlier work this builds on: `runs/deep/DEEP_PLAN.md`, `runs/deep/stageB/STEPS_0_2.md`.

## 1. Bottom line

- Best public score is **0.71716**. The honest dev reference is **0.7337** (blend + Sinkhorn, cross-fitted).
- We lose almost everything on one kind of row: a student error whose code we have **never seen on that question**.
  Every model we have trained scores 0.30–0.44 there. Claude scores 0.715.
- To reach 0.80 we need about **0.63** on those rows. No amount of extra data, bigger backbone or blending has moved
  that number past 0.44 with our current recipe.
- This pass found six concrete defects inside the components (section 3). Two earlier conclusions were drawn from
  confounded comparisons (section 4).
- My honest estimate: 0.80 by 17 Nov with the money left ($22.10 pod, $13.39 Claude) is unlikely, about 5–10%.
  0.74–0.76 is about 25–35%. The plan below is the route with the best chance, and it stops spending early if a gate fails.

## 2. What 0.80 needs, in numbers

| Row type | Share of dev | Ours (blend + Sinkhorn) | Claude | Needed for 0.80 |
|---|---|---|---|---|
| No error (NONE) | 34% | 0.926 | 0.934 | keep |
| Error, code already seen on this question | 31% | 0.856 | 0.812 | keep |
| Error, code new for this question, or new question | 35% | **0.442** | **0.715** | **about 0.63** |

- Our blend on the first two row types plus Claude on the third would score **0.818**. So the target is reachable in
  principle with a "memoriser + generalist" design. We have the memoriser. We do not have the generalist.
- The same row type is about half of the error rows on test, so gains there should carry to the leaderboard.

The third row type, by model (raw scores, no Sinkhorn):

| Model | Claude-labelled rows used | All rows | New-code / new-question rows |
|---|---|---|---|
| Image model, teacher labels only | 0 | 0.632 | 0.343 |
| Image model (deployed member) | 3,427 | 0.653 | 0.362 |
| Text 8B, teacher labels only, 3 epochs (deployed) | 0 | 0.667 | 0.363 |
| Text 8B, 1 epoch | 2,900 | 0.666 | 0.420 |
| Text 8B, 1 epoch | 11,426 | 0.692 | 0.440 |
| Text 32B (deployed) | 3,427 | 0.652 | 0.357 |
| Qwen3-235B, no training | 0 | 0.584 | 0.448 |
| Qwen3-235B, 4 examples in the prompt | 0 | 0.650 | 0.407 |
| Deployed 3-way blend | | 0.718 | 0.431 |
| Claude Opus | | 0.819 | 0.715 |

Four times more Claude data moved the hard rows by +0.02. A 235B model with no training is no better than our 8B there.
The recipe is the limit, not data volume or size.

## 3. Component by component: what is wrong

"New" marks findings from this pass. The rest confirm the earlier analysis.

| Component | What it does | What is wrong | Evidence | Fix |
|---|---|---|---|---|
| **Transcriber** (Qwen3-VL-8B) | Photo → text lines | The annotated error span is present in only 55% of machine transcripts, and line order is sometimes scrambled. **Retracted after cross-check (section 8):** I first blamed the answer key in its prompt. The data does not support that. The repetition penalty is untested. | In one dev row it moved three lines to the end, which hides an "unfinished answer" error. | No change until experiment D1 shows reading limits a reasoning model. |
| **Text classifiers** (8B, 32B; 65% of the blend) | Question + transcript → one vector → 76-way head | The decision is read before any reasoning, and codes are bare IDs. The models learn "this question usually gets this code". | Step curve from the earlier analysis: 0.45 / 0.80 / 0.92 / 1.00 with 0 / 1 / 2 / 3+ examples of the exact question-code pair. | Reason first, then choose among described candidates (Stage B). |
| | | **New:** trained on teacher and Claude transcripts, scored on machine transcripts. The model never sees machine-style mistakes in training. | `run_text32_full.sh` passes no `--fit-transcripts`, so the default `gold` applies. Each text model loses 0.025–0.035 going from teacher to machine transcripts. | Train on machine transcripts of the Claude-labelled pool photos. The transcriber never saw those photos, so their noise matches test. |
| | | **New:** no weights were saved. Only the test probabilities exist. | `src/k12/text_diagnose.py` has no save call. | Add saving. Without it we cannot probe, re-score or pseudo-label with these members, and a retrain is needed to reproduce the submission. |
| **Image classifier** (Qwen3-VL-8B) | Photo + question → one vector → 76-way head | Same "decide before reasoning" head. **New:** it was trained on 3,427 of the 26,202 Claude-labelled photos. It is the only member that sees the handwriting. | `runs/pod/full-distill/epoch-2/metadata.json`: `fit_rows` 4,927. | Use it as the base for an image-reading student (experiment E2). |
| **Blend + Sinkhorn** | Multiply three outputs, rebalance label mix | Nothing broken. It cannot create a right answer that no member ranks. | Truth is outside our top 10 on 16 of the 106 hard rows. Best pick among the three members would give 0.795. | No more work here. |
| **Stage B candidate student** (built, not trained) | Transcript + 10 described candidates → writes reasoning → picks a letter | **New:** 23% of code descriptions are cut at 140 characters in every prompt, mid-sentence. The student is supposed to learn code meaning from this text. | `scripts/build_candidate_dataset.py:149`. Prompts use 1,133 of 2,048 tokens at p99, so full text fits. | Use full descriptions, rebuild, rerun the gates. Free. |
| | | **New:** same transcript mismatch as the text classifiers (Claude and teacher transcripts in training, machine transcripts at test). | `runs/deep/stageB/data/cand_dev.jsonl` uses machine transcripts. | Same fix as above, if experiment D1 says reading matters. |
| **Evaluation** | 301 dev rows, one fold | **New:** the standard error of one score is 0.023. Several ideas were accepted or dropped on differences of 0.005–0.02. | Row bootstrap of the blend. | Judge by paired bootstrap and by the hard-row slice. Add a held-out set of Claude-labelled pool photos for agreement checks. |

Also tested and dropped in this pass: matching a student to others with the same final answer on the same question. It
covers 52 of 301 rows and scores 0.827 there against the blend's 0.885.

## 4. Two earlier conclusions that need correcting

1. **"Model size is not the bottleneck."** The 32B was 4-bit, trained for 1 epoch with 3,427 Claude rows. The 8B it was
   compared with had 3 epochs, or 11,426 Claude rows. With equal information the larger model was better (235B over 32B,
   +0.054 to +0.066). The right statement is: size alone does not fix the hard rows (235B scores 0.41–0.45 there), and
   whether size helps inside a reason-first recipe is untested.
2. **"Better reading is worth only +0.006 to +0.012."** That was measured through classifiers that mostly ignore the
   transcript. A model that reasons depends on it far more. On error rows, split into well-read and badly-read halves:

   | Model | Well-read half | Badly-read half |
   |---|---|---|
   | Qwen3-235B, no training (reads our transcript) | 0.565 | 0.403 |
   | Text 8B classifier (reads our transcript) | 0.632 | 0.537 |
   | Claude (reads the photo itself) | 0.777 | 0.745 |

   Where the transcript has the wrong number of lines (40 rows), 235B falls to 0.321 and Claude is at 0.804. Part of the
   drop is that long workings are harder, but Claude barely moves. So a text-only student may be capped by our reader.
   Experiment D1 measures this directly.

## 5. Research plan

Everything paid needs your explicit approval first. Order is by information per dollar.

### Phase 0: free fixes (CPU, about 2 hours)
- F1. **Done.** Full code descriptions in the Stage B data. Before, 7,378 of 9,705 training prompts had a description
  cut into the next phrase; now 0. Rows, targets and weights are unchanged, p99 length is 1,214 tokens, all gates pass.
- F2. **Done, not yet run on a GPU.** `text_diagnose.py` now saves the adapter and head.
- F3. Dropped. The answer-key claim did not survive the cross-check (section 8).
- F4. Set aside about 1,500 Claude-labelled pool photos, never used in training, as a second check set. Not done.

### Phase 1: three diagnostics that decide the design

| ID | Question | How | Cost | Reading the result |
|---|---|---|---|---|
| **D1** | Is the reader or the reasoner the limit? | Claude on the 301 dev rows **without the photo**: once with our machine transcript, once with the teacher transcript. Dev rows are training photos, so this is inside your API rule. | about $5 API | Hard-row score ≥ 0.65 on machine transcripts: reading is fine, a text student has full headroom, run E1 only. ≥ 0.65 on teacher but ≤ 0.58 on machine: fix the reader (E3) or use an image student (E2). ≤ 0.58 even on teacher transcripts: only an image student (E2) can work. |
| **D2** | How much does our image model look at the photo? | Score dev with the saved image checkpoint three ways: real photo, blank photo, photo of a different student on the same question. | about $0.5 pod, or free and slow on this Mac | If top-1 stays the same on most error rows without the real photo, the model is answering from the question. That confirms the head must change, not the data. |
| **D3** | Can an 8B learn to diagnose like Claude? | Stage B smoke, then the mid-epoch check already built into `pod_run_v5b.sh`. | $0.3, then part of E1 | Abort if hard-row score < 0.35 at 50% of the epoch. |

### Phase 2: build the generalist (one of these, chosen by D1)

| ID | Experiment | Cost | Gate to keep it |
|---|---|---|---|
| **E1** | Text candidate student (prepared): Qwen3-8B writes root line, reasoning, then a letter; fused with the current blend. | about $5 | Fused gain ≥ +0.015 with paired CI above 0; hard rows ≥ 0.50; NONE ≥ 0.915; seen-code rows ≥ 0.83. |
| **E2** | Image-reading student: Qwen3-VL-8B sees the photo, the question and the 10 described candidates, writes transcript line, reasoning, letter. Trained on Claude's full answers for up to 26k photos plus teacher rows. New code, built from `candidate_student.py` and `vision.py`. | $8–12 estimate; a $0.5 smoke fixes the number | Same gates as E1. Run only if D1 shows the transcript caps a text student. |
| **E3** | Reader upgrade: retrain the transcriber on teacher + Claude transcripts, machine-transcribe a slice of the pool, retrain the student on matched transcripts. | $6–10 estimate | Error span present in ≥ 0.68 of dev transcripts (now 0.555), then E1's gates. |

E2 and E3 do not both fit in the remaining $22. D1 picks one.

### Phase 3: fuse and finalise (CPU)
- Fuse the student with the current blend by cross-fitted geometric weight. The blend stays; weight 0 is allowed.
- Stage CSVs under `artifacts/` with sha256. I ask before every single submission.
- Finals: one safe (current 0.71716 file), one fused if it passed its gates.

### Timeline to 17 Nov 00:00 UTC
- 12 Nov: Phase 0. With approval: D1 and D2.
- 13 Nov: E1 (4–5 hours on a pod), fuse on CPU.
- 14 Nov: E2 or E3, chosen by D1.
- 15 Nov: fusion, test sanity checks, at most two check submissions. No new training after 12:00.
- 16 Nov: pick the two finals by 18:00.

## 6. Budget

| | Left | Plan uses |
|---|---|---|
| RunPod (cap $80, spent $57.90) | $22.10 | D2 $0.5 + E1 $5 + E2 or E3 $8–12 = $14–18 |
| Claude API (limit $200, spent $186.61) | $13.39 | D1 about $5 |
| Kaggle GPU | off (your 8 Nov instruction) | none |

## 7. Files behind the numbers
- Slice scores, 235B fusion, reading split, final-answer lookup: computed this pass from the bundles under `runs/pod/`,
  `runs/pod/probe235/`, `runs/claude/dev_opus.jsonl` and `runs/arch/reading-impact/per_row_reading.csv`.
- Reference and Stage B state: `runs/deep/stageB/reference.json`, `runs/deep/stageB/gate_v5b.json`.
- Budgets: `reports/runpod_budget.json`, `reports/claude_budget.json`.

## 8. Cross-check, later on 12 Nov: can this plan beat Claude?

Short answer: **no. On the evidence we have, the plan can approach Claude but not pass it.** All numbers are dev MAP@3;
intervals are paired bootstraps over questions.

**8.1 Copying Claude cannot beat Claude.** I gave our system Claude's real dev answers and fused them with our blend
(weight cross-fitted). That is the best case for any student trained to copy Claude.

| System | All rows | Against Claude |
|---|---|---|
| Claude alone | 0.819 | |
| Our blend + Sinkhorn | 0.737 | |
| Blend fused with Claude's real answers | 0.824 | +0.005 [−0.009, +0.019] |
| Claude + question memory from teacher labels | 0.816 | −0.003 [−0.015, +0.007] |
| Claude + blend, learned stacker | 0.803 | −0.016 [−0.034, +0.001] |
| Claude + learned Claude→teacher remap | 0.817 | −0.002 [−0.008, +0.004] |

Nothing we own adds a measurable gain on top of Claude. Claude's 79 wrong first answers fall on 69 different
(Claude code, teacher code) pairs, so there is no repeated mistake to learn a correction for.

**8.2 How good the student must be.** A simulated student that gives Claude's answer on a share of rows and our best
existing student's answer on the rest, fused with the blend:

| Share of rows where student = Claude | Student alone | Fused system |
|---|---|---|
| 0% (what we have) | 0.692 | 0.737 |
| 50% | 0.754 | 0.760 |
| 70% | 0.780 | 0.780 |
| 80% | 0.793 | 0.795 |
| 90% | 0.807 | 0.810 |
| 100% | 0.819 | 0.824 |

For 0.80 the student must itself score about 0.80, which means matching Claude on more than 80% of rows. Above a
student score of about 0.76 the blend adds almost nothing.

**8.3 Where our students are today.** Agreement with Claude's first answer:

| Text 8B student trained with | All rows | Hard rows |
|---|---|---|
| 0 Claude rows | 0.502 | 0.264 |
| 2,900 Claude rows | 0.645 | 0.377 |
| 11,426 Claude rows | 0.688 | 0.491 |

About +0.11 on hard rows per 4× data. All 26,202 rows would give roughly 0.56 under the current recipe. So data volume
alone will not reach 80%; only a recipe change (reason first, described candidates) could, and that is untested.

**8.4 Claims from sections 3–4 that changed.**
- Answer key in the transcriber prompt: **retracted.** The key appears in the last two lines of 25.9% of machine
  transcripts and 26.2% of teacher transcripts. On rows with a wrong final answer: 8.9% against 8.4%. Only 6 of 336
  numbers the transcriber invented come from the answer key.
- Reading matters more for a reasoning model: **holds, weakly.** Controlling for length and row type, the badly-read
  half costs the untrained 235B −0.154 [−0.288, −0.018] and Claude −0.028 [−0.124, +0.068]. The same split costs the
  8B classifier −0.059 even on teacher transcripts, so part of the effect is row difficulty.
- Within-question nearest-neighbour lookup over Claude's pool labels (earlier probe) scores 0.42–0.45 alone, and my
  final-answer lookup lost to the blend. Retrieval is not a route.

**8.5 What this means.**
- The realistic ceiling of "distil Claude into an open student" is just under Claude, and getting to 0.80 needs a
  near-perfect copy. I have no evidence that an 8B can do that.
- I found no ingredient in our data that beats Claude. Stating a plan to beat Claude would be a guess.
- The cheapest experiment that could change this picture is D1 (about $5 API): Claude on the dev rows without the
  photo, reading our machine transcripts. Its score is a hard upper limit for every text-only student. If it is under
  0.80, the prepared $5 text student cannot reach 0.80 and should not be run for that purpose.

## 9. Result of the text-student run (E1), 12 Nov 20:15 UTC

**The student failed all four dev gates. It is not used. Test predictions were stopped and the pod was terminated.**
Pod `flm24as8yzphmb`, RTX 6000 Ada, 18:06–20:16 UTC, **$2.14** (quoted about $5). RunPod total is now $60.05 of $80.

| Dev rows (301) | New student | Old 8B student | Blend + Sinkhorn | Claude |
|---|---|---|---|---|
| All | 0.625 | 0.692 | 0.737 | 0.819 |
| No error | 0.822 | 0.906 | 0.926 | 0.934 |
| Code already seen on the question | 0.647 | 0.746 | 0.856 | 0.812 |
| Hard rows | 0.417 | 0.440 | 0.451 | 0.715 |

| Gate | Needed | Got |
|---|---|---|
| Fused gain over the cross-fitted reference | ≥ +0.015, CI above 0 | −0.011 [−0.030, +0.009] |
| Fused score on hard rows | ≥ 0.50 | 0.462 |
| Student agrees with Claude's first answer on hard rows | ≥ 0.65 | 0.387 |
| Same, on rows where blend and Claude disagree | ≥ 0.45 | 0.255 |

What the outputs show:
- Answers were well formed (99.7% parsed) and read like real diagnoses, so this is not a formatting failure.
- The student is over-confident: it gives one code ≥ 0.95 probability on 187 of 301 rows and is right on 69% of those
  (42% on hard rows). Its second and third choices carry little information, which hurts MAP@3 and fusion.
- The two candidate orders pick the same code on only 70% of rows. Accuracy is 0.66 when they agree and 0.31 when
  they do not.
- It names a root line within one line of the teacher's on 67% of error rows. Accuracy is 0.50 when the line is right
  and 0.26 when it is wrong.
- Top-1 accuracy on hard rows is 0.321 against the blend's 0.292, so there is a small first-choice signal, but not
  enough to pass any gate.

Conclusion: "reason first, then choose among described candidates" on Qwen3-8B with 9,705 rows does not make the
student diagnose like Claude. It agrees with Claude less often than the plain classifier student does (0.387 against
0.491 on hard rows). The saved adapter is in `runs/deep/stageB/pod_out/v5b_full/adapter`; the full report is
`runs/deep/stageB/fuse_arm1.json`.

## 10. Claude without the photo (D1, cheap version), 12 Nov 20:38 UTC

Claude Opus on the 301 dev rows with **no photo and no question diagram**, reading only our machine transcripts
(`scripts/claude_label.py --transcripts`, run `dev_opus_text_tx`). Cost **$1.79** (3 pilot rows + one batch of 298).
Claude API total is now $188.40 of $200. All 301 answers parsed.

| Dev rows | Claude, no photo | Claude, with photo | Our blend | Our old 8B student |
|---|---|---|---|---|
| All | **0.771** | 0.819 | 0.737 | 0.692 |
| No error | 0.931 | 0.934 | 0.926 | 0.906 |
| Code already seen on the question | 0.752 | 0.812 | 0.856 | 0.746 |
| Hard rows | **0.637** | 0.715 | 0.451 | 0.440 |

- Losing the photo costs Claude −0.048 overall [−0.076, −0.020] and −0.078 on hard rows. So **0.771 is the ceiling for
  any text-only model reading these transcripts**, and it is under 0.80.
- The loss is spread evenly: error rows in the well-read half lose 0.08, in the badly-read half 0.06. Rows whose question
  has a diagram (47) lose less (−0.025) than the rest (−0.052), so the missing diagram is not the cause.
- On the **same transcripts our models read**, Claude scores 0.637 on hard rows against our 0.44–0.45. Of our 0.27 gap
  to Claude on hard rows, about 0.19 is reasoning on identical input and about 0.08 is reading.
- Fusing no-photo Claude with our blend gives 0.751 cross-fitted (+0.014 [−0.020, +0.048], unstable weights), so the
  blend does not lift a text-only model past 0.80 either.

Conclusion: a model must read the photo itself **and** reason close to Claude's level to reach 0.80. Our 8B could not
do the reasoning part even from text (section 9), so an 8B image student has no evidence behind it.

## 11. New design work, 12 Nov 21:00–22:30 UTC: where examples exist we already match Claude; can we make examples?

All CPU and helper agents; no pod or API spend.

**11.1 The hard rows split by example coverage (106 dev rows).** "Example" = a labelled working of the same question
with the same root code, from teacher labels or Claude's 26,202 pool labels.

| Group | Rows | Old 8B student | Blend | 235B untrained | Claude, no photo | Claude, photo |
|---|---|---|---|---|---|---|
| A. Claude labelled this exact pair on another photo | 32 | **0.677** | 0.573 | 0.510 | 0.688 | 0.839 |
| B. Question has pool photos (median 81 labelled), pair absent | 31 | 0.204 | 0.290 | 0.242 | 0.505 | 0.586 |
| C. Question has no pool photos | 43 | 0.434 | 0.477 | 0.550 | 0.694 | 0.717 |

- In group A the old student equals Claude reading the same transcript. One example of the exact pair is what our
  models need; teacher-labelled or Claude-labelled both work.
- Group B is hard for every open model (0.16–0.29), trained or not. It is not only memorisation.
- Only 30% of hard rows have their pair in Claude's pool labels, and 28% of test rows sit on questions with no pool photos.

**11.2 Error catalogue pilot (synthetic examples).** Four helper agents wrote, blind, 1,503 synthetic student workings
for 115 dev questions (10–12 error codes per question; inputs were question, answer key, taxonomy and correct workings
from fit/pool rows only). Files: `runs/deep/catalog/`, training file `data/v6/synthetic_catalog_dev_pilot.jsonl`.
Measures were fixed before scoring.

| Measure | A | B | C | Seen pair |
|---|---|---|---|---|
| True code has a catalogue example | 0.97 | 0.71 | 0.84 | 1.00 |
| Text matcher alone, real examples | 0.609 | none exist | none exist | 0.681 |
| Text matcher alone, synthetic examples | 0.328 | 0.269 | 0.244 | 0.446 |
| Blend → blend fused with synthetic matcher | 0.573 → 0.557 | 0.290 → 0.269 | 0.477 → 0.415 | 0.848 → 0.844 |

- **Pre-fixed result: negative.** Fusing the synthetic matcher into the blend gives −0.015 on dev [−0.029, −0.002].
- The catalogue does contain the right code for 71–84% of rows that had no example at all.
- With the same matcher, synthetic examples carry about half the signal of real ones (0.33 vs 0.61, 0.45 vs 0.68).
- Exploratory, after the fact: final-answer matching fires on 39 rows and is right 20–40% of the time on hard rows;
  number overlap ranks the true code first 23–32% of the time against 10% by chance; a "plausible codes" prior does nothing.
- Not tested: training a model on the catalogue, which is how real examples produced the 0.34 → 0.68 jump.

**11.3 Coverage-aware blend.** Add-on member and weight chosen by whether the question has pool photos (known at test
time). In-sample it looks like +0.008 on pool questions (old student) and +0.027 on the rest (235B untrained).
Cross-fitted over 20 fold splits it is −0.005 [−0.023, +0.012]. **Negative.**

**11.4 Prepared, not launched.** `text_diagnose.py --synthetic FILE` adds catalogue rows to the fitting set
(compile-checked only). The decisive test is a paired run of the old-student recipe with and without the pilot
catalogue, same seed, scored on groups B and C.

## 12. Paired training test of the error catalogue, 12–13 Nov

Pod `9fn8wkgrpay5yx`, RTX 6000 Ada at $1.09/hr, 21:40–00:14 UTC, **$2.81** (quoted about $3.50). RunPod total $62.85 of $80.
Same recipe and seed twice (Qwen3-8B text classifier, 1 epoch, 1,199 teacher rows + 2,856 Claude rows); arm 2 adds the
1,503 synthetic catalogue rows at weight 0.5. Arm 1 reproduced the earlier 25%-scale run exactly (0.6661).
Script `runs/deep/catalog/pod_run_catalog.sh`; outputs `runs/deep/catalog/pod_out/`.

**Pre-fixed pass mark: FAIL.** Needed: B+C up ≥ 0.08 with the paired range above 0; seen-pair and no-error rows not
down by more than 0.02.

| Dev rows, machine transcripts | Arm 1 | Arm 2 (catalogue) | Change | Paired range |
|---|---|---|---|---|
| All 301 | 0.666 | 0.648 | −0.018 | [−0.054, +0.018] |
| B+C, the target (74) | 0.360 | 0.437 | +0.077 | [−0.013, +0.167] |
| B (31) | 0.274 | 0.312 | +0.038 | |
| C (43) | 0.422 | 0.527 | +0.105 | |
| A (32) | 0.557 | 0.375 | −0.182 | [−0.323, −0.049] |
| Seen pair (94) | 0.681 | 0.674 | −0.007 | |
| No error (101) | 0.911 | 0.866 | −0.045 | [−0.085, −0.008] |

On teacher transcripts the same two models give 0.692 → 0.720 (+0.028), B+C +0.095, A unchanged, and +0.053
[+0.003, +0.106] on the 177 rows whose questions are in the catalogue.

Exploratory, after the fact:
- The catalogue arm predicts NONE first on 30.6% of rows against 35.2% (truth 33.6%): it added 1,503 error rows and no
  correct workings. Full label balancing removes most of the NONE loss (−0.010) but the arm is still −0.006 overall.
- As a fourth blend member, cross-fitted: arm 1 −0.001, arm 2 −0.003. No gain.
- Reading: the catalogue shows each question with about ten different codes, so the model can no longer answer from
  the question alone. That helps where the usual code is wrong (B, C) and hurts where it is right (A), most of all
  when the transcript is noisy.

## 13. Pointwise yes/no ranker, prepared 13 Nov (not launched)

**Why.** The first-place Eedi 2024 solution handled misconceptions unseen in training with a pointwise ranker: the model
reads the question, the wrong answer and one candidate misconception as text and answers yes or no, trained with many
wrong candidates per right one. We have tried a 76-way head (memoriser) and a listwise student that writes its own
reasoning; we have not tried this. Source: Kaggle write-up "1st Place Detailed Solution", Eedi - Mining Misconceptions in
Mathematics.

**Design.** `src/k12/pointwise_ranker.py`. One prompt per (working, candidate): context, "Candidate diagnosis: CODE: name -
description [example]", "Is this candidate the root major error ...? Answer Yes or No." Score = logit(Yes) − logit(No) from
the frozen LM head; LoRA rank 32. A row's candidates form a group (right ones plus wrong ones from the same row, 6 in all);
loss is soft cross-entropy over the group. At test time the blend's top 10 candidates are scored and the result is fused
with the blend by `scripts/fuse_candidate.py`.

**Data.** `data/v7/pw_{train,dev,test}.jsonl`, built from the Stage B candidate files (fit rows and pool rows only; no dev
row in training). Train 8,505 rows (2,398 teacher views, 6,107 Claude); the test run uses all teacher rows + 2,800 Claude
rows = 5,198 rows, 31,188 pairs, 9.3M tokens. Dev 3,010 pairs (0.9M tokens), test 34,980 pairs (10.7M tokens).

**Checked on CPU.** Trains, scores and writes a bundle with a tiny model; the fusion script reads the bundle.

**Pass mark, fixed before any run.** Against the cross-fitted reference 0.7337: fused gain ≥ +0.015 with the paired range
above zero; fused hard rows ≥ 0.50; fused no-error rows ≥ 0.915 and seen-pair rows ≥ 0.83. On the pod, test rows are scored
only if the ranker alone reaches 0.45 on the hard dev rows.

**Cost estimate.** From the measured Qwen3-8B training speed on an RTX 6000 Ada (about 1.4–1.5k tokens/s): 8B about 2 hours
($2.2) for the dev test, $2.9 with test predictions; Qwen3-14B scaled by size, not measured: about 3.5 hours ($3.8), $5.1
with test predictions. Script: `runs/deep/pointwise/pod_run_pointwise.sh`.

## 14. Result of the pointwise ranker test (Qwen3-14B), 13 Nov 05:42 UTC

Pod `592bq6m2h2n8ny` (RTX 6000 Ada). Training 02:04–05:36 UTC, 1,300 updates, 729 tokens/s, loss 1.49 → 0.85, no errors.
Dev scoring 3,010 pairs in 306 s. Fusion: `scripts/fuse_candidate.py`, report `runs/deep/pointwise/fuse_pw14.json`
(blend × ranker^w, then Sinkhorn λ; w and λ cross-fitted over 5 question folds; chosen w 0.5, λ 0.75 in 4 of 5 folds).

| Dev rows | n | Reference (cross-fit) | Ranker alone | Fused (cross-fit) | Claude |
|---|---|---|---|---|---|
| All | 301 | 0.7337 | 0.7087 | **0.7525** | 0.8189 |
| No error | 101 | 0.9257 | 0.8911 | 0.9109 | 0.9340 |
| Seen pair | 94 | 0.8564 | 0.7642 | 0.8670 | 0.8121 |
| Hard rows | 106 | 0.4418 | 0.4858 | 0.5000 | 0.7154 |
| Group A | 32 | 0.5417 | 0.6042 | 0.6562 | 0.8385 |
| Group B | 31 | 0.2903 | 0.3387 | 0.3441 | 0.5860 |
| Group C | 43 | 0.4767 | 0.5039 | 0.4961 | 0.7171 |

Paired question bootstrap, fused minus reference: all +0.0188 [+0.0011, +0.0368]; hard rows +0.058 [+0.014, +0.102];
no-error −0.015 [−0.032, 0.000]; seen pair +0.011 [−0.010, +0.034]; group A +0.115 [+0.030, +0.216].

**Pass mark.** Gain ≥ +0.015 with range above zero: met (+0.0188, lower bound +0.001, so only just). Hard rows ≥ 0.50: met
exactly on the line (53.0 of 106; the script's own check printed false because of floating-point rounding). Seen pair ≥ 0.83:
met (0.867). No-error ≥ 0.915: **missed** (0.9109, 0.4 of a row short). Three of four; by the rule as written the pass mark is
not met. The on-pod gate (ranker alone ≥ 0.45 on hard rows) passed at 0.486, so test rows were scored as planned.

First design of the day with a positive fused gain whose range excludes zero. The gain is on hard rows, largest where Claude
had labelled the exact pair (group A). It is one of several experiments judged on the same 301 rows, and the lower bound is
close to zero, so treat +0.019 as an estimate with a real chance of being smaller on the leaderboard.

**Test predictions and close-out, 06:46 UTC.** Test scoring: 3,498 rows, 34,980 pairs in 3,608 s. Pod terminated by the
collector at 06:46 UTC after the final copy; cost $5.25 (RunPod total $68.10 of $80). Adapter (157 MB), dev and test bundles are
in `runs/deep/pointwise/pod_out/pw_full/`. Fused test file staged with w 0.5, λ 0.75 in `artifacts/submission-3way-pw14/`
(3,498 rows, ids in test order, three distinct valid codes per row; top-1 differs from the 0.71716 public file on 11.4% of rows;
NONE top-1 share 0.344). Not submitted; the user decides.

**Public score, 13 Nov 07:08 UTC.** Submitted with the user's permission ("Yes, submit it"), ref 58299898: public MAP@3
**0.73183**, up +0.0147 from 0.71716. Dev had said +0.019; the public gain is inside the dev range and close to its centre.
Public rank 74 of 426 at 07:08 UTC (was 82 with 0.71716); 29 teams are at 0.80 or above.

## 15. Ranker v2 design, 13 Nov 07:45 UTC (prepared, not launched; needs the user's approval)

**What the first ranker's dev predictions show (CPU, no spend).**

| Exact (question, code) examples the ranker trained on | Error rows | Blend | Ranker alone | Fused | Claude |
|---|---|---|---|---|---|
| 0 | 83 | 0.394 | 0.444 | 0.444 | 0.669 |
| 1 | 37 | 0.770 | 0.667 | 0.775 | 0.779 |
| 2–3 | 40 | 0.821 | 0.679 | 0.829 | 0.829 |
| 4–7 | 23 | 0.870 | 0.819 | 0.884 | 0.884 |
| 8+ | 17 | 0.843 | 0.931 | 0.931 | 0.843 |

- The first run trained on 1,881 of the 7,618 error rows Claude labelled in the pool (a 200-per-code cap, then a 2,800-row sample).
  Distinct pairs covered: 680 used, 1,327 in the pool; pairs with ≥ 4 examples: 120 used, 510 in the pool. 69% of test rows are on
  questions with at least one Claude error row in the pool.
- With a 600-per-code cap, 4,265 unused error rows can be added; 57 of 200 dev error rows (23 of 106 hard rows) move up a bucket.
  If each moved row took the mean of its new bucket the fused score would go 0.755 → 0.788; at half that effect 0.772. The bucket
  means are confounded (common pairs are easier), so read this as an upper bound.
- Earlier classifier scaling study (`reports/claude_scaling.json`): 25% → 100% of Claude rows gave +0.026 [−0.002, +0.053].
- Hard rows, truth rank under the ranker: 1st 39, 2nd 20, 3rd 10, 4th–10th 22, not offered 15 of 106. Widening the list to 20
  raises recall 0.858 → 0.925 but those codes have low blend probability; not part of v2.
- Keeping the blend's no-error probability in the fusion was tested on CPU: −0.009 [−0.021, +0.002] against the current fusion. Rejected.

**Design.** Continue the saved 14B adapter (`runs/deep/pointwise/pod_out/pw_full/adapter`) on rows it has not seen: 4,265 new
Claude error rows (600-per-code cap), about 1,100 unseen Claude no-error rows, plus one more pass over the 1,199 teacher rows so
the teacher's labelling convention is not forgotten. About 6,560 rows, 39k pairs, 11.7M tokens; lr 5e-5, same prompt, same group
size, same scoring. Fuse as before.

**Cost (measured speed 729 tokens/s, RTX 6000 Ada $1.09/hr).** Training 4.5 h = $4.87; setup, dev and test scoring 1.2 h = $1.33.
About $5.10 for the dev result, $6.20 with test predictions; working limit $7.50 (RunPod would stand at up to $75.60 of $80).

**Pass mark, fixed before the run.** Against the current fused system (cross-fitted 0.7525): fused v2 ≥ 0.7625; hard rows ≥ 0.52;
no-error rows ≥ 0.905; seen-pair rows ≥ 0.85. On the pod, test rows are scored only if the ranker alone reaches 0.725 on dev
(v1: 0.709). Dev has now judged several designs, so the public leaderboard decides whether a passing v2 replaces the 0.73183 file.

**Launched 13 Nov 07:42 UTC** with the user's approval ("okay run it on the RTX 6000 ada"): pod `303vccef0iaoz5`, RTX 6000 Ada, $1.09/hr.
Training set built: 6,542 rows (4,262 new Claude error rows, 1,081 new Claude no-error rows, 1,199 teacher rows), about 39k pairs, 11.9M tokens;
`data/v8/pw_train_v2.jsonl`, script `runs/deep/pointwise/pod_run_pointwise_v2.sh`. CPU check: continue-from-adapter trains, saves and reloads identically.
On-pod checks before training: the uploaded adapter must reproduce the first run's scores on 20 dev rows; a short continuation must run end to end.

**Restart, 13 Nov 08:12 UTC.** The first pod reached training (adapter check passed: mean score difference 0.027, same top answer on 20 of 20 rows; 713–729 tokens/s)
but was terminated at 08:10 UTC by the local collector, whose SSH call lacked host-key options and so never reached the new pod; 10 failed polls trigger a shutdown.
Cost $0.56, nothing copied back. Collector fixed and checked against the new pod. Same job restarted on pod `mjkm0rj3wn5en8`; job start 08:29 UTC after a slow upload.
The $7.50 limit covers both pods (this pod: collector limit $6.75, safety timer $6.90).

**Result, 13 Nov 13:09 UTC: failed.** Training ran 08:37–13:01 UTC (1,636 updates, 720 tokens/s, no errors). Pod terminated 13:09 UTC;
adapter and dev predictions are in `runs/deep/pointwise/pod_out_v2/pw2_full` (adapter sha256 0c4b210b…). Cost $5.41 for this pod, $5.97 for the job
including the lost pod; RunPod stands at $74.07 of $80. No test rows were scored because the on-pod gate failed.

| Dev MAP@3 (301 rows) | All | No error (101) | Seen pair (94) | Hard (106) | A | B | C |
|---|---|---|---|---|---|---|---|
| Ranker v1 alone | 0.7087 | 0.891 | 0.764 | 0.486 | 0.604 | 0.339 | 0.504 |
| Ranker v2 alone | 0.6960 | 0.891 | 0.738 | 0.473 | 0.635 | 0.317 | 0.465 |
| v1 and v2 averaged, alone | 0.7176 | 0.891 | 0.780 | 0.497 | 0.630 | 0.382 | 0.481 |
| Blend, cross-fitted λ (reference) | 0.7337 | 0.926 | 0.856 | 0.442 | 0.542 | 0.290 | 0.477 |
| Fused with v1 (submitted, public 0.73183) | 0.7525 | 0.911 | 0.867 | 0.500 | 0.656 | 0.344 | 0.496 |
| Fused with v2 | 0.7464 | 0.921 | 0.862 | 0.478 | 0.630 | 0.339 | 0.465 |
| Fused with the v1/v2 average | 0.7597 | 0.921 | 0.872 | 0.506 | 0.656 | 0.360 | 0.500 |

Paired differences (bootstrap over questions): v2 alone − v1 alone −0.013 [−0.041, +0.015]; fused v2 − fused v1 −0.006 [−0.020, +0.008];
fused average − fused v1 +0.007 [−0.002, +0.017]; fused average − blend reference +0.026 [+0.009, +0.044].

Pass mark: on-pod gate (alone ≥ 0.725) failed at 0.696; fused ≥ 0.7625 failed (0.7464); hard ≥ 0.52 failed (0.478); no-error ≥ 0.905 met (0.921);
seen-pair ≥ 0.85 met (0.862). The v1/v2 average also misses the pass mark (0.7597, hard 0.506).

What the run showed (`scratchpad/design/a5.py`):
- The example-count curve in the evidence table was mostly confounding, as warned. Error rows whose exact pair gained 4 or more examples went
  0.782 → 0.746 (n=42); rows that gained 1–3 went 0.655 → 0.678 (n=43); rows that gained none went 0.542 → 0.513 (n=115).
  The 53 rows that moved up a bucket went 0.664 → 0.695; the projection assumed about +0.12 on those rows.
- v2 is better at first place and worse at second and third: top answer right on 192 rows against 186 for v1 (23 gained, 17 lost), but MAP@3 is lower.
  The two rankers disagree on the top answer on 60 of 301 rows, which is why their average beats both.
- More Claude-labelled rows of the same kind do not lift the ranker. Hard rows stay near 0.50 fused against Claude's 0.67.

## 16. Next-step design, 13 Nov 16:00 UTC (CPU only; nothing launched, nothing spent)

Left to spend: RunPod $5.93 of $80, Claude API $11.60 of $200. Deadline 17 Nov 00:00 UTC. All scores are dev MAP@3; "20 splits" means the
fusion settings were cross-fitted over 20 different 5-fold question splits and compared with the submitted system (blend × ranker v1).
Scratch scripts: `scratchpad/design/a6.py` … `a12.py`.

**16.1 Where the hard rows stand (106 rows, fused with v1, 0.500).** Truth is 1st on 36, 2nd on 26, 3rd on 12, 4th–10th on 16, lower on 16.
When we are wrong the top answer is a code seen on that question 25 times, another unseen code 31 times, a pool-only code 7 times, no-error 7 times;
it is in the truth's family only 13% of the time. Claude reading the same transcripts is right on 28 hard rows where we are wrong; we are right on 9
where it is wrong. The losses are different diagnoses, not near misses, and not one over-used code (ours picks one of the six commonest codes on 37% of
hard rows; the truth is one of them on 40%).

**16.2 Ideas tested on CPU.**

| Idea | Result against the submitted system | Verdict |
|---|---|---|
| Boost error codes never seen on the question (teacher labels) | −0.007 [−0.017, +0.002]. The fused system already puts 0.30 of its mass on such codes; the truth is one 0.32 of the time. The models are calibrated here; they pick the wrong unseen code. | Rejected |
| Same boost, codes unseen in teacher and pool labels | chosen boost is 0 in every fold | Rejected |
| Same boost, pool-only codes | −0.001 [−0.010, +0.009] | Rejected |
| Ranker weight depends on blend confidence | −0.001, better in 45% of 20 splits | Rejected |
| Re-tune the three blend weights with the ranker present | −0.005, better in 5% of splits | Rejected |
| Arithmetic mix instead of product | −0.005 | Rejected |
| Rank-based bonus | −0.003 | Rejected |
| Add the 8B classifier trained on all 11,426 Claude rows (0.692 alone) | +0.004 with v1, −0.004 with the ranker average | Noise |
| Swap it for the teacher-only 8B member | −0.003 with v1, +0.003 with the average | Noise |
| Add the catalogue-trained or the reading classifier | −0.002 to −0.003 | Rejected |
| **Average ranker v1 and v2** | **+0.0069, better in 95% of 20 splits; +0.007 [−0.002, +0.017] on the standard split** | Only one that holds |

**16.3 What moves hard rows, measured.** Held-out error rows whose pair is unseen: 71.6% with 600 teacher rows, 62.9% with 900, 55.7% with 1,200.
Fused score by teacher examples of the exact pair: none 0.500 (106 rows), one 0.820 (61), two or three 0.955 (26). Claude-labelled examples of the
pair give 0.656 (group A). More teacher-style labels are the one ingredient with a large measured effect, and we have no source of them. The test-time
candidate text already carries teacher spans from all 1,500 rows, and every blend member used for test was trained on all 1,500, so retraining the ranker
on the dev rows would change roughly 4% of rows by a small amount and cannot be checked on dev. Not proposed.

**16.4 Design: ranker average.** Score the 3,498 test rows with the saved v2 adapter (no training), average v1 and v2 (geometric mean of their
candidate probabilities), fuse with the blend exactly as the submitted file does (w 0.5, λ 0.75, both chosen by every fold on dev).
- Dev: 0.7597 against 0.7525 on the standard split (no-error 0.921, seen pair 0.872, hard 0.506). v2 gets the first answer right more often than v1
  (192 rows against 186) and v1 ranks second and third better; they disagree on the top answer on 60 of 301 rows.
- Cost: adapter upload about 16 min, setup 5 min, check 2 min, test scoring 60 min (v1 took 3,608 s), copy and terminate 3 min: about 1.45 h on an
  RTX 6000 Ada at $1.09/hr, about $1.60; working limit $2.20. Script `runs/deep/pointwise/pod_run_pw2_test.sh` (syntax-checked, not run).
- On-pod check before test scoring: the uploaded adapter must reproduce v2's dev scores on 20 rows, or the pod stops (about $0.45 spent).
- There is no new dev result to wait for; the dev evidence is the row above. The public leaderboard decides whether the new file replaces 0.73183,
  and a submission needs the user's explicit yes.
- Expected: v1's dev gain of +0.019 became +0.0147 on the leaderboard, so +0.007 on dev suggests about +0.005: public about 0.737, plausible range
  0.730–0.745. It does not change the distance to 0.80.

**16.5 Honest position.** With $5.93 I have no design with evidence that reaches 0.80, and I found none that would at a larger budget either: every
open model we have trained or run sits at 0.44–0.50 on hard rows against Claude's 0.64 on the same transcripts.

### 16.6 Result of the ranker average, 14 Nov 00:50 UTC

Approved by the user at about 23:20 UTC on 13 Nov ("Yes, run it"). Pod `9wihb4qkifb0jp` (RTX 6000 Ada, $1.09/hr), 23:24–00:45 UTC, **$1.47** against a quote of about $1.60 and a limit of $2.20. RunPod total $75.54 of $80.

- Upload from this Mac fell to about 160 kB/s, which would have pushed the job past its limit. The adapter was therefore sent as 16-bit floats (257 MB instead of 514 MB; worst relative rounding error 4.9e-4, far below the model's own bfloat16 arithmetic) in eight parallel chunks, checksum verified on the pod.
- On-pod check on 20 dev rows: same candidates, same top answer on 20 of 20, mean score difference 0.032 (an exact reload of v1 gave 0.027 on its own check). Passed.
- Test scoring: 3,498 rows, 34,980 pairs, 3,727 s (9.4 pairs/s). Same ten candidates per row as v1; v1 and v2 disagree on the top answer on 21.5% of test rows (19.9% on dev).
- Averaged with `scripts/average_bundles.py` (geometric mean), fused with `scripts/fuse_candidate.py` at w 0.5, λ 0.75 (all five folds and full dev choose these). Dev through the official script: **0.7597** (hard rows 0.506), +0.026 [+0.009, +0.044] over the cross-fitted blend reference; +0.007 [−0.002, +0.017] over the submitted file.
- Staged: `artifacts/submission-3way-pw14avg/submission.csv` (sha256 1c404546…, 3,498 rows, ids in test order, three distinct valid codes per row). Against the submitted file the top answer differs on 2.7% of rows and the top-three set on 13.6%. **Not submitted**; needs the user's explicit yes.

**Leaderboard, 14 Nov 00:55 UTC.** Submitted with the user's permission for this file (ref 58326821): **public 0.72550**, against 0.73183 for the v1-only file (−0.006) and below the 0.730–0.745 range I had given. The best file is still `artifacts/submission-3way-pw14` (rank 77 of 433). 781 test rows differ between the two files (94 in the top answer, 687 only in second or third place). The dev gain of +0.007 had an interval of −0.002 to +0.017 on 301 rows; the leaderboard says the true effect of adding ranker v2 is zero or negative, which agrees with v2 being the weaker ranker alone (0.696 against 0.709). Ranker v2 is rejected for the final ensemble.

## 17. What the Kaggle code and discussion tabs show (14 Nov 01:40 UTC)

Read: all 21 discussion threads, the host's two blog posts linked from them, and 13 of the 23 public notebooks (code pulled to the scratchpad, not run).

**Host reference scores on the public leaderboard split, models used off the shelf with no training on the labels** (blog post of 17 Oct 2026, "Understanding how students think"):

| Model | Public MAP@3 |
|---|---|
| Claude Opus 5.5 | 0.792 |
| GPT-6 Astra (low) | 0.790 |
| Gemini 4 Argon | 0.775 |
| Qwen 3.8 27B (zero-shot) | 0.757 |
| Gemma 4 31B (zero-shot) | 0.710 |

The same post says the leading teams were at 0.815–0.823 on 17 Oct "using open-weight models that run on Kaggle's free notebook GPUs", and that the public leaderboard is about 29% of the test set (about 1,000 workings).

**What this means for us.** Our whole pipeline (public 0.73183) is built on Qwen3-generation models (Qwen3-32B, Qwen3-14B, Qwen3-8B, Qwen3-VL-8B). `Qwen/Qwen3.8-27B` (Apache-2.0, vision-language, released August 2026, 27.8B parameters, FP8 and 4-bit builds on Hugging Face, several Kaggle mirrors) scores 0.757 on the public split with no training at all, by the host's own measurement. We never ran it. This is the first lead with outside evidence that the base model, not the method, is our main weakness.

**Rules confirmed by the host.** Models up to the size of Nemotron 550B are allowed (ensembles too, total around that size); compute is unrestricted; API labels on train and unlabelled data are allowed, never on test; the final solution must run offline; winners must hand over the training and inference pipeline, and both will be verified. Root code is "most of the time the earliest" error but can be the one the teacher judged the main cause. NONE share in test "should be similar" to train.

**Host research blog (20 Aug 2026).** On 400 workings with frozen text: fine-tuning on about 3,400 labelled rows lifted Gemma 4 31B from 52% to 64%, Nemotron Super 120B from 45% to 63% and Nemotron Ultra 550B from 59% to 66% top-1; best closed model 73%. Size alone buys little; fine-tuning buys 7–18 points.

**Public notebooks.** None scores above 0.635 and none uses Qwen3.8-27B (they use Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3.5-9B with question priors, TF-IDF, answer-match priors and reranking). We already have or have rejected each of those ideas. Nothing to borrow from the code.

**Proposed test (not run, needs approval).** Stage 1: Qwen3.8-27B-FP8 zero-shot on the 301 dev photos with our existing Claude prompt, thinking off and on, on one RTX 6000 Ada: about $1. Pass mark: alone at least 0.74 on dev, or fused with the current system at least 0.7675 (+0.015). Stage 2 only if it passes: test predictions, fuse, stage a CSV.

### 17.1 Result: Qwen3.8-27B zero-shot on dev, 14 Nov 04:10 UTC

Approved by the user ("Yes, run it"). Pod `o2cnknpzd2vm11` (RTX 6000 Ada, $1.09/hr), 02:42–03:57 UTC, **$1.37** against a quote of about $1 and a limit of $1.50. RunPod total $76.91 of $80.

Setup: `Qwen/Qwen3.8-27B-FP8` on vLLM 0.33.0, the same prompt and 1024-pixel photos Claude was given, answers constrained to the JSON schema. The first server start failed on a missing build tool (`ninja`, for vLLM's sampling kernels); fixed with `vllm download-kernels` (now in the script), about 6 minutes lost. The GPU has no native FP8, so throughput was only about 150 output tokens a second; the no-thinking run was stopped at 102 photos to leave time for the thinking run, which reached 270 of 301 photos before the time limit (the 31 not started are the last rows in file order, all error rows). All comparisons below are on the same rows.

| 270 dev photos | All | No error | Seen pair | Hard |
|---|---|---|---|---|
| Submitted system | 0.772 | 0.911 | 0.868 | 0.502 |
| Qwen3.8-27B, thinking (effort medium), alone | 0.759 | 0.924 | 0.709 | 0.608 |
| Fused (submitted x Qwen, cross-fitted) | **0.805** | 0.922 | 0.851 | 0.614 |
| Claude Opus with photo, for reference | 0.823 | 0.934 | 0.795 | 0.717 |

- Fused minus submitted: **+0.033 [+0.006, +0.060]**; alone minus submitted −0.013 [−0.063, +0.035]. Folds choose floor 0.01, weight 1, lambda 0.5 (four of five).
- Without thinking (102 photos): 0.657 alone against 0.717 for the submitted system on those rows. Thinking is what makes the model useful here.
- Pass mark: fused at least +0.015 met; "alone at least 0.74" is borderline (0.759 on rows where our system scores 0.772 rather than 0.752).
- Zero-shot it is weak where question-specific experience matters (seen pair 0.709) and strong where we are weak (hard rows +0.11). That is why fusing works, and why fine-tuning it on our labels is the natural later step.
- Sending only the half of rows where our system is least sure (top probability below 0.82) keeps the whole gain (+0.039 against +0.039, fixed settings, in-sample) with 61% of the tokens.

**Proposed next step (not approved).** Score the 1,593 test photos (45.5%) where the submitted system's top probability is below 0.82 with the same model and settings, fuse, stage a CSV. Measured speed 12.5 s per photo gives about 6.7 GPU-hours: about $7.80 on two RTX 6000 Ada pods (about 3.5 hours), working limit $9.50, which needs the $80 cap raised to $88. Expected public about 0.76 (range 0.74–0.78).

### 17.2 Result: Qwen3.8-27B on the least-confident test photos, 14 Nov 07:55 UTC

Approved by the user at about 04:20 UTC ("yes, run it"; cap raised to $88). Two RTX 6000 Ada pods, 04:24–07:28 and 04:24–07:50 UTC, **$7.08** against a quote of about $7.80 and a limit of $9.50. RunPod total $83.99 of $88.

- All 1,593 selected test photos answered (submitted system's top probability below 0.817), none failed, none cut for length; about 2,456 output tokens and 15 seconds each. Pod A ran about 20% slower than pod B, so the last 70 photos of its final part were moved to pod B mid-run.
- Behaviour on test matched dev: Qwen's top answer equals ours on 40% of sent rows (39% on dev), is in our top three on 73% (72%), and is NONE on 11% (11%).
- Dev under the exact test procedure (only least-confident rows fused; `scripts/fuse_vlm.py`): 0.804 against 0.775 on the same 270 rows, +0.029 [+0.002, +0.055]; chosen floor 0.01, weight 1, lambda 0.5.
- **Submission 10 (58335588), 07:21 UTC, with the user's explicit "submit now":** the file built from the first 1,392 answers, `artifacts/submission-3way-pw14-q38-partial`. **Public 0.76266, the new best** (+0.031 over 0.73183; the estimate was 0.76). Rank about 61 of 437.
- Complete file (all 1,593 answers) staged and validated: `artifacts/submission-3way-pw14-q38/submission.csv`; it differs from the submitted file in the top answer on 2.6% of rows. Not submitted.
- This is the first dev gain since the ranker that transferred to the leaderboard at full size.

### 17.3 Result: second-opinion (rerank) tests on dev, 14 Nov 09:30 UTC

Approved by the user at 08:35 UTC ("Yes, go ahead", cap $94); at 08:55 UTC the user set a hard limit of the $6.52 then in the RunPod account. One RTX 6000 Ada, 08:37–09:23 UTC, **$0.83**. RunPod ledger $84.82.

Both tests ran on the 135 dev rows where Qwen3.8 is used (submitted system's top probability below 0.817). Qwen3.8 (thinking, effort medium) saw the photo again with its own first transcription and codes and our system's top eight codes with probabilities; `rerank_ex` also saw up to four teacher-labelled workings of other students on the same question (fit rows only). Current fused system on these rows: 0.7136 (no error 0.637, seen pair 0.823, hard 0.652).

| Variant | Score on the 135 rows | Difference | Whole-dev effect |
|---|---|---|---|
| rerank_ex used alone | 0.7123 | −0.001 [−0.059, +0.056] | −0.001 |
| ours × rerank_ex² | 0.7296 | +0.016 [−0.031, +0.059] | +0.008 |
| ours × rerank_plain × rerank_ex (best of about 15 tried) | 0.7333 | +0.020 [−0.027, +0.070] | +0.010 |
| rerank_plain used alone | 0.6827 | −0.031 [−0.093, +0.029] | −0.015 |
| ours × rerank_plain | 0.7074 | −0.006 [−0.047, +0.033] | −0.003 |
| ours × mean of the three Qwen answers² | 0.7185 | +0.005 [−0.037, +0.047] | +0.002 |

- Pass mark (+0.01 whole-dev, clear of noise) not met. Stage 2 not run. The second-answer arm was dropped before it ran (it could not be afforded on test under the $6.52 limit).
- Teacher examples lifted seen pairs inside these rows (0.86–0.88 against 0.82) but not hard rows (0.62–0.66 in every variant). Top-1: rerank_ex 61.5%, rerank_plain 57.8%, current fused 62.2%.
- All three Qwen answers agree on 84 of 135 rows (top answer right 70%); where they differ, 37%.

**Free analysis afterwards (same day).**

- Same 102 rows, thinking off against thinking medium: 0.657 → 0.794; hard rows 0.549 → 0.711 (n = 34).
- Our medium-effort run is worth about 0.755 on full dev, which converts to about 0.728–0.734 public (dev runs 0.021–0.027 above public for our system and for Claude). The host measured Qwen3.8-27B at 0.757. Our prompt reproduces the host's Claude figure (dev 0.819 → about 0.792–0.798 expected, host 0.792), so the shortfall of about 0.025 for Qwen is most likely a setting: we ran reasoning effort medium, the model's default is the maximum; we also ran FP8 weights.
- Hard rows (83): Qwen's first answer is right on 53%, the truth is among its three on 70%, in our top three on 73%, and in one or the other on 84%. When Qwen's first answer is wrong it is a different error family 85% of the time; Claude is right on 41% of those.
