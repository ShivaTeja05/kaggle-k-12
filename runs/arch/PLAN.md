# K-12 architecture plan (12 Nov 2026)

This plan pulls together 6 diagnoses, 4 proposals, 4 adversarial CPU tests and 3 judges. Nothing here was trained and nothing was submitted. All numbers are on the 301-row dev fold (fold 0) unless marked otherwise. Each number is tagged:
- **M** = measured. Where tuning is involved it was cross-fitted, and CIs are question-grouped bootstrap 95% intervals.
- **E** = estimate or simulation.
- **O** = oracle diagnostic. It uses Claude's dev outputs or gold labels, so it cannot be deployed.

**Summary**
- Almost all of the 0.10 gap to Claude comes from one thing: naming an error type that is **new for a question we already know**. It does not come from reading, model size, NONE handling or the combiner.
- One gain survived every adversarial test: Sinkhorn label balancing. It costs nothing (CPU only) and gives +0.019 on dev, about +0.01 expected on public.
- The only architecture change worth GPU money is a **reason-then-choose 8B student** that picks from the blend's top-10 candidates. It gets one gated run of about $5.
- No offline proxy we have supports that student yet. 0.80 by 17 Nov is not a realistic expectation.

---

## 1. Where the 0.10 gap to Claude actually lives

**Reference blend.** Geometric blend of three models, reproduced at **0.7182** by all six diagnosis passes:

| Weight | Model | Dev bundle |
|---|---|---|
| .35 | Qwen3-32B text, machine transcripts | `runs/pod/text32/validation_tx_dev_epoch_1.npz` |
| .30 | Qwen3-8B text, machine transcripts | `runs/pod/tx/text_tx_dev_e3.npz` (`_e2` gives 0.6977) |
| .35 | Qwen3-VL-8B image classifier distilled from Claude | `runs/pod/dev-distill/validation_epoch_2.npz` |

Claude Opus zero-shot scores **0.8189**. The gap is **0.1008, CI [0.063, 0.138]** (M).

| Where | Rows | Blend | Claude | Share of gap | Kind |
|---|---|---|---|---|---|
| Error rows from a seen question whose true code never appeared for that question in the fit rows | 84 | 0.431 | 0.712 | about 78% (+0.282, CI [0.204, 0.366]) | M |
| Error rows from **unseen questions** | 22 | 0.432 | 0.727 | about 21% | M |
| Error rows whose code **was** seen for that question | 94 | 0.830 | 0.812 | −5.5% (blend wins) | M |
| NONE rows | 101 | 0.916 | 0.934 | about 6% | M |

One diagnosis (ensemble-anatomy) counts 113 "seen question, new code" rows instead of 84. The difference is that it also counts NONE rows where NONE was new for the question. The conclusion is the same: these rows carry 0.0786 of the gap, CI [0.053, 0.106].

**What this means.** Our fine-tuned models have learned which codes usually go with each question. When the student made a different kind of mistake, they fall back to the question's usual codes.
- On error rows whose code is new for the question, the blend's top-1 is one of that question's training codes 36% of the time. Claude's is 11% (M).
- Where the code was already seen for the question, we already beat Claude.

**Other views of the same gap** (all M):
- **Wrong part of the taxonomy.** On 55 rows the blend's top-1 is in the wrong taxonomy section; together they account for 0.074 of the gap.
  - Section accuracy on error rows is 0.63 for the blend and 0.72 for Claude.
  - The main confusion is between generic codes (CAL, WOP, FRM, ANS, OTH) and topic-specific codes.
  - Biggest families: ALG 0.021 (blend 0.21 vs Claude 0.77 on 11 rows), WOP 0.017, ANS 0.011, EXP 0.011.
- **Ranking, not recall.** The truth is in the blend's top 3/5/10 on 0.831/0.894/0.940 of rows.
  - A perfect reranking of the top 10 would score 0.940 (O).
  - Claude, restricted to the blend's top 10, still scores 0.802–0.807 (O). The candidate list does not stop us reaching 0.80.
- **Low confidence points to the problem rows, but only roughly.** The least confident fifth of rows (61 rows) holds 49% of the gap; both models score 1.000 on the most confident fifth. But blend confidence separates the unseen-pair rows only weakly (AUC 0.674).

**What the gap is not** (all M):

| Suspect | Best measured gain | Why it is not the lever |
|---|---|---|
| Reading / transcriber | +0.0155 (32B on gold transcripts, CI [+0.006, +0.026]); +0.022 if all text models had gold (E) | 17.0 of the 30.3 gap points are on rows our transcriber read exactly right. Claude reads no better than our transcriber (median page character error rate 0.124 vs 0.095). |
| NONE detection / answer checker | perfect NONE oracle +0.041; crude checker +0.002 | The blend already separates NONE well (AUC 0.947). |
| Model size / more labels | 235B few-shot scores 0.650 alone. Going from 0% to 100% Claude labels moved the single 8B from 0.553 to 0.692, but the ensemble only 0.715 vs 0.718. | More of the same supervision does not reach unseen pairs. The best 8B (scale_100) agrees with Claude's top-1 on only 49% of unseen-pair rows. |
| Combiner / priors / stacking / reweighting | every cross-fitted variant between −0.031 and +0.006 | Covers question priors (−0.004), LR and gradient-boosting stackers, NONE gates, family bias and routing. |
| Choosing a secondary error instead of the root | about 0.018 at stake | Only 8–9 rows. |

**Honest baseline.** The 0.718 includes about **0.012 of optimism** from picking weights on dev. Cross-fitted weights give 0.706, which matches the public 0.7085 (M).

**Root cause in the training recipe** (M, from reading the code):
- `src/k12/text_diagnose.py` classifies from the **last prompt token, before any reasoning** happens.
- The explanation target puts the code first (`ROOT: X / WHERE / WHY`).
- Claude wrote those explanations with the gold label already in hand. They mention the answer-key numbers in 58% of rows, against 88% for Claude's own zero-shot reasoning.
- Codes appear only as `id: name`, never with descriptions. The 921-token code list takes up 79% of every prompt.

## 2. Recommended pipeline

```
                    test photo
                        │
        ┌───────────────┴────────────────┐
        │ S0  Qwen3-VL-8B transcriber     │  (exists; tx_dev.json / tx_test.json)
        └───────────────┬────────────────┘
                        │ machine transcript
   ┌────────────────────┼─────────────────────────────┐
   │ STAGE A: memorising candidate generator (exists) │
   │  32B tx (.35)  +  8B tx e3 (.30)  +  VL distill (.35)  [+ 32B-all retrain only if cross-fit Δ>0]
   │  geometric blend → P0, top-10 candidates (recall@10 0.94)
   └────────────────────┬─────────────────────────────┘
                        │ top-10 codes (+NONE) with name + description
   ┌────────────────────┴─────────────────────────────┐
   │ STAGE B (one gated GPU bet): reason-then-choose student      │
   │  Qwen3-8B LoRA, plain causal-LM SFT, no custom head          │
   │  input : grade, topic, question_text, correct_answer,        │
   │          numbered transcript, 8-12 candidates (alphabetical) │
   │          NO question_id, NO full 76-code list (~400 tok)     │
   │  output: ROOT_LINE → SOLVE/WHERE/WHY → CODES: c1 c2 c3       │
   │  train : 26k Claude zero-shot traces (class-capped, OTH-00↓) │
   │          + 1,199 fit gold rows (3x, last 30% of steps)       │
   │          hard negatives = codes already seen for the question│
   │  greedy vLLM, ≤320 new tokens                                │
   │  → P_B = emitted rank shares (+ optional CPU TF-IDF head on  │
   │          its own rationale, + description similarity)        │
   └────────────────────┬─────────────────────────────┘
                        │
   ┌────────────────────┴─────────────────────────────┐
   │ STAGE C (CPU): P ∝ P0 · P_B^w   (w cross-fitted, 5 question folds) │
   │              → Sinkhorn balancing, λ = 0.5, toward train label     │
   │                marginals, bias estimated label-free on test        │
   │              → top-3 → staged CSV → ASK USER before submitting     │
   └──────────────────────────────────────────────────┘
   If Stage B fails its gate: ship  Stage A → Sinkhorn  (exists today, $0).
```

**Why this shape, and where each part comes from:**
- **Stage A stays.** It wins on rows whose code was seen for the question (0.830 vs Claude 0.812). Every proposal and judge agreed.
- **Sinkhorn balancing** (label-structure diagnosis). It is the only component that survived adversarial testing.
- **Reason first, then choose, trained as plain fine-tuning** (pragmatist proposal). This fixes the problem that the code is chosen before any reasoning. Plain fine-tuning was chosen because it carries the least engineering risk: the repo has no trainer for a classification head after the rationale (the R1D proposal) and no code for reading letter probabilities (the verifier proposal).
- **Hard negatives from the question's own codes** (verifier proposal). They attack the question-to-code shortcut directly.
- **Hide question_id; show descriptions for the candidates only** (structured and R1D proposals). Dropping the full code list is also what makes descriptions affordable: about 400 tokens instead of 3,002.
- **Corrected gates** (all three adversarial testers). The proposals' own gates could pass with zero real gain.
- **Train on fit rows only, and use the same model for the dev gate and for test** (pragmatist and structured proposals). This keeps dev numbers honest and saves a retrain.

**Not adopted:**
- The mechanism-text bottleneck (structured proposal). Passing the decision through text loses information: −0.025 against fusing Claude's codes directly (M). Its own kill threshold is already met.
- The R1D custom head and same-question batch sampler: too much new code for a single attempt.
- 4 samples × 2 candidate orders per row: unmeasured, and the largest inference bill.

## 3. Ordered build plan

**Budget available:**
- **RunPod.** About $54 was spent by 00:01 UTC 12 Nov. Pod `a7cevylabnswz1` (A100, $1.79/hr) runs until about 04:30 and will add about $8. Keeping the $1 safety margin, **about $15 is really spendable** under the $80 cap.
- **Claude API.** $186.61 of $200 spent, **$13.39 left**, for training and unlabelled photos only.
- **Deadline:** 17 Nov 00:00 UTC.

### Step 0: Stage the Sinkhorn CSV and ask for permission
- **Compute and cost:** CPU, $0, already done.
- **What changes:** λ=0.5 Sinkhorn balancing on the current blend. The CSV is already at `runs/arch/label-structure/staged_submission_balanced_l0.5.csv`.
- **Gain:** **+0.019 dev (M)**: 0.7370, CI [+0.006, +0.033], with λ=0.5 chosen in all 5 folds.
  - Expect about **+0.01 public (E)**. The dev fold was built to match the fit rows' label counts, which flatters this step.
  - With the noise a random split would add, the gain averages +0.011 and stays positive in 98% of draws.
- **Gate:** submit only with your explicit OK.
  - A public score is also the only real test that test labels follow the train label mix.
  - With the wrong target mix the gain flips to −0.066 (M).

### Step 1: Score the 32B-all retrain when it lands (about 04:30 UTC)
- **Compute and cost:** CPU, no extra cost (the pod time is already committed).
- **What changes:** add it as a 4th member, with its weight in {0..0.5} cross-fitted, with and without Sinkhorn. Do **not** extend the pod; terminate it when done.
- **Gain:** 0 to +0.01 (E). Precedent: adding scale_100, the best single model, gave −0.007 to −0.014 (M).
- **Gate:** keep it only if the cross-fitted gain over blend+Sinkhorn is above 0 and the CI is not entirely below 0.

### Step 2: Build the Stage B dataset
- **Compute and cost:** CPU, $0, about 5 hours. New script under `runs/arch/student/`.
- **Targets.** Built from `runs/claude/unl_all_v4.jsonl` in Claude's order (ROOT_LINE → reasoning → CODES), not from the code-first `data/v3` targets.
  - Cap about 300 rows per first code.
  - Keep 3k NONE-first rows.
  - Weight OTH-00 at 0.3 (Claude's precision on it is 0.21).
- **Gold rows.** The 1,199 fit gold rows get WHERE/WHY rebuilt from `train_explain_opus` with the code last. They are repeated 3 times in the final 30% of steps.
- **Candidate lists** (8–12 per row, alphabetical, each written `id: name - description`):
  - the label and Claude's other two codes;
  - every code seen for that question in fit gold rows or Claude labels;
  - same-section siblings, NONE, and random codes to fill.
- **No dev leakage.** Any "new code for this question" weighting is computed against fit rows only, and dev rows never touch training. The compliance judge flagged a leak of this kind in the R1D proposal.
- **Dev and test candidates** come from the blend's top 10. Cap the 20 looping and 26 overlong test transcripts.
- **Gain:** none by itself; it enables Stage B.
- **Gate (sanity checks):**
  - no dev trace_ids in training;
  - 99th-percentile prompt length under 1.6k tokens;
  - the label is in the candidate list on 100% of training rows, and on about 94% of dev rows (the same as recall@10).

### Step 3: Smoke-test training and generation on a pod
- **Compute and cost:** GPU (RunPod), about $0.5–1, about 1 hour, on an A6000 ($0.59/hr) or L40S ($1.09/hr).
- **Why it is needed:** neither path exists in the repo today. `text_diagnose.py` only has a classification head, and `scale_probe.py` / `vllm_ranker.py` only run base models through vLLM.
- **What it does:** 200 training steps, then generate on 20 dev rows.
- **Gate:** the output parses (`CODES:` present on at least 95% of rows), and measured throughput puts a full epoch under $4. Otherwise stop and ship step 0.

### Step 4: Train Stage B for one epoch, on fit and Claude rows only
- **Compute and cost:** GPU, about $3–4.5. This is an estimate scaled from scale_100's 12,666 s per epoch; it has not been timed.
- **What changes:** plain fine-tuning of a LoRA on Qwen3-8B, about 14k rows of about 1.1k tokens.
- **Interim abort:** at 50% of the epoch, generate on 100 dev rows (about $0.3).
- **Gain: estimate only.**
  - Simulation, assuming the student's mistakes are random: a student scoring 0.741 alone fuses to 0.769, and one scoring 0.713 fuses to 0.754.
  - If its mistakes overlap the blend's, the same quality fuses to only 0.741, which is +0.004 over Stage 0.
  - The best real stand-in (scale_100 on machine transcripts) is −0.008 against blend+Sinkhorn (M).
  - Central estimate: **+0 to +0.02 over blend+Sinkhorn**.
- **Gate (interim):** abort if, on those 100 rows, the student's recall@3 on unseen-pair rows is no better than the blend's.

### Step 5: Dev generation and the real gate (301 rows)
- **Compute and cost:** GPU about $0.3, then CPU.
- **What changes:** fuse P0 · P_B^w with w cross-fitted, then apply Sinkhorn.
- **Gate: all four must hold** (corrected gate, agreed by the judges). If any fails, terminate the pod and ship step 0 or step 1.
  - (a) The fused cross-fitted score beats **blend+Sinkhorn (0.737)** by at least **+0.015**, with the bootstrap CI's lower bound above 0.
  - (b) MAP@3 of at least 0.50 on the 106 unseen-pair and unseen-question rows (blend 0.431, scale_100 0.44).
  - (c) Top-1 agreement with Claude of at least 0.65 on the unseen-pair rows (scale_100: 0.49).
  - (d) Agreement with Claude of at least 0.45 on the 102 rows where the blend and Claude disagree (best existing student: 0.245).

### Step 6: Test generation, fusion and staging
- **Compute and cost:** GPU about $1–1.5, then CPU.
- **What changes:** run the same fit-only Stage B model on the 3,498 test transcripts, with candidates from the full-train test bundles. Stage the CSV.
- **Gate:** check that the weight w chosen on dev still behaves sensibly when applied to the full-train test bundles (see question 2 in section 6). Then ask you before submitting.

### Claude API reserve ($13.39)
- Spend nothing by default.
- The only justified use is Opus zero-shot traces on the 1,199 fit **training** photos (about $8.3 at the measured $0.0069 per photo).
- Do it only if step 5 shows that the gold rows' label-conditioned rationales are the weak link (checked with a gold-only ablation). Expected gain 0 to +0.01 (E).
- AGENTS.md requires a measured development gain before each API stage. Never on test photos.

**Totals and timeline.**
- Planned RunPod GPU spend: **about $5–7**. That fits inside the roughly $15 headroom with room for one failed attempt.
- No Kaggle GPU: your CPU-only instruction for K-12 still stands.
- Schedule: step 0 now; step 1 around 05:00 UTC 12 Nov; steps 2–3 on 12 Nov; steps 4–5 by 13 Nov; step 6 by 14 Nov. That leaves 2 days of slack for a failure and for your submission decisions.

## 4. What not to do, and why

| Do not | Evidence (M unless marked) |
|---|---|
| Tune blend weights on dev in-sample, or quote 0.718 as the honest baseline | Cross-fitted weights give 0.706 (−0.012, CI [−0.021, −0.004]); public is 0.7085. |
| Build more combiners, stackers, question/topic priors, NONE gates, family bias or diversity reranks | Every cross-fitted variant lands between −0.031 and +0.006. Question priors give −0.004 to −0.005. |
| Upgrade or retrain the transcriber, including 5-fold out-of-fold transcriber retrains | Perfect reading is worth at most +0.0155 (M) / +0.022 (E). Reading quality outside the root line is worth +0.001. |
| Run bigger models (32B/235B variants, or 235B on test) | 235B few-shot gives +0.016 with a CI crossing 0, is −0.017 against Stage 0, and costs about $15 to run on test. Its gain is entirely on code-seen rows; unseen-pair rows go 0.431 → 0.429. |
| Buy more Claude labels for volume | Going from 0% to 100% Claude labels left the ensemble at 0.715 vs 0.718. Claude labels on other photos of the same question contain the gold code for only 32–42% of unseen-pair rows, and retrieving from them fuses at +0.003. |
| Build the mechanism-text bottleneck (structured proposal) | Going through text loses −0.025 against fusing codes directly. Break-even needs 82–86% of Claude's mechanism quality; the best stand-in reaches 0.76–0.78 and gives +0.0024. |
| Build the R1D custom post-rationale head, same-question batch sampler and 5x sampling | Too much new, untested code for one attempt. Its 70%-agreement gate corresponds to +0.008 (CI [−0.011, +0.026]), which is noise. |
| Add the 4×2 sampling or the optional $5 second epoch | Unmeasured. Revisit only if step 5 passes with margin and budget remains. |
| Treat the Claude-text "ceilings" (0.7795 / 0.778 / 0.772 / 0.832) as evidence for an 8B student | They mostly decode Claude's own decision (79–91% top-1 agreement with Claude) (O). Rationales written by open models, run through the same decoders, give −0.013 to +0.013. |
| Put all 76 code descriptions in the prompt | 3,002 tokens; 100% of prompts would be truncated. |
| Submit anything without your explicit permission for that specific file | Your rule from 11 Nov. |

## 5. Honest expected range

| Scenario | Dev (cross-fitted) | Public (E) | Probability (E) |
|---|---|---|---|
| Stage A + Sinkhorn only (ships whatever happens) | 0.734–0.737 (M) | about 0.715–0.72 (about +0.01 over 0.7085). Could be **lower** if test labels do not follow the train label mix. | n/a |
| Plus the 32B-all retrain as a 4th member | +0 to +0.01 (E) | +0 to +0.01 | about 40% that it helps at all |
| Stage B passes the corrected gate | 0.75–0.76 (E) | about 0.73–0.75 | about 25–35% |
| Stage B reaches Claude-like accuracy on unseen pairs (needs about 0.77 alone and 0.60 on unseen pairs) | about 0.79–0.80 (E) | about 0.78 | at most 10–15% |

**Nothing measured supports 0.80+.** Every deployable stand-in we have (distilled 8Bs, 32B/235B zero-shot and few-shot, retrieval) adds −0.017 to +0.008 over blend+Sinkhorn.
- Reasonable expectation for the public board by 17 Nov: **0.72–0.74**, with 0.75+ only if Stage B clears its gate.
- Dev gains on unseen pairs will probably transfer to public at somewhat less than face value (E). Test questions have 3.66 training rows each, against 2.78 fit rows for dev questions. That suggests only about 40–45% of test rows fall in the new-code regime.

## 6. Important questions no stage answered

1. **Do test labels follow the capped train label mix?**
   - Sinkhorn's sign depends on it: −0.066 with Claude's label mix, −0.019 with a uniform one (M).
   - Train looks capped near 57 rows per code, which suggests test is too, but nobody verified it.
   - One permitted public submission of the staged CSV is the cheapest check.
2. **Do the dev-tuned settings transfer to the test models?**
   - Fusion weights and λ are fitted on dev bundles from models trained on fit rows only. Test uses models trained on all of train (`text32-full`, `text-full`, `full-distill`).
   - I checked: **no fit-only test bundles exist**.
   - Full-train models have seen more codes per question, so they may be sharper and memorise more. Nobody measured whether the same w and λ still work.
   - Mitigation: compare the confidence and entropy distributions of the dev and test bundles, and check sensitivity to w across [0.5w, 2w].
3. **How much do results vary from seed to seed?**
   - No fine-tuned model has a second seed. The +0.032 from epoch 2 to epoch 3, and every single-model comparison, may be partly seed noise.
   - On 301 rows the CIs are already about ±0.03, so a Stage B gate missed or passed by 0.01 is not decisive.
4. **Is breaking the question shortcut worth anything on its own?**
   - Masking or dropping question_text and correct_answer during training, and penalising the question prior, were never tested.
   - Step 4's hard negatives are the plan's only attack on the shortcut. A single run cannot separate their effect from the reason-first change.
5. **How much worse is the student on its own rationales than on Claude's?** Unmeasured. Step 5 only measures the end result. If it fails, we will not know whether the student cannot reason, or cannot reason from text it wrote itself.
6. **Do Claude's 26k labels cover the right codes on other questions?**
   - We know coverage on the same question is weak (32%).
   - Nobody measured, for the 106 unseen-pair dev rows, how many training traces carry the true code on other questions of the same topic or section.
   - That decides whether moving an error type across questions can be learned from this data at all.
7. **How big is the public split, and what is the daily submission limit?** Both unknown. So we cannot say whether a +0.01 change will even show on public, or how many staged CSVs you can realistically test before 17 Nov.
8. **What exactly is the running 32B retrain?**
   - Its output folder (`text32-explain`) is still empty.
   - Check its dev bundle and config before step 1. If it uses the code-first targets, it is a model-size bet, which section 1 says is not the lever.
9. **Does Sinkhorn still help once Stage B is fused in?** Balancing each member separately was tested and was no better than balancing the blend. Nobody tested it after adding Stage B, so step 5 must choose λ and w together.
10. **Will choices made on dev hold on the private board?**
    - Every selection used one 301-row fold.
    - Selecting on the 106-row unseen-pair group (CI about ±0.08) invites overfitting.
    - No proposal kept a second fold, or k-fold predictions on the 1,199 fit rows, for confirmation; no out-of-fold predictions exist for those rows.

## Key files

- Plan: `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/PLAN.md`
- Diagnoses: `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/{ensemble-anatomy,combiner-headroom,label-structure,reading-impact,teacher-gap,recipe-audit}/`
- Proposals: `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/design-{structured,combine,distill,pragmatist}/`
- Adversarial tests: `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-{structured,combine,distill,pragmatist}/`
- Sinkhorn code and staged CSV (not submitted):
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure/s7_sinkhorn.py`
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure/staged_submission_balanced_l0.5.csv`
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure/test_blend_balanced_l0.5.npz`
- Corrected gate scripts to reuse:
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-pragmatist/t3_corr_gate.py`
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-pragmatist/t6_real_gate.py`
  - `/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/test-combine/complement.py`