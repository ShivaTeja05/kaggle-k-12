# Architecture and training changes from the literature

11 November 2026. Five parallel paper surveys and one rules check, merged with the coordinator's CPU probes under `runs/arch/*` and three new CPU probes run for this report (`runs/research/{overcorrection,knn-probe,family-probe}/`). `runs/arch/PLAN.md` and `runs/research/REPORT.md` did not exist yet when this was written; this file uses their probe outputs directly and should be reconciled with PLAN.md once it lands.

All dev numbers are fold 0 (301 rows), MAP@3, with the paired question-bootstrap 95% CI where one exists. Reference blend = **0.718**. Claude Opus zero-shot = **0.819**. Gains marked "est." are estimates, not measurements.

---

## 1. Summary

**What the literature and our probes agree on.** Rescaling or recombining the outputs of our existing three models is exhausted: stacking, routing, question priors, kNN, family factorisation and verifiers give ≤ +0.002 or lose. The one exception is **label-shift correction**, which is free and measured positive. Bigger models are not the answer either:
- ErrorRadar finds that math-specialised and larger models are not better diagnosers.
- MathCog finds every one of its 18 models below F1 0.5.
- In MAP 2025, Qwen3-8B to 32B gained only +0.003 for the winner.
- Our 235B few-shot scored 0.650.

The remaining gains come from **how the models are trained and what they are shown**:
- Gold-label refinement after pseudo-label training.
- Putting the question's candidate codes and same-question examples in the prompt.
- Seed ensembles.

This supports the user's "architecture, not size" view.

### Top 5 changes, ranked by expected gain per dollar

| # | Change | Evidence | Expected dev gain | Cost | Gate |
|---|---|---|---|---|---|
| 1 | **Per-member label-shift correction** (replace the Claude-mix prior with the gold prior) before pooling | Saerens 2002; Menon 2021; Eedi 3rd +0.036 private. **Ours, measured:** Sinkhorn on blend **+0.019 [+0.006, +0.033]**; label-shift on 8B **+0.027 [+0.011, +0.045]** | +0.02 to +0.03 (measured) | $0, CPU | Passed. Adopt now. |
| 2 | **Two-stage training of the 8B text reasoner**: Claude pseudo-labels, then 1–2 epochs on gold-only rows at low learning rate (LR) | AES 2.0 1st (+0.015 LB); CUG-KD 2605.14752 (stage 2 +0.086 on a 55-class/220-row set); sequential vs mixed training 2506.24093 (won 78% of configs); Eedi 1st | +0.005 to +0.02 *on top of #1* (est.) | ~$3–5 | Stage B beats stage A by ≥ +0.01 on dev **after** label-shift correction on both, CI lower bound > −0.005 |
| 3 | **Candidate-and-exemplar member**: 8B LoRA with the question's top-k candidate codes, each with its description and one same-question labelled example, in the prompt; listwise letter-logits scoring | MAP 1st ("Common Misconceptions" line; suffix classification), MAP 3rd (+0.002–0.004), MAP 4th (per-label few-shot block), Eedi 1st (few-shot refs **+0.036**), Novice-Learner 2310.02439 (91.9% from 4 candidates vs 39.8% from 100), MiRAGE 2511.01182 (fusing different module types 0.83/0.81 → 0.92). **Ours:** gate simulation, a top-10 member matching Claude on 25% of rows → **+0.030 [+0.010, +0.051]** | +0.01 to +0.03 (est.) | ~$4–6 | Fused gain ≥ +0.01 over the #1-corrected blend, cross-fitted, CI lower bound > 0 |
| 4 | **Seed ensemble at the final full-data refit** (2–3 seeds of the winning 8B recipe on all 1,500 gold rows) | MAP 1st (seed ensemble beat fold ensemble; judged every change on 3 seeds); AES 1st; MAP 4th refit +0.0012; MAP 6th (a weak extra seed still helped) | +0.005 to +0.015 on test (est.; not measurable on dev) | ~$3–4 per seed | None on dev; refit only the recipe that passed its gate |
| 5 | **Round-2 relabelling of the unlabelled pool**: score the 26k rows with the gold-tuned 8B on Claude's transcriptions; train on agreement-filtered mixtures of Claude and our model | AES 1st round-2 pseudo-labels; MAP 7th (50/50 label/OOF targets, kept only where top-1 agreed); LLKD 2411.08028; "LLM annotations at your own risk" 2504.15432 (teacher errors are systematic) | +0.005 to +0.015 (est.) | ~$1.5 inference + a stage-A rerun | Only after #2 passes; same gate as #2 |

Total for #2 + #3 ≈ $7–11 of the ~$17 left. Run #2 and #3 as **separate arms** so each gain can be attributed. #4 and #5 compete for what remains.

### Rules answer in one paragraph

- **Format:** CSV upload, not a code competition; no runtime limit.
- **Modifying models:** fine-tuning, LoRA/QLoRA, new heads, merging, pruning and quantisation are all allowed. The host's only conditions are that you share the weights of the final model and that the solution reproduces offline with no API.
- **Size cap:** all models together must be no larger than Nemotron 550B, about 512 GB of weights at 8-bit or below.
- **Pseudo-labels:** API pseudo-labels on train and unlabelled data are allowed for this competition only, never on test, and must be included in the winner package.
- **External data and models:** allowed if public and free (or "reasonable"); each one's own licence still applies.
- **Winner licence:** Apache 2.0, or another OSI licence with no commercial restriction. Pretrained models or data under incompatible licences are carved out, but the host advises avoiding commercially restricted models.
- **What winners deliver:** training code, inference code, environment description, documentation, weights and all pseudo-labels. Both training and inference are verified for the top 10.

Details and quotes are in §5.

---

## 2. What our own measurements already settle

These constrain which paper ideas are worth anything.

| Fact | Number | Source |
|---|---|---|
| Teacher–blend gap | 0.101 [0.063, 0.138] | `runs/arch/teacher-gap/gap.json` |
| Biggest loss bucket: top-1 in a *different section* | 55 rows, 0.139 MAP lost (Claude: 45 rows, 0.100) | same |
| Section accuracy on error rows | 32B .57, 8B .605, VL .59, blend .63, Claude .72 | `label-structure/hierarchy.json`, `family-probe/hie_probe.json` |
| If our family choice matched Claude's (oracle soft gate) | 0.772 | `design-structured/s3_teacher_gate.json` |
| Claude data scaling, 8B text reasoner | 0% → 25%: **+0.113**; 25% → 100%: +0.026 [−0.002, +0.053] | `reports/claude_scaling.json` |
| Perfect reading (gold transcription) | +0.025 to +0.035 for a single model | `reading-impact/results.json` |
| Label-shift correction | blend +0.019 [+0.006, +0.033]; 8B +0.027 [+0.011, +0.045] | `label-structure/sinkhorn_balance.json`, `research/q5-datasets/label_shift_correction_probe.json` |
| Blend top-10 recall | 0.94–0.95 | `test-combine/cpu_test.json` |
| A top-10 member matching Claude on a fraction q of rows | q=.25: 0.754; q=.5: 0.769 (on the Sinkhorn base 0.737) | `design-pragmatist/gate_sim.json` |
| TF-IDF head on *Claude's* dev reasoning text | 0.709 alone. Reasoning text maps to codes easily; producing that reasoning offline is the hard part | `design-pragmatist/reason_head_probe.json` |
| Zero-/few-shot 32B and 235B reasoning restricted to blend top-10, Sinkhorn base | −0.015 to −0.002 | `design-pragmatist/offline_reason_probe.json` |
| 235B few-shot as a 4th member | +0.016 on the raw blend; +0.008 [−0.013, +0.031] after Sinkhorn | `combiner-headroom/s4_pool.json`, `design-combine/verifier_sim.json` |
| Few-shot Opus vs zero-shot | −0.008 [−0.034, +0.017] | labelling notes |
| Claude rank hit rates on dev | rank 1 .738, rank 2 .123, rank 3 .060, miss .080 | `research/knn-probe/claude_ranks.json` |

**Already in the recipe** (so the matching paper ideas add little):
- Distilling Step-by-Step: label head plus auxiliary explanation loss at 0.3, nothing generated at inference.
- CanDist-style soft targets from Claude's ranked top 3: 0.7/0.2/0.1 in `claude_unl_v4.jsonl`, close to the measured 0.74/0.12/0.06.
- Label-Supervised-LLaMA heads.
- Confidence-weighted pseudo-labels (`v4w`).

### New CPU probes run for this report

| Probe (paper behind it) | Result | Verdict |
|---|---|---|
| **Over-correction audit** (PINK, EMNLP 2026: VLMs "fix" student errors in 42–66% of transcriptions) | Our transcriber turns a wrong final answer right on **8 of 173** rows where gold is wrong (4.6%), and the reverse on 6. Claude's dev transcriptions: 7/173 (4.0%). At most 0.011 MAP at stake. | Not our problem. Our transcriber was fine-tuned on verbatim annotator text. `overcorrection/audit.json` |
| **Within-question kNN** over 1,199 gold + 26,161 Claude-labelled transcriptions (kNN-LM; Eedi generate-retrieve-rerank) | Alone 0.42–0.44. Fused, cross-fitted: −0.006 to 0.000 on both the raw and the Sinkhorn base. | Dead in this form (char n-gram TF-IDF). The fine-tuned heads already absorb this data. `knn-probe/knn_probe.json` |
| **Family factorisation** (HiE, NeurIPS 2023): P(section) from a re-weighted log-pool of member section marginals × p(code \| section) from the blend | −0.004 [−0.009, −0.001] raw; −0.008 [−0.015, −0.002] on Sinkhorn | Dead. No member is better at family than the blend (.63). A family fix needs *new* evidence, not re-weighting. `family-probe/hie_probe.json` |

---

## 3. The ranked plan, in detail

### 1. Per-member label-shift correction ($0, adopt now)

**What.** Each member was trained on a mix in which Claude pseudo-labels dominate. Their class prior follows the natural unlabelled distribution (for example NOT-01 is 57× over-represented relative to gold), while dev and test are stratified like train. Divide each member's posterior by its training-mix prior, multiply by the gold prior raised to τ, then pool. Sinkhorn balancing of the pooled output is the transductive version.

**Evidence.**
- Saerens, Latinne & Decaestecker 2002 (EM prior re-estimation).
- Menon et al. 2021, logit adjustment (CIFAR-100-LT balanced error 61.64 → 58.24 post-hoc).
- Eedi 2024 3rd place: a prior-shift multiplier gave private 0.564 → 0.600.
- Wu & Gales 2021: calibrate the pooled ensemble, not each member, so fit the final temperature and NONE bias after pooling.

**Ours.** Measured gains are in §2: Sinkhorn λ=0.5 on the blend, and per-model τ=0.75 on the 8B, both cross-fitted with positive CIs.

**Next CPU test.** Apply the per-member correction with τ cross-fitted per member, then pool, then Sinkhorn, and report whether the two gains stack. Check on test that the EM-estimated NONE rate is close to 33.7%. Do not push the prior toward balance: Balanced Softmax and LDAM optimise balanced error, which MAP@3 does not reward.

### 2. Two-stage training: Claude pseudo-labels, then gold (~$3–5)

**What.** Keep the current 8B text-reasoner recipe as stage A (1,199 fit rows plus Claude rows, soft targets, explanation loss). Add stage B: 1–2 epochs on the 1,199 gold fit rows only, at about 0.3× the learning rate. Add a 1:1 replay of Claude rows where Claude and our model agree, to limit forgetting. Optionally give double weight to "near-miss" gold rows, where stage A ranks the gold code 2nd or 3rd or by a small margin.

**Why it should work here.**
- Gold rows are about 9.5% of the gradient steps in the current mixed run (1,199 of 12.6k).
- The annotators' root-attribution convention ("pedagogically central cause, not first error") differs from Claude's.
- Teacher errors are systematic, not random (2504.15432).

**Evidence.**

| Source | Gain |
|---|---|
| AES 2.0 1st (old/new labels under different criteria): pretrain then fine-tune | +0.015 public LB or more |
| CUG-KD (arXiv 2605.14752), stage-2 hard-row refinement | +0.009 MAP@3 on MAP-Charting; +0.086 on a 55-class/220-row algebra set (closest to our size) |
| 2506.24093 (vision): sequential beat mixing in 69 of 89 configurations, especially when synthetic data dominates | — |
| Eedi 1st: teacher pseudo-labels then refinement | +0.044 |

**Caveat.** Part of what stage B fixes is the prior, which change #1 already corrects for free. Judge stage B only after correcting both arms.

**Cost.** The 8B scale_100 run took 3.5 h. No adapter was saved locally (`runs/pod/scaling/*` holds only bundles), so stage A must be retrained. Total about 4–4.5 h including dev and test scoring: about $3 on an RTX A6000, $5 on an A100.

**Implementation.** A `--stage-b-epochs/--stage-b-lr` option in `src/k12/text_diagnose.py` (not edited here, per scope). Also save the stage-A adapter, so #5 and the seed runs can reuse it.

**Cheap add-ons in the same run, at no extra cost:**
- R-Drop and EMA (MAP 3rd, about +0.001 CV).
- Evaluate a slightly later checkpoint than the minimum-loss one (MAP 7th: +0.007 LB).
- Change the auxiliary explanation target to "root line + one-sentence error statement" instead of the full why. Key tokens carry the benefit (2406.14511).

### 3. Candidate-and-exemplar member (~$4–6)

**What.** A new Qwen3-8B LoRA whose prompt contains:
- Question, correct answer and machine transcript.
- A "Candidate codes" block of k ≈ 8–10 codes. Each has a one-line description and **one labelled example of that code on the same question**, drawn from fit-gold rows first, then Claude rows, never the row itself.
- Scoring by letter logits over the candidates.
- Test-time averaging over 2 candidate orders (Eedi 2nd: reversed-order TTA).

**Candidates.**
- At inference: blend top-10 ∪ the question's gold and Claude codes. Blend top-10 recall is 0.94–0.95. Never use a hard question mask: in our data the truth is in the fit codes for its question on only 47% of dev error rows.
- At training: the question's codes ∪ Claude top-3 ∪ same-section negatives, padded to k. During training, drop 0–2 of the exemplars at random (Eedi 1st, MAP 4th).

**Why this, and not more of the same.**
- Our family errors come from the head choosing among 76 codes with no reference to what the code looks like on *this* question. MAP 1st–7th all conditioned on per-question candidates.
- MAP 4th's per-label same-question few-shot block was its largest single gain (CV +0.0013, private +0.0012 at about 0.947).
- Eedi 1st's few-shot reference examples gave +0.036 private.
- Selecting from 4 candidates instead of about 100 lifts GPT-4 from 39.8% to 91.9% (2310.02439).
- A member built differently from the heads is what made MiRAGE's fusion work (0.83/0.81 → 0.92).

**Two warnings.**
- MAP 4th found *nearest-neighbour retrieved* examples raised CV but hurt LB. Use per-label exemplars, not kNN neighbours. Our kNN probe agrees.
- Zero- and few-shot versions of this (32B and 235B, our probes) did not help. It must be fine-tuned.

**Cost.** The prompt grows by roughly 1–1.5k tokens. Expect about 1.5–2× the 8B reasoner's time per row, roughly $4–6 on an A6000 for train, dev and test.

**Gate.** Cross-fitted fused gain ≥ +0.01 over the #1-corrected blend, CI lower bound > 0. Also report alone-in-top-10 MAP and the share of blend-wrong/Claude-right rows it fixes. The gate simulation says 25% of those rows is worth +0.03.

**Prior attempt.** Not yet validated in this form. The 10 Nov run with `--examples 3 --descriptions` (pod `jktv441e4crx35`) used 3 random same-question gold examples with no candidate structure. I found no recorded dev result for it; check whether its bundle exists before spending.

### 4. Seed ensemble at the final full-data refit (~$3–4 per seed)

**What.** For the members that go into the final test blend, refit on all 1,500 gold rows plus Claude rows, with 2–3 seeds per recipe, and average probabilities.

**Evidence.**
- MAP 1st: per-seed Qwen3-8B scored 0.9433–0.9450, the 3-seed ensemble 0.9455; the seed ensemble beat the fold ensemble.
- MAP 4th: full-data refit +0.0012 private.
- MAP 6th: a weak extra seed still helped.
- AES 1st: re-running only the fine-tune stage per seed was nearly as good as full retraining. With #2's saved stage-A adapter, extra seeds cost only stage B, about $0.5 each.

Also retrain the transcriber on all 1,500 rows (handoff note).

**Gate.** None on dev. Never refit a recipe that has not passed its own dev gate.

### 5. Round-2 relabelling of the unlabelled pool (~$1.5 plus a rerun)

**What.**
1. Score the 26k Claude-labelled unlabelled photos with the stage-B 8B, using Claude's own transcriptions as input (they exist for all 26k).
2. Build new soft targets as 0.5·Claude + 0.5·ours. Keep full weight where the top-1 codes agree, and down-weight (not drop) disagreements to 0.3.
3. Retrain stage A.

**Evidence.**
- AES 1st round-2 pseudo-labels.
- MAP 7th: 50/50 original/OOF targets, external rows kept only where OOF top-1 agreed.
- LLKD (2411.08028).
- 2504.15432: LLM-label training plateaus early; consistency filtering recovers part of the gap.

**Gate.** Only if #2 passes. Same gate as #2.

### Lower-ranked options with real evidence

**Student-aligned minimal correction as an intermediate input.**
- Srivatsa et al. 2025 (2509.01395): first-error localisation with Llama3-70B went 42.5 → 61.3 on VtG. It *collapsed* for Qwen2.5-Math, and our root is not the first error.
- It needs the corrected solution as input for all training rows, which means a 32B generation pass over about 13k rows. Too expensive now.
- A dev-only feasibility check costs about $1.5.

**Auxiliary heads for final-answer correctness and error family** (MAP 3rd: slight gain).
- Our run E (family and has-error auxiliary heads on the VL model) showed no gain.
- Gold final-answer-correct fusion is +0.014 [+0.002, +0.030], but that uses the gold flag. The CPU checker version gives +0.002.
- Only worth adding as a free auxiliary loss in #2 or #3.

**235B few-shot as a 4th member.** About $15 for +0.008 [−0.013, +0.031] once Sinkhorn is applied. Poor value.

**Synthetic rare-code transcriptions** (MAP 2nd and 7th: +0.002–0.005; MalruleLib: cross-template transfer is weak).
- 5 rare codes (LIN-01, PRO-01, PRO-02, SET-01, STA-03) have no unlabelled photos, so only synthetic data can reach them. They are about 1–2% of rows.
- This needs API spend on generated (not real) data. Ask the user before doing it.

---

## 4. What not to do, with the evidence

| Idea | Why not |
|---|---|
| Better handwriting recognition (CoMER, TAMER, PosFormer, CAN, ICAL, Uni-MuMER, MathWriting, CROHME/HME100K training) | These are single-expression LaTeX recognisers that do not transfer to multi-line working with prose, crossings-out and diagrams. Uni-MuMER needs about 20 h on 8×A800. Perfect reading is worth only +0.03, and over-correction is 4% (§2). |
| arXiv 2508.06107 ("Mask & Match", self-supervised HMER) | **Withdrawn** by the authors (v2, 28 Aug 2025: "require substantial improvement"). Handwritten expression recognition on CROHME; no numbers in the abstract; irrelevant. |
| EfficientNet + GNN formula paper (Jimin Guo) | Formula region detection plus GNN character recognition; reports 56.9% expression recognition. Older and weaker than the specialist recognisers above, which already don't transfer. |
| Process reward models (PRM800K, Math-Shepherd) and "first error step" localisers (ProcessBench) | They target the first error on olympiad solutions; ours is the pedagogically central root. Off-the-shelf PRMs generalise poorly (ProcessBench F1 31.5–56.5). |
| Self-consistency / majority vote (ProcessBench maj@8) | Our members are deterministic classification heads; nothing to sample. |
| Rationale-first generation at inference | Mixed evidence: 2406.14511 says rationale-after-label is better for small students; GenCLS++ (2504.19898) says classification works best without explicit reasoning. Our zero-shot reasoning probes lost. |
| Larger backbone at the same recipe | MAP 1st: 8B → 32B gained +0.003. Our 235B few-shot scored 0.650. A 235B fine-tune is estimated at $150–300 (scale-probe notes). |
| More random Claude labels | 25% → 100% gave +0.026 with a CI crossing 0. s1/LIMO: curated sets come close to full data. The 5 rarest codes have no unlabelled photos. |
| Question priors, Claude per-question priors, kNN, HiE family factorisation, logistic stacker, router, confidence-weighted pooling | Measured on dev, all between −0.023 and +0.002 (`runs/arch/*`, §2). |
| Balanced Softmax, LDAM, full Dirichlet calibration | The first two optimise balanced error, which is the wrong target. Dirichlet calibration has 76×76 parameters for 301 rows. |
| MathCog, MathCCS, MathAgent, VEHME | They label skills (MathCog), need student-history agents (MathCCS), give < +0.01 (MathAgent), or cost about 6 A100-days (VEHME). |

---

## 5. Rules: what the competition allows when modifying models

All quotes are from the rendered pages. Short link prefix: `C` = `https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings`.

| Question | Answer | Key quote | Source |
|---|---|---|---|
| Fine-tuning, LoRA, new heads, merging, pruning, quantising? | **Allowed.** No rule restricts how a model is changed; the host encourages fine-tuning open models. | "No specific model is required. You are free to use any AI model as long you share the weights of final model used to generate test predictions and we can reproduce it on our side" | C/discussion/747839, 748667 |
| Size limit | All models in the pipeline together must be at most about Nemotron 550B: up to about 512 GB of weights at 8-bit or below. No compute limit. | "Combined model together need to be ~512GB" / "Compute: No restrictions" | C/discussion/748667, 755632 |
| Open weights / offline | Weights must be shareable; no API in the final solution; must run without internet. Not confirmed whether base model + adapter is accepted. Safe route: merged weights, or exact base ID + adapter + one-command script. | "No external APIs allowed in final solution. Solution should be reproducible without internet access" | C/discussion/746450 |
| Winner licence | Apache 2.0 (OSI-approved, no commercial restriction). | "6. WINNER LICENSE TYPE — Open Source (Apache 2.0)" | C/rules §1.6, §2.5 |
| Third-party model/data licences | Carve-out: you don't have to relicense incompatible pretrained models or data. Any open-source *code* used must be OSI-licensed with no commercial limit. The host advises avoiding models with commercial restrictions. | "In the event that input data or pretrained models with an incompatible license are used … you do not need to grant an open source license … for that data and/or model(s)." | C/rules §2.5, Foundational §6.c; C/discussion/754757 |
| External public datasets | Allowed if public and equally accessible at no cost (or reasonable under the Reasonableness Standard). No need to post them in the forum. Each dataset's own licence applies: MathWriting is CC BY-NC-SA and HME100K is research-only, so avoid both. Eedi/MAP data carry their own competition terms. | "publicly available and equally accessible to use by all Participants … at no cost" | C/rules §2.6.a–b |
| API pseudo-labels | Allowed on unlabelled **and** labelled data, for this competition only. Never on test. Must be included in the winner package. | "You can pseudo label unlabeled as well as labeled data" / "they can't be just predictions on test set directly from Claude or OpenAI" | C/discussion/751440, 754757, 747177 |
| What winners deliver | Training code, inference code, a description of the compute environment, documentation (architecture, preprocessing, loss, hyper-parameters), model weights, and every manual or programmatic label made on the unlabelled data. Must run without human intervention. The host verifies training and inference and reproduces the top 10. A Kaggle notebook is recommended but not required. | "delivered software code should include training code, inference code, and a description of the required computational environment" / "both training & inference pipeline for winners would be requested and verified" | C/rules §2.8; C/discussion/756885, 747177 |
| Hand-labelling and data use | No hand-labelling or human prediction on test; manual transcription of test images is also banned. Data is for this competition only, and "the same applies to models trained on it". 5 submissions per day, 2 final picks. | "Hand-labelling the test set is prohibited." | C/rules §2.6A.b, Foundational 4.b; C/data |

**Risks for our pipeline:**
- Keep a provenance manifest of every Claude call (trace_id, split, model, date, prompt version) showing that no test image was sent. Ship the labelling scripts.
- Qwen3 and Qwen3-VL are Apache 2.0, so they are safe. Avoid Qwen2.5-3B/72B (Qwen research licence).
- Pin seeds, decoding settings and library versions, because the top 10 are re-run.
- Do not publish fine-tuned weights outside the winner process.
- Anthropic's own terms on training with outputs were not checked here. A narrow misconception classifier is very likely fine, but it is worth one check.

---

## 6. Literature by area (condensed)

Numbers were checked against the abstract or HTML unless marked "unverified".

### Student error and misconception diagnosis

| Paper | Finding | Transfer to us |
|---|---|---|
| MathCog, [2504.00843](https://arxiv.org/abs/2504.00843) | 18 LLMs all F1 < 0.5 on TIMSS cognitive-skill diagnosis; worse on vague evidence | Low (skills, not errors); supports "size doesn't fix it" |
| ErrorRadar, [2410.04509](https://arxiv.org/abs/2410.04509) | 5-category error classification: GPT-4o 53.1, humans 60.7; conceptual errors hardest (GPT-4o 9.2%); math-specialised models worse | Confirms family confusion is the core difficulty |
| FERMAT, [2501.07244](https://arxiv.org/abs/2501.07244) | Transcribe-then-reason helps weak VLMs a lot (Llama-90B error localisation 0.18 → 0.41); taxonomy context in the prompt raises GPT-4o detection 0.658 → 0.702 | Our two-stage design is validated; supports descriptions in the prompt |
| PINK / over-correction, [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.476) | VLMs silently fix student errors in 42–66% of transcriptions | Measured on ours: 4.6%. Closed |
| Srivatsa et al., [2509.01395](https://arxiv.org/abs/2509.01395) | Student-aligned correction lifts first-error localisation (Llama3-70B 42.5 → 61.3); hurts Qwen2.5-Math | Speculative; low priority |
| Daheim et al., [2407.09136](https://arxiv.org/abs/2407.09136) | Error description is the best intermediate representation | Supports the "root line + error statement" auxiliary target |
| ProcessBench, [2412.06559](https://arxiv.org/abs/2412.06559); PRM800K, [2305.20050](https://arxiv.org/abs/2305.20050) | PRMs generalise poorly; critic plus maj@8 helps | Wrong target; skip |
| CUG-KD, [2605.14752](https://arxiv.org/abs/2605.14752) | Qwen3-4B student 0.9585 MAP@3 beats its 72B teacher on MAP-Charting; stage-2 hard-row refinement +0.086 on a 55-class/220-row set | **Supports #2** |
| MAP host paper, [AIME-Con 2026](https://aclanthology.org/2026.aimecon-main.33) | Top teams at ≥ 0.948; multi-seed validation "most critical" | Supports #3, #4 |
| Eedi generate-retrieve-rerank, [2602.02414](https://arxiv.org/abs/2602.02414) | Fine-tuned Qwen2.5-7B beats zero-shot Claude Sonnet 4.5; reranking helps every model; zero-shot 3B rerank hurts | Supports #3 (must be fine-tuned) |
| Novice Learner, [2310.02439](https://arxiv.org/abs/2310.02439) | GPT-4 91.9% from 4 candidates vs 39.8% from about 100 | Supports #3 |
| MalAlgoPy, [2410.12294](https://arxiv.org/abs/2410.12294); MalruleLib, [2601.03217](https://arxiv.org/abs/2601.03217) | Synthetic misconception traces; weak cross-template transfer (40.5–46.5% vs 56–65%) | Synthetic data only on our own questions; low priority |

### Handwriting recognition

| Paper | Finding |
|---|---|
| TAMER, [2408.08578](https://arxiv.org/abs/2408.08578) | ExpRate on CROHME14/16/19: 61.2/60.3/62.0; HME100K 68.5–69.5 |
| PosFormer, [2407.07764](https://arxiv.org/abs/2407.07764) | +1.8 on M2E multi-line |
| Uni-MuMER, [2505.23566](https://arxiv.org/abs/2505.23566) | CROHME average 79.7; tree chain-of-thought +2.2; needs 8×A800 for 20 h |
| MathWriting, [2404.10690](https://arxiv.org/abs/2404.10690) | CC BY-NC-SA licence |

All are single-expression recognisers. Verdict: don't spend on reading.

### Kaggle solutions

**MAP 2025** ([1st](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution), [2nd](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd), [3rd](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution), [4th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951), [5th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution), [7th](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution)):
- Per-question candidates in the prompt (all top 7).
- Same-question few-shot blocks (4th).
- Seed ensembles beat fold ensembles (1st).
- Full-data refit (+0.0012).
- 50/50 label/OOF distillation with an agreement filter (7th).
- R-Drop/EMA (+0.001).
- A later checkpoint (+0.007).
- Retrieved neighbours raised CV but hurt LB (4th).

**Eedi 2024** ([1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution), [2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution), [3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost)):

| Team | Technique | Gain |
|---|---|---|
| 1st | Few-shot references | +0.036 |
| 1st | Teacher pseudo-labels on synthetic data | +0.044 |
| 1st | Claude-distilled chain-of-thought | +0.019 |
| 2nd | Reversed-order TTA | small |
| 3rd | Prior-shift multiplier | +0.036 private |

**AES 2.0** ([1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/writeups/flg-1st-place-solution-trust-cv-and-lb-a-bit)): two-stage pretrain/fine-tune on data labelled under different criteria gave +0.015 LB or more; 3-seed averaging was essential.

### Distillation

| Paper | Finding | Transfer to us |
|---|---|---|
| Distilling Step-by-Step, [2305.02301](https://arxiv.org/abs/2305.02301) | Multi-task rationale loss: ANLI 43.6 → 49.6 | Already our recipe |
| CoT-distillation mysteries, [2406.14511](https://arxiv.org/abs/2406.14511) | Rationale after the label works better; key tokens carry most of the effect | Shorten the auxiliary target |
| CanDist, [2506.03857](https://arxiv.org/abs/2506.03857) | Top-k candidate targets: TREC 81.0 → 87.8 | Mostly in place (0.7/0.2/0.1) |
| s1, [2501.19393](https://arxiv.org/abs/2501.19393); LIMO, [2502.03387](https://arxiv.org/abs/2502.03387) | Curated small sets come close to full data | Don't buy random labels |
| DeepSeek-R1, [2501.12948](https://arxiv.org/abs/2501.12948) | Supervised fine-tuning distillation beats RL for small models | — |
| 2504.15432 | LLM-label training is less stable and plateaus earlier | Supports #2, #5 |
| 2506.24093 | Sequential training beat mixing in 78% of configurations | Supports #2 |

### Classification, long tail and calibration

| Paper | Finding | Transfer to us |
|---|---|---|
| Menon logit adjustment, [2007.07314](https://arxiv.org/abs/2007.07314); Saerens 2002 | Post-hoc prior adjustment | **#1** |
| Wu & Gales, [2101.05397](https://arxiv.org/abs/2101.05397) | Calibrate after pooling | — |
| HiE, [2302.00368](https://arxiv.org/abs/2302.00368) | Coarse-classifier refinement of fine-grained predictions | Tested: −0.004 |
| kNN-LM, [1911.00172](https://arxiv.org/abs/1911.00172) | Interpolate a kNN distribution with the model | Tested: −0.006 to 0 |
| Feature-weighted linear stacking, [0911.0460](https://arxiv.org/abs/0911.0460) | Blend weights that depend on meta-features | Our stacker: +0.002 |
| Label-Supervised LLaMA, [2310.01208](https://arxiv.org/abs/2310.01208); GenCLS++, [2504.19898](https://arxiv.org/abs/2504.19898) | Classification heads on LoRA LLMs; no chain-of-thought for classification | Already our head design |
| MiRAGE, [2511.01182](https://arxiv.org/abs/2511.01182) | Retrieval 0.83 + rerank 0.81 → fused 0.92 MAP@3 on MAP data | Supports #3 |
| Calibrate Before Use, [2102.09690](https://arxiv.org/abs/2102.09690) | Content-free bias correction | Free check on the heads |

---

## 7. Reproduction (this report's probes)

All run with `PYTHONPATH=src:scripts .venv/bin/python`. Aggregate outputs only; no student text is printed.

- `runs/research/overcorrection/audit.py` → `audit.json`
- `runs/research/knn-probe/knn_probe.py` → `knn_probe.json`
- `runs/research/knn-probe/claude_ranks.py` → `claude_ranks.json`
- `runs/research/family-probe/hie_probe.py` → `hie_probe.json`
