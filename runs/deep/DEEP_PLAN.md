# K-12: Deep research on our weaknesses and the plan to fix them

_12 Nov 2026. 21 agents, CPU and free web only, no spend. Key numbers re-checked independently by the coordinator: the NONE-phrase leak (160 / 1 / 10 rows in train / dev / test), the pair-count step curve (error-row MAP 0.451 / 0.798 / 0.922 / 1.000 for 0 / 1 / 2 / 3+ fit examples of the exact question-code pair), and code IDs inside the WHY targets (about 18-22% depending on whether NONE counts)._

## 1. Executive summary

**Where we are.** Our best public MAP@3 is **0.71716** (blend + Sinkhorn); it was 0.7085 before balancing. On the 301-row dev fold the blend + Sinkhorn scores 0.737. The honest reference is **0.7337**, because the 0.737 used a Sinkhorn λ chosen in-sample. Claude Opus zero-shot scores 0.819 on the same dev rows. Spending is on hold: $22.10 RunPod and $13.39 Claude API are left if you approve them, there are 4.5 days to the deadline, and you get 4 Kaggle submissions a day.

**Where the points are (measured on dev).** The 0.10 gap to Claude, CI [0.063, 0.138], is not spread evenly:

| Row regime | Share of dev | Ours (blend+Sinkhorn) | Claude | Verdict |
|---|---|---|---|---|
| NONE rows | 34% | 0.926 | 0.934 | Fine |
| Seen pair (gold code already seen on this question in training) | 31% | **0.856** | 0.812 | We beat Claude here |
| New pair or unseen question | 35% | **0.451** | 0.715 | The whole gap is here |

On test, about **51% of error rows** [0.47, 0.56] fall in the "new pair" regime, the same as dev. Gains measured there should therefore carry over to the leaderboard at about face value: each +0.10 on new-pair rows is worth about **+0.034 public** (estimate).

**The diagnosis in one sentence.** Our models learn *which codes go with which question*, not *what each code means*. They make the decision from one vector at the end of the prompt, before any reasoning, from codes shown as bare IDs. The result is a step-shaped learning curve: error-row MAP is 0.45 with zero examples of the exact (question, code) pair, 0.80 with one, and 1.00 with three or more. Thousands of examples of the same code on *other* questions add almost nothing (per-code Spearman −0.16).

**The moves that matter, in order:**
1. **Fix the training data of the candidate student before any GPU spend (CPU, $0).** As built, it still teaches "never pick a code this question hasn't shown". The likelihood ratio for an unseen truth is 0.19 in its Claude rows against 0.49 on dev. It also leaks a NONE shortcut: the phrase "answered the question correctly" appears in 160 of 8,505 training prompts, 1 of 301 dev prompts and 10 of 3,498 test prompts. And 17.5% of its rationales name the code ID before the answer.
2. **Teach the confusions we actually make (CPU, $0).** Use mined hard negatives and sibling-contrast rows (ALG-04/03/06, WOP-01/02/ANS-02, generic vs specific) instead of random distractors.
3. **Train the candidate student once, and fuse it with Stage A rather than replacing Stage A (≈$5, needs approval).** This is the only lever with large *measured* headroom. Fusing the blend with a reason-first generator's letter gives +0.105 [0.074, 0.137] dev when Claude is the generator, and +0.048 [0.014, 0.083] even for a crude ID-free readout of Claude's reasoning. But a student only pays if its diagnoses match Claude's on more than about 60% of rows, and nobody knows yet whether an 8B gets there.
4. **A second arm only if the first one works (≈$4–5, needs approval).** Either the same recipe on a stronger open base (Qwen3.5-9B), or a second seed.
5. **Stop spending on everything else.** That covers the transcriber, symbolic checkers, post-hoc debiasing, more Claude labels, PRMs and bigger backbones. Each one was measured as null, negative or capped below about +0.007.

**Honest expected range.** The candidate student has a 25–35% chance of passing its gates. If it passes, a realistic public score is about **0.73–0.75**. If it fails, we stay at about **0.717**. The expected value is roughly +0.005 to +0.012 public. **Nothing we have measured supports 0.80+, and 0.82 is not a realistic target for this deadline.**

---

## 2. How our system works, in fundamentals

```
                       ┌───────────────────────────── STAGE A (deployed, public 0.717) ─────────────────────────────┐
 photo ──► Qwen3-VL-8B transcriber (LoRA on 1,199 gold transcripts) ──► "L1 | 2a-1 = ...\nL2 | ..."              │
                                                     │                                                             │
 question_text, key, grade, topic ───────────────────┤                                                             │
                                                     ▼                                                             │
              ┌── Qwen3-32B QLoRA:  [question | key | transcript] ──► hidden state at LAST prompt token ──► 76-way head ─► P32
              ├── Qwen3-8B  LoRA :  same prompt                      ──► hidden state at LAST prompt token ──► 76-way head ─► P8
 photo+q ─────┴── Qwen3-VL-8B    :  image + question (distilled from Claude labels) ─────────────────────► 76-way head ─► PVL
                                                     │                                                             │
                    P ∝ P32^.35 · P8^.30 · PVL^.35   (geometric blend)                                              │
                                                     │                                                             │
                    Sinkhorn λ=.5: push predicted label marginals toward the train label mix                       │
                                                     ▼                                                             │
                                                 top-3 codes                                                       │
                       └────────────────────────────────────────────────────────────────────────────────────────────┘
 STAGE B (built, not trained): Qwen3-8B LoRA sees [question | key | transcript | 10 lettered candidates, each with name +
 description + an example span] and WRITES "ROOT_LINE: Lk / WHY: ... / ANSWER: X". The letter probabilities are read
 AFTER the reasoning, averaged over 2 candidate orders, then fused geometrically with Stage A (cross-fitted weight).
```

**What each piece computes and learns**
- **The transcriber** turns pixels into text lines. It is a language model, so when the handwriting is ambiguous its decoding prior prefers mathematically coherent text. Sometimes it "repairs" the student's wrong step.
- **The text classifiers (32B, 8B)** read the prompt and take one vector, at the last prompt token, through a linear head over 76 codes (`out[:, -1]` in `src/k12/text_diagnose.py` and `src/k12/pod_train.py`). They are also trained with an explanation LM loss, but its target starts with `ROOT: CODE (name)`, so the label comes before the evidence even there. Codes are integer classes; the model never reads their descriptions.
- **The VL image classifier** does the same from the photo plus the question, trained on Claude's labels. It is the most diverse member, and it is the best one on rows where the transcript is wrong (0.325 vs 0.16–0.18).
- **The blend and Sinkhorn** multiply the three distributions, then rescale so the predicted label mix matches the training label mix. That buys +0.018 on dev and +0.009 on public.

**Why that produces our failure pattern**
- **The cheapest feature wins.** With 1,199 gold rows (median 2–3 per question), and 18–36 Claude rows per covered question in the training mixes, the quickest way to lower cross-entropy is to recognise the question and output its usual code. The question text sits verbatim at the start of every prompt. Even with it masked, the working alone identifies the question 65% of the time.
- **Nothing asks the model to reason.** The decision vector is read before any analysis is generated, so the model has no gradient pushing it to find the wrong line and then name what is wrong with it.
- **Code meaning has no way in.** With codes as bare IDs, nothing ties "ALG-04 on question 17" to "ALG-04 on question 230". So one example of the exact pair makes the model nearly right, while examples on other questions barely help.
- **The Claude labels made it worse.** In the Claude-augmented mix, the share of training error rows whose code is new for their question collapsed from 59% to 8%. Question identity now explains 35% of label entropy instead of 23%. Going from 0% to 100% Claude rows lifted the single 8B from 0.553 to 0.692, but left the ensemble flat (0.715 → 0.718), because what it mostly taught was more memorisation.

The same mechanism is also our strength. On seen pairs, memorising annotator conventions beats Claude (0.856 vs 0.812, +0.044 over 28 rows). **So the fix is to add a component that generalises, never to remove the memoriser.**

---

## 3. Our weaknesses, ranked

The "at stake" figures are dev MAP@3 points per 301 rows. They overlap, so do not add them up.

### W1. Code meaning does not transfer across questions (largest lever)
- **Mechanism.** The models learn (question, code) pairs, not codes. They find the right line but choose a generic code (CAL-01, OTH-00, VIS-01) or the wrong sibling. Sometimes they never work out the correct solution path at all.
- **Evidence.**
  - Error-row MAP by number of fit examples of the exact pair: 0.454 with none (84 rows), 0.798 with one, 0.922 with two, 1.000 with three or more. Claude is nearly flat across those bins (0.71–0.89).
  - The gold code exists on another fit question for **96.4%** of new-pair rows (median 24 examples). The data is there; the transfer is not.
  - Casebook mechanisms M3 (right line, wrong error type), M4 (confusable sibling) and M1 (never solved): gaps 0.023, 0.020 and 0.018, all CIs excluding 0. On M4 the truth is already in our top 3 on 100% of rows (we score 0.409, Claude 0.955).
  - Generic-vs-specific is the largest single error relation (31% of lost points). ALG-04 and ALG-06 have top-1 precision 0.00; WOP-02 has 0.33 against Claude's 0.86.
- **At stake.** About **0.049** (the competence residual left after a shortcut oracle). M1+M3+M4 total 0.061.
- **Typical failure.** A student multiplies exponents across different bases. The model finds the line but outputs OTH-00 instead of EXP-02.

### W2. The question prior overrides the evidence
- **Mechanism.** The prior is useful: a top-1 among the question's fit codes is right 61% of the time vs 53% otherwise. But it is applied unconditionally, so the model returns the question's usual code even when the work does not show that error (M5).
- **Evidence.**
  - On error rows of seen questions, our top-1 equals the question's modal code 37.1% of the time, against 23.6% for the truth.
  - Test, measured label-free: two students on the same question get the same top-1 from us 42.3% of the time, against 27.4% for the true labels in train (+0.149 [0.109, 0.189]).
  - M5 is 19 rows with gap 0.029 [0.014, 0.048], and it is *confidently* wrong (median p = 0.58).
  - The 8B is the strongest memoriser.
- **At stake.** Oracle removal of the shortcut is worth **+0.030**. Realistic training-time recovery is +0.005 to +0.015 (estimate).
- **Typical failure.** A student conjoins 2a−1 into 1a. We output EXP-03, because EXP-03 is what this question usually gets.

### W3. The transcriber "corrects" student mistakes on the root line
- **Mechanism.** On 21 dev misses the machine transcript loses or alters the root span. On 11 of them it actively repairs the mistake: for example 20−17=3 becomes 8+5=13, or "Jab. 11" becomes "Jawab: 5". The text models then see a correct line.
- **Evidence.**
  - The same 32B scores 0.159 on machine transcripts vs 0.460 on gold transcripts for those rows (+0.021 [0.005, 0.039]).
  - But after Sinkhorn, perfect reading for the 32B is worth only +0.012 overall.
  - A reader at Claude's fidelity is worth **+0.006** [0.0006, 0.012] (8 rows up, 1 down). With λ cross-fitted that drops to +0.0006.
- **At stake.** Ceiling +0.012 to +0.019; reachable about +0.003 to +0.006.
- **Typical failure.** The working says 25√2. The transcript says 5√2. The model predicts NONE.

### W4. No localisation of the root line before classification
- **Mechanism.** Annotators label the first error that changes the result (about 94% of the time it is the first erroneous step). A pooled head instead favours the most visible error, often a later slip.
- **Evidence.**
  - M2 is 8 rows, 0.292 vs Claude 0.812.
  - But an *oracle* that deletes every annotated secondary code from our ranking gains only **+0.0072** (5 rows change). After Sinkhorn we pick a secondary at top-1 exactly as often as Claude (6 vs 6 rows).
  - The localisation oracle (telling a classifier the gold root line) gives +0.04 to +0.05 on new-pair rows, but its CI touches 0.
- **At stake.** About 0.007 directly. The rest overlaps with W1 and is delivered by a reason-first output format.
- **Typical failure.** The root error is over-distribution in L1. We output CAL-01 for a later x=17 arithmetic slip.

### W5. The distillation data carries the wrong prior
- **Mechanism.** The Claude pool reflects natural photos. It is 71% NONE, covers only 284 questions (about 92 photos each), and its error labels are 60% generic vs 40% in gold. Its volume lands on the same questions and deepens W2.
- **Evidence.**
  - BBSE shows Claude is *not* NONE-biased on curated photos: P(NONE | gold NONE) = 0.931, P(NONE | gold error) = 0.035. So the 71% is a real shift in the label mix, and reweighting is the correct fix.
  - The v5 caps over-correct the generic share to 0.328, against 0.398 in gold.
  - 987 test rows (28%, on 210 questions) have no photo of their question in the pool at all.
- **At stake.** Not separable; it acts through W1 and W2. Reweighting is estimated at 0 to +0.01.

### W6. Candidate recall caps any reranker on exactly the rows we lose
- **Mechanism.** Stage B reranks our top 10, so wherever the blend memorised the wrong code, the truth can sit deep in the list.
- **Evidence.**
  - Candidate recall@10 is about 0.85–0.87 on new-pair rows and 0.82 on unseen-question rows, against 1.0 on seen pairs.
  - The truth is in the top 10 on only 56% of M1 misses and 74% of M5 misses.
- **At stake.** About 13 of the 106 target rows cannot be reached. That costs about 0.01–0.015 of Stage B's possible gain (estimate).

---

## 4. What is NOT our problem (stop spending here)

| Hypothesis | Evidence that it is not the bottleneck |
|---|---|
| Model size | Under the current recipe the 32B fine-tune (0.652) does not beat the 8B (0.667), and it has the worst new-pair MAP (0.349). |
| Reading quality in general | Perfect gold transcripts for the ensemble give +0.0155 before balancing and +0.012 after. 17 of the 30 gap points sit on rows read exactly. Only the corrective-error slice (W3) is real, and it is small. |
| NONE detection | 0.926 vs Claude 0.934. Blend P(NONE) has AUC 0.947, against 0.82–0.84 for symbolic or regex checks. |
| Label ambiguity / near-synonyms | 16% of our wrong top-1s are near-synonyms (Claude: 19%). The strict irreducible loss is 0.006–0.013. M7 convention rows are lost by Claude too (gap 0.002). |
| Data volume | Per-code recall does not grow with per-code count, and the codes already exist on other questions. More random Claude labels cannot reach the 210 uncovered test questions and are 71% NONE. |
| Combiners, stackers, calibration | Symbolic and TF-IDF stackers: −0.004 to +0.002. Prior division chose a=0 in 5/5 folds. Self-prior: −0.0077. kNN: −0.006 to 0. Zero-shot 235B as a 4th member: −0.002. |
| Symbolic step checking | It can check only about 40% of lines, localises at 42% (Claude about 85%), and two thirds of the gap is on codes a checker cannot see. Mal-rules fire on 8 dev rows. |
| PRMs, off-the-shelf rerankers | PRMs output step correctness with no code (and the best one is non-OSI). Qwen3-Reranker is a weaker version of Stage B with the same recall cap. |
| Masking the question text | The working alone identifies the question 65% of the time. Question dropout: +0.005 [−0.021, +0.027]. |

Approaches **killed by their own pre-declared tests** in this round:
- W1-A1, one-sentence error bottleneck plus second readout: readout increment 0.000 even with Claude as generator.
- W2-A3, factorised PoE: NONE and seen-pair rows collapse.
- W4-C, prefix-scan PRM: localisation at chance.
- W5-C, synthetic mal-rule injection: supply 10× below estimate, no gain.
- W3-A2, contrastive decoding: the targeted mechanism is rare (1 of 141 key convergences).
- Post-hoc "specific over generic" rule: −0.0056 cross-fitted.
- Appending raw cross-question exemplars to every candidate card: −0.052 on new-pair matching.

---

## 5. What research and practice offer

Each technique is matched to a weakness, with what we measured ourselves.

**1. Reason first, then choose; do not compress the rationale** (W1, W4)
- *What it is.* Generate an analysis, then a label: chain-of-thought distillation (Hsieh et al. 2023, "Distilling step-by-step"). Critic models order "locate the step, analyse, then verdict" (ThinkPRM 2504.16828; DeepCritic; ProcessBench 2412.06559).
- *Why it fits.* It moves the decision after the evidence, which directly undoes the last-token shortcut. Our root label is the first error on about 94% of rows, which matches ProcessBench's target.
- *Evidence.* CoT helps most on maths and symbolic tasks (+12.3 points on math vs about +0.7 elsewhere; Sprague et al. 2409.12183). For small students, label-first can sometimes win (2406.14511), and *Reasoning's Razor* (2510.21049) recommends fusing think-on and think-off outputs. **Ours:** Claude's reason-first letter fused with the blend gives +0.105; full ID-free reasoning gives +0.048, while one-sentence summaries give only +0.015 to +0.024 (CI includes 0). **Keep rationales full and remove code IDs from them.**

**2. Show label semantics; listwise reranking over candidates** (W1, W6)
- *What it is.* Present each candidate's name and description and pick among them, as in Eedi 2024 (retrieve 32–64 → rerank 8 → listwise 5, with LLM-enriched misconception text) and FIRST (2406.15657), which reads the ranking from the first token's logits.
- *Why it fits.* Codes become text the model can match against the working, so meaning can transfer to new questions.
- *Evidence.* Eedi teams credit enriched label text and staged reranking. **Ours:** this is the v5 student design. Caution: Eedi's "discount seen labels" trick does not port, because our codes *are* seen, just on other questions.

**3. Hard negatives from the real confusion set** (W1)
- *What it is.* Train against the distractors the model actually confuses: DPR (2004.04906), NV-Retriever's positive-aware mining, Eedi's 24 negatives per positive, contrast sets (Gardner et al. 2020), counterfactual pairs (Kaushik et al. ICLR 2020).
- *Why it fits.* A discriminative model only learns the boundaries its negatives expose. Our M3/M4 losses are sibling confusions where the truth is already a candidate.
- *Evidence.* **Ours:** co-ranking neighbours plus the question's fit codes cover **82%** of our real dev confusions, against at most 65.5% for the current v5 recipe. Description similarity alone covers only 27%.

**4. Shortcut learning and group reweighting** (W2)
- *What it is.* Networks pick the simplest predictive feature (Geirhos et al. 2020; Shah et al. 2020). The fixes are reweighting toward the under-represented group (Group DRO, Sagawa et al. 1911.08731; JTT 2107.09044; Idrissi et al. 2022 data balancing) or a product of experts with a bias model (Clark et al. 1909.03683; VQA-CP).
- *Why it fits.* "Question → usual code" is our VQA-CP-style prior.
- *Evidence.* **Ours:**
  - Debiased-focal weights leave the binary seen/unseen veto untouched (learnt shortcut strength 2.12 → 2.12).
  - Group weights on new-pair rows act on it directly. In a content-bearing proxy, **K=2** gives +0.036 on new/unseen rows. K≥3 costs seen-pair and NONE rows, and the gain flips sign when the model has little capacity.
  - Full or factorised PoE destroys NONE detection.
  - Importance weights fade when training runs to convergence (Byrd & Lipton 2019), but a one-epoch LoRA is far from that.

**5. Label-shift correction** (W5)
- *What it is.* Under a shift in the label mix, reweighting by π_target(y)/π_source(y) is unbiased (Saerens et al. 2002; BBSE, Lipton et al. 2018). Pair counts can be damped with an effective-number style factor (Cui et al. 2019).
- *Why it fits.* BBSE shows the pool's shift is in the label mix, not in Claude's labelling.
- *Evidence.* **Ours:** the v5-as-built proxy is a *harmful* fusion partner (−0.015 [−0.027, −0.004]). The prior-matched proxy is not harmful, but the proxy is too weak to show a positive fused gain, so treat this as a correctness fix.

**6. Preference optimisation over reasoning traces** (W2/M5, conditional)
- *What it is.* Iterative RPO, DPO plus an NLL term on the model's own correct vs incorrect chains (2404.19733; GSM8K 55.6 → 81.6), and GenRM (2408.15240).
- *Why it fits.* It can push down whole reasoning paths that end in the memorised code.
- *Evidence.* **Ours:** about 17% of fit rows would yield a "shortcut-rejected" pair. The degradation sweep says it cannot help unless the student's diagnoses already roughly follow Claude's. **Deferred.**

**7. Stronger open base inside the right recipe** (W1)
- *What it is.* Zero-shot skill on unseen-pair rows rises with capability: Qwen3-32B 0.397, Qwen3-235B 0.466 (CI includes 0). Qwen3.5-9B is Apache-2.0, has native vision, and reports much better maths than Qwen3 at the same size (vendor figures).
- *Why it fits.* Capability may matter once the shortcut is removed. Our 32B-vs-8B null was measured under the shortcut recipe.
- *Evidence.* Untested. That is why it goes in a second arm, not the first.

**8. Faithful reading** (W3)
- *What it is.* Distil a more literal teacher's transcripts (olmOCR 2502.18443; Kim & Rush 2016) with token weighting on the spans that matter (Rho-1, 2404.07965). VLMs over-trust fluent text priors ("Words or Vision", 2503.02199), and visual contrastive decoding targets that (2311.16922).
- *Evidence.* **Ours:** Claude's transcripts are more literal (root span present 0.685 vs 0.555), but the reachable dev gain is about +0.006. Not funded.

**What practice tells us.** Deployed graders (Gradescope, ASSISTments QuickComments) cluster same-question answers and leave the tail to humans, which is our Stage A. Published benchmarks show open VLMs far below human experts on handwritten error classification (ScratchMath 2603.24961: best open 39.0 vs humans about 80). The last 0.08 to Claude is the hard part, and an incremental gain is the realistic expectation.

---

## 6. The plan, step by step

Ordered by value per dollar and by dependency. Steps 0–2 run on CPU now. Everything new goes under `runs/deep/stageB/`; `src/`, `scripts/` and `data/v5/` are not edited.

### Step 0. Lock the reference and the fallback (CPU, now)
- **What.** Record the cross-fitted reference (0.7337), the slice references (new/unseen 0.451, seen-pair 0.856, NONE 0.926) and the label-free test Simpson index (0.423). Keep the 0.71716 submission CSV as the fallback final.
- **Why.** Dev has 301 rows; gains under about 0.005 are noise. The in-sample 0.737 is slightly optimistic, so every later comparison needs one fixed, honest yardstick.
- **How.** Write `runs/deep/stageB/reference.json` using `runs/arch/label-structure/s7_sinkhorn.py` logic with cross-fitted λ.
- **Cost/time.** $0, 30 minutes. **Approval:** none. **Gate:** reproduces 0.7337 ± 0.001. **Gain:** none (it is the yardstick).

### Step 1. Build the corrected candidate-student training set, v5b (CPU, now)
- **What.** One consolidated builder, `runs/deep/stageB/build_v5b.py`. It imports helpers from `scripts/build_candidate_dataset.py` and writes `runs/deep/stageB/data/cand_{train,dev,test}.jsonl`. Changes:

  **(a) Hygiene, no judgement involved**
  1. Strip the gold verdict phrase "[student answered the question correctly]" from transcripts in train prompts (160/8,505 rows), and from dev/test prompts (1 + 10) for consistency. *Why:* it appears on 22.7% of gold NONE transcripts but almost never in machine transcripts, so it is a NONE cue that disappears at test time.
  2. Remove code IDs and "fits X best" sentences from the WHY targets (1,486 rows, 17.5%), keeping full-length rationales. *Why:* an ID inside the rationale turns reasoning back into a hidden 76-way ID choice. Removing IDs shrank the measured transfer advantage from 0.522 to 0.480, which shows the old evidence partly rode on ID leakage. Shortening rationales lost signal, so do not shorten.
  3. Pass each gold row's annotated non-root codes as `extra` distractors. Secondary coverage rises from 0.549 to 1.0. *Why:* the model must be trained on the "reject the visible later slip" decision it faces at test time.

  **(b) Teach the real confusions (W1-A2)**
  4. Distractors: replace random same-section siblings with each truth code's confusion neighbours. Use the top 8 by Claude co-ranking plus 3× gold co-annotation (fit rows only), joined with the question's own fit codes, with at most 3 generic neighbours per row.
  5. About 1,200 short 2–3-candidate contrast rows for the sibling clusters (ALG-04/03/06, WOP-01/02/ANS-02, MEA-04/FRM-01, CAL-01/INT-04, OTH-00/VIS-01 vs topic codes). Gold rows that annotate both a generic and a specific code state the convention in the WHY: the root is the specific one (81.5% of such train rows). The 27 generic-root cases go in too, so the model learns the condition, not a blanket rule.
  6. Leave-pair-out exemplar masking: hide the truth's "On this question" example with probability p, the smallest p in {0.3, 0.5, 0.7} that brings the pooled unseen-candidate likelihood ratio into 0.45–0.6. *Why:* today the data teaches "a candidate with a same-question example is the answer".
  7. Do **not** append cross-question exemplars to the cards (measured −0.052).

  **(c) One combined weighting scheme (W2, W4, W5)**
  
  w = label-shift ratio^0.5 × 1/√(pair count) × stratified positional weight (length bucket × ANS vs rest) × 2 if the row's truth is unseen for its question (LOO), × 2 for gold rows on pool-uncovered questions. Then renormalise so the NONE share is about 0.338 and the total Claude mass is unchanged.
  - *Why each factor:* label shift fixes the generic share (0.328 vs gold 0.398). K=2 acts on the binary veto without the seen-pair cost of K≥3. Stratified positional weights correct the pool's late ROOT_LINE bias (pool +0.175 over Claude-on-dev) without cutting ANS-01 by 24% as the unstratified version did. Gold ×2 targets the 28% of test the pool cannot reach.

  **(d) Smarter candidate fill (W6)**
  
  Keep 10 slots. When the blend's top-1 probability is below 0.5, fill slots 9–10 with the specific sibling of a generic top-1 and the topic's most frequent unseen codes, instead of blend ranks 9–10. Adopt only if Step 2's recall check passes.
- **How.** `PYTHONPATH=src:scripts .venv/bin/python runs/deep/stageB/build_v5b.py`
- **Cost/time.** $0, about 5–6 hours of CPU engineering. **Approval:** none.
- **Expected gain.** Not measurable on CPU; it becomes part of Step 4. Combined effect over plain v5 is estimated at +0.005 to +0.02 dev. The pieces overlap and must not be added.

### Step 2. CPU gates on v5b, plus a CPU smoke of the trainer
- **What.** One gate script, `runs/deep/stageB/gate_v5b.py`, reusing `runs/deep/shortcut/s7_seen_cue_lr.py` and `s9_reweight_design.py` logic.
- **Why.** The proxies cannot rank recipes inside fusion: cross-fitted weight 0 for every proxy. So the only things we can check before spending are the *statistics the run will learn from*, plus whether the code runs.
- **Gates (all must pass):**
  - ≥ 0.70 of the 84 dev confusion pairs appear as distractors for that truth in ≥ 5 training rows.
  - Pooled unseen likelihood ratio between 0.45 and 0.6.
  - Kish ESS ≥ 4,000.
  - ANS-01 weighted share within ±20% of current.
  - Claude NONE share about 0.34.
  - p99 prompt ≤ 1,600 tokens.
  - Every ROOT_LINE in train targets refers to a line that exists.
  - For change (d): new/unseen recall@10 rises by ≥ 3 points and overall recall does not fall.
- **If the gates fail.** If ESS fails, drop the K=2 factor first, then temper label-shift to ^0.33. If coverage fails at k=8 neighbours, keep the hygiene and weights and drop the contrast rows.
- **Trainer smoke.** `PYTHONPATH=src:scripts .venv/bin/python -m k12.candidate_student train --device cpu --train runs/deep/stageB/data/cand_train.jsonl --predict runs/deep/stageB/data/cand_dev.jsonl --limit 40 --output runs/deep/stageB/cpu_smoke` (tiny Qwen2 path). It must parse letters and write bundles readable by `scripts/fuse_candidate.py --subset`.
- **Cost/time.** $0, 1–2 hours. **Approval:** none.

### Step 3. One approval request, then a 100-step pod smoke (≈$0.3)
- **What.** Ask you once for a capped package.
  - RunPod RTX 6000 Ada (about $1/h, Secure Cloud, ≥48 GB), **cap $11** of the $22.10.
  - Smoke $0.3 → arm 1 about $5 → arm 2 about $5, each conditional on the previous gate.
  - No Claude API spend. No Kaggle GPU (AGENTS.md: continue on CPU). Data goes over SSH; no credentials on the pod.
  - The pod is terminated after every arm, and spend is logged in `reports/runpod_budget.json`.
- **Why.** The candidate student is the only measured large headroom, and a one-epoch run is the cheapest way to find out whether an 8B can use it.
- **How.** Adapt `runs/arch/student/pod_run.sh` to the new data path: `python -m k12.candidate_student train --train .../cand_train.jsonl --limit 400 --predict .../cand_dev.jsonl ...`
- **Gate.** Loss falling, no NaNs, letter parse rate ≥ 0.95 on 50 dev rows, throughput ≥ 1.5 rows/s, valid ROOT_LINE format.
- **Gain.** None by itself; it protects the $5 run.

### Step 4. Arm 1: the corrected candidate student, full epoch (≈$4–5)
- **What.** Qwen3-8B LoRA, 1 epoch on v5b. Predict dev and test with both candidate orders. Fuse with Stage A.
- **Why.** It tests the only hypothesis that can move new-pair rows: reason before choosing, among described candidates, with the shortcut statistics corrected.
- **How.**
  - `train --train runs/deep/stageB/data/cand_train.jsonl --predict cand_dev.jsonl cand_test.jsonl --interim-predict cand_dev.jsonl --interim-at .5 --interim-limit 100 --orders 2`
  - Then on CPU: `scripts/fuse_candidate.py --student <dev bundle> --generations ... --cand-dev runs/deep/stageB/data/cand_dev.jsonl --report runs/deep/stageB/fuse_arm1.json`, and `--stage-test` only if the gates pass.
  - The student is trained on fit rows. Test exemplars use all 1,500 gold rows, as in the spec. No full-data refit: not affordable, and the fusion weights were calibrated on the fit-trained student.
- **Interim abort at 50%.** Stop the pod if the student alone on the first 100 dev rows scores new/unseen MAP < 0.35, or its parse rate is < 0.9.
- **Go/no-go gates** (SPEC, judged against the *cross-fitted* reference 0.7337):
  - (a) fused Δ ≥ +0.015 with CI lower bound > 0;
  - (b) new/unseen error rows ≥ 0.50;
  - (c) top-1 agreement with Claude ≥ 0.65 on unseen-pair rows;
  - (d) ≥ 0.45 where the blend and Claude disagree;
  - plus: fused seen-pair ≥ 0.830, NONE rows ≥ 0.915, student-predicted NONE rate 0.30–0.38, and test Simpson index lower than 0.423.
- **Cost/time.** About 1.4 h training at the measured 1.686 rows/s, plus about 1.5 h generation (estimate) → $3.5–5. Wall clock about 4 h.
- **Expected gain.** Estimate, low confidence. If it passes (25–35%): +0.015 to +0.04 dev → about +0.01 to +0.03 public. If it fails: 0, and we keep Stage A.

### Step 5. Arm 2, only if arm 1 shows signal (≈$4–5)
- **What.** It depends on what arm 1 shows:
  - **Arm 1 passes, or reaches new/unseen ≥ 0.47 without passing:** run the same v5b recipe on **Qwen3.5-9B**, after a $0.3 smoke. Smoke risks: linear-attention kernels, LoRA targets (the linear-attention projections must be added to `TARGETS` in a forked trainer under `runs/deep/stageB/`), and the transformers version. If that smoke fails within 1 hour, run arm 1's recipe on Qwen3-8B with a different seed and candidate order instead.
  - **Arm 1 is clearly negative (new/unseen < 0.45):** stop GPU spending.
- **Why.** If the recipe works, a second student is both a test of base capability (32B 0.397 vs 235B 0.466 zero-shot) and a second, partly decorrelated ensemble member. Seed and model ensembles were what helped in the MAP 2025 solutions. If the recipe does not work, a bigger base will not rescue it: the shortcut comes from supervision, not size.
- **How.** Fuse both students: P ∝ P0 · P_B1^w1 · P_B2^w2, with weights cross-fitted on a small grid.
- **Gate.** Same as Step 4. The second student must add ≥ +0.005 fused over arm 1 alone, CI lower bound ≥ 0, or it is dropped.
- **Expected gain.** +0 to +0.01 dev over arm 1 (estimate).

### Step 6. Deferred, probably skipped: one RPO round
- **What.** DPO plus NLL on traces sampled from a phase-a checkpoint. Rejected traces end in the question's usual wrong code.
- **Why it is deferred.** It needs new code (a phase-a save and a DPO loss), costs about $2.5, and only pays if the student already diagnoses like Claude on more than about 60% of rows.
- **Condition to run.** Arm 1 passes all gates, at least $6 of the cap is left, and at least 1.5 days remain. Otherwise skip. Expected about +0.002 overall.

### Step 7. Staging and submissions (CPU; each submission needs your explicit OK)
- **What.** Stage fused CSVs under `artifacts/` with sha256 and `"submitted": false`. Ask before each specific submission.
- **Why.** The rules require it, and public LB checks protect against a pipeline bug on test that dev cannot catch.
- **Gate.** The test-bundle sanity checks must pass before asking:
  - top-1 NONE share about 0.33–0.37 (the blend gives 0.354);
  - Simpson index below 0.423;
  - no missing rows;
  - the order-1 vs order-2 agreement on test similar to dev.

**What I recommend not funding at all:** transcriber retraining (W3), a multimodal Stage B, PoE losses, prefix-scan PRMs, mal-rule synthesis, more Claude labelling (keep the $13.39 unspent), any post-hoc prior rule, and Kaggle GPU jobs.

---

## 7. Timeline to 17 Nov 00:00 UTC

| When (UTC) | Work | Decision point |
|---|---|---|
| **Thu 12 Nov** | Steps 0–2 on CPU: reference, v5b build, gates, CPU smoke. Send you the approval request with exact caps. | **D0:** v5b passes the CPU gates → ask for approval. |
| **Fri 13 Nov** | Step 3 smoke, then Step 4 arm 1 (about 4 h on the pod), then fusion on CPU. | **D1** (by about 20:00): arm 1 pass / signal / fail. Stage the arm 1 CSV if it passes and ask about one submission. |
| **Sat 14 Nov** | Step 5 arm 2 if D1 allows; otherwise CPU-only consolidation. | **D2:** keep or drop arm 2. **If no GPU approval by 14 Nov 12:00, stop all GPU planning** and finalise on Stage A. |
| **Sun 15 Nov** | Fusion of everything, test sanity checks, staged CSVs; at most 1–2 permitted submissions as pipeline checks. Step 6 only if its conditions hold. | **No new training starts after 15 Nov 12:00.** |
| **Mon 16 Nov** | Choose the two finals by 18:00, leaving slack before 00:00. | Final selection. |

**Final-submission strategy: one safe, one ambitious.**
- **Safe:** the current blend + Sinkhorn (public 0.71716). *Why:* it is proven on public. Its main strength, seen pairs, carries over well to private, because test questions have more training rows per question than dev (3.66 vs 2.78).
- **Ambitious:** blend + candidate student(s), fused with cross-fitted weights, **only if** the student passed the dev gates. *Why:* its gain sits on new-pair rows, about 51% of test error rows, and those transfer at about face value. If no student passed its gates, the ambitious slot becomes the safe blend with cross-fitted λ, a near-duplicate. We do not put an unvalidated model in a final slot to "have something different".
- **Do not select by the public LB alone.** Public noise is roughly ±0.01 (estimate; the public share of test is not known to us). A fused model that is better on dev with a CI above 0 and about equal on public is preferred over the safe one. If the fused model is lower on public by more than about 0.015, treat that as a possible test-pipeline problem and investigate before selecting it.

---

## 8. Risks and how we guard against them

| Risk | Guard |
|---|---|
| **Leaderboard overfitting** (4 subs/day, small public split) | Use the public LB only as a sanity check. Select finals on cross-fitted dev with question-grouped CIs plus the label-free test checks. Never tune weights on public. |
| **Dev noise** (301 rows; per-slice n = 22–101) | Compare against the cross-fitted 0.7337. Treat gains under 0.005 as noise. Require CI lower bound > 0 for gate (a). Choose fusion weights by 5-fold question-grouped cross-fitting. Report slice results, not only the total. |
| **Shortcut reappears inside the student** | Gate on the likelihood ratio before training, Claude agreement (gates c, d) after it, and the test Simpson index as a label-free monitor. |
| **Student hurts what Stage A does well** | Mandatory geometric fusion with Stage A. Seen-pair guard ≥ 0.830 and NONE guard ≥ 0.915. Weight 0 is always allowed. |
| **Budget overrun / runaway pod** | Hard cap $11 inside the remaining $22.10. Watchdog margin from `reports/runpod_budget.json`. Interim abort at 50% of the epoch. Terminate after each arm. No API spend. |
| **One-shot engineering failure** | CPU smoke with tiny Qwen2 first, then a $0.3 pod smoke. No new loss code in arm 1: only data and weights change, and the trainer already reads `weight` and `soft`. |
| **Rules** | Final inference offline on open-licence weights (Qwen3 / Qwen3.5 Apache-2.0). Claude was used only for train and unlabelled photos, never test photos. No hand-labelling of test. No Kaggle GPU jobs (your 8 Nov instruction). Every submission needs your explicit OK. |
| **Leakage** | Keep the builder's rule that no fold-0 trace serves as an exemplar for train or dev. Keep the held-out duplicate filter. The verdict-phrase strip removes a train-only NONE cue. |
| **Time** | Training cutoff 15 Nov 12:00. Finals chosen 16 Nov 18:00. Fallback CSV staged from day 0. |

---

## 9. Open questions we still cannot answer, and the cheapest way to answer them

1. **Can an 8B student diagnose like Claude on more than about 60% of new-pair rows?** This decides everything. Cheapest answer: arm 1's interim read at 50% (100 dev rows, inside the same run), then gates (c)/(d).
2. **Does base capability help once the shortcut is removed?** The 32B-vs-8B null was under the shortcut recipe. Cheapest answer: arm 2 on Qwen3.5-9B with the same data, about $5, only after arm 1 shows signal.
3. **Is reason-first actually better than label-first for this student?** The literature is split. Cheapest answer: read the letter distribution under a forced empty rationale ("WHY: -") on the dev rows with the same adapter. That is one extra prefill, minutes of pod time, and it gives a free think-off member to fuse.
4. **How big is public LB noise?** We do not know the public share of test. Cheapest answer: none that is safe. Assume about ±0.01 and do not chase public differences smaller than that.
5. **Why did Sinkhorn transfer at only half its dev value (+0.009 vs +0.018)?** A possible cause is the test label mix differing from the train marginals Sinkhorn targets. Cheapest answer: compare the test top-1 marginal before and after balancing against train marginals on CPU. If it matters, use λ = 0.25 for the safe variant. This is a judgement call, not something we can measure.
6. **Is the test new-pair share really about 0.51?** It rests on three estimators, one validated. Cheapest monitor: the label-free test Simpson index and the prior-mode agreement of any new bundle.
7. **Which root-line accuracy metric is "the" metric?** Claude scores 0.625, 0.75 or 0.848 depending on definition. Fix one (exact gold-span containment) before reading any ROOT_LINE number from the student.
8. **The M7 convention rows (about 0.06 MAP)** cannot be learned from Claude and are partly label artifacts. No cheap answer; we accept that loss.

---

## 10. Sources

**Our measurements** (all CPU, dev fold 0, question-grouped bootstrap):
- Gap anatomy: `runs/arch/PLAN.md`; `runs/deep/casebook/{labels.csv, mechanisms.json, we_win_labels.csv}`
- Learnability, regime, coverage: `runs/deep/learnability/{p1_learning_curves, p2_ambiguity, p2b_generic_specific, p3_question_regime, p4b_strata, p5_pool_coverage, p6_projection}.json`, `per_code.csv`
- Shortcut: `runs/deep/shortcut/{s1..s9}.json`
- Symbolic: `runs/deep/symbolic/{localisation, stacker, malrules, gap_anatomy}.json`
- Literature probes: `runs/deep/literature/{p1_transfer_coverage, p1b_conflict_matched, p3_root_vs_first_error}.json`
- Open models: `runs/deep/open-models/{segments, contrasts, exemplar_cue, cost_estimate}.json`
- Approach tests:
  - `runs/deep/test-W1/w1_readout_addon2.json`
  - `runs/deep/test-W2/{w2_eval, w2_eval_matched}.json`
  - `runs/deep/test-W3/{t1_reader_ceiling, t2_claude_reader_sim}.json`
  - `runs/deep/test-W4/{t1_premise, t2_gate, t3_dev}.json`
  - `runs/deep/test-W5/{c0_ceiling, c_test, c_test2}.json`
  - `runs/deep/approach-W1..W5/`
- Student design and gates: `runs/arch/student/SPEC.md`, `src/k12/candidate_student.py`, `scripts/build_candidate_dataset.py`, `scripts/fuse_candidate.py`, `reports/cand_dataset.json`

**Literature**
- Reasoning and critics: Sprague et al. 2409.12183; 2406.14511 (CoT-augmented distillation); Reasoning's Razor 2510.21049; Hsieh et al. 2023 (Distilling step-by-step); ProcessBench 2412.06559; ThinkPRM 2504.16828; Lessons of PRMs 2501.07301; Srivatsa et al. 2509.01395; MathMistake Checker 2503.04291
- Reranking and negatives: FIRST 2406.15657; DPR 2004.04906; NV-Retriever (Moreira et al. 2024); Eedi 2024 1st/2nd place write-ups (kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics); MAP 2025 1st place write-up; Gardner et al. 2020 (contrast sets); Kaushik et al. ICLR 2020 (counterfactual data)
- Shortcuts and debiasing: Geirhos et al. 2020; Shah et al. 2020; Clark et al. 1909.03683; VQA-CP leaderboard; Sagawa et al. 1911.08731; JTT 2107.09044; Idrissi et al. 2022; Byrd & Lipton 2019; Utama et al. ACL 2020; Kirichenko et al. 2204.02937; Zhou et al. (Batch Calibration)
- Label shift: Saerens et al. 2002; Lipton et al. 2018 (BBSE); Azizzadenesheli et al. 2019; Cui et al. 2019
- Preference training: Iterative RPO 2404.19733; GenRM 2408.15240; DPO (Rafailov et al. 2023); self-consistency 2203.11171
- Reading: olmOCR 2502.18443; Kim & Rush 2016; Rho-1 2404.07965; Words or Vision 2503.02199; VCD 2311.16922
- Mal-rules: MalAlgoPy 2410.12294; MalruleLib 2601.03217
- Domain benchmarks: ScratchMath 2603.24961; MathCCS 2502.13789; ErrorRadar 2410.04509
- Models: Qwen3.5-9B, Qwen3.6-27B and Gemma-4-31B model cards (Hugging Face, vendor-reported figures)

Budget figures are from `reports/runpod_budget.json` (cap $80) and `reports/claude_budget.json` ($186.61 of $200 spent).