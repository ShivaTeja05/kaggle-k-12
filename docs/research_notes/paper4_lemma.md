# Paper 4 — LEMMA: extraction notes

Conventions: "p.N" is the PDF page. "Not stated" means not stated in the paper. Numbers marked "≈" were read off a
chart by eye; all others are transcribed from tables in the page images. Prompts are described, not reproduced.

## 1. Citation and summary

Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H. Vicky Zhao, Conghui He, Lijun Wu.
"LEMMA: Learning from Errors for MatheMatical Advancement in LLMs." arXiv:2503.17439v2 [cs.LG], 30 May 2025, 25 pages.
Code: github.com/pzs19/LEMMA.

LEMMA fine-tunes 7–8B solver models on trajectories of the form "partly wrong solution, cut at the first wrong step →
reflection phrase → correct solution". Wrong solutions come from the student model itself and from a teacher (GPT-4o)
told to inject an error of a type sampled from the student's measured error-type distribution; each is corrected two
ways ("Fix & Continue", "Fresh & Restart"). With 88.90k such trajectories LLaMA3-8B reaches 79.2 GSM8K / 38.3 MATH
against 65.5 / 19.3 for plain fine-tuning, and the largest ablation loss comes from not cutting at the first error.

## 2. Which of our bottlenecks it addresses, and how directly

| Our bottleneck | How directly | Why |
|---|---|---|
| Synthetic student errors carry half the signal and damage other rows (review 5.5) | Indirect, design guidance only | It says which errors to synthesise (measured type distribution, per question, low temperature, mixed with real ones). It never measures whether injected errors resemble real ones, and never trains a classifier on them. |
| Small model learned the form of reasoning, not the substance (5.4) | Partial | Gives a chain format anchored on the first wrong step, and shows that anchor matters most. Evidence is solver accuracy, not diagnostic accuracy. |
| Hard rows: unseen (question, code) pairs, 0.500 against Claude 0.715 | Not addressed | Its out-of-distribution tests are new questions for a solver, not new error types for a classifier. |

The mismatch must be stated plainly. LEMMA learns p(correction | the model's own half-finished wrong solution). The
wrong step is part of the model's **output**; its type is never a training label and is never verified. We need
p(code | a student's finished working): the wrong step is **input** and the type is the label.

Transfers:
- Sampling error types from a measured distribution, per question, instead of uniformly (p.4).
- The injection prompt's structure (p.23) and low sampling temperature (p.4–5).
- The chain skeleton: correct prefix, first wrong step, explicit statement of what was wrong, corrected continuation.
- Two complementary chain styles per example, and keeping only chains that end at the verified answer (p.5–6).
- Allocating more synthetic examples where the model fails, n = f × k_max (p.8).

Does not transfer:
- The "no truncation" hazard (model learns to write errors) is an output-side effect; a wrong working in our input
  cannot cause it.
- The taxonomy: 12 generic types, five of which account for essentially all observed errors, against our 75 fine codes. Types were assigned
  by GPT-4o with no human agreement check.
- The errors are language-model errors (question misreading is 35–49%), not children's.
- Scale and settings: 88.90k trajectories, 7–8B text-only non-thinking models. We have about 1,000 rows and a 27B
  thinking vision model. No result exists below ≈9k samples (Fig. 5).
- Unverified injection is harmless to LEMMA (only the correction is supervised and it is answer-checked). For us the
  injected type would be the label, so injection errors become label noise.

## 3. Exact method

### 3.1 Formulation (p.3)

$$\tau = (s_1, s_2, \ldots, s_n, \hat{a})$$

$$\tau^{b} = (s^{g}_{1}, \ldots, s^{g}_{k}, s^{b}_{1}, \ldots, s^{b}_{m}, \hat{a}^{b})$$

$$\tau^{r} = \tau^{b} \oplus RP \oplus \tau^{g}$$

$$\tau^{b}_{sub} = (s^{g}_{1}, \ldots, s^{g}_{k}, s^{b}_{1}), \qquad \tau^{r} = \tau^{b}_{sub} \oplus RP \oplus \tau^{g}$$

$s^g$ are correct steps, $s^b$ incorrect steps (the paper's sentence labels both $s^g$, a typo), $\hat a$ the predicted
answer, $RP$ the reflection phrase, $\tau^g$ a correct trajectory. The training set is $\{\tau^r\}_{i=1}^{N}$ (p.5).
There is no loss equation and no algorithm box. The only other equation is, for LEMMA-Hard (p.8):

$$n = f \times k_{max}, \qquad k_{max} = 20$$

with $f$ the student's failure rate on the question and $n$ the number of trajectories synthesised for it.

### 3.2 Error taxonomy (Table 1 p.5, Table 12 p.21)

Adapted from Li et al. (2024d); QM, CC and NO were added by the authors. Definitions paraphrased.

| Code | Type | Definition |
|---|---|---|
| CA | Calculation | wrong during computation |
| CO | Counting | wrong during counting |
| CV | Context value | an entity's attribute does not match the given information |
| HA | Hallucination | invented, unrelated statements that contradict the question |
| UC | Unit conversion | wrong during unit conversion |
| OP | Operator | one operator wrongly applied in an expression |
| FC | Formula confusion | a formula used in a setting where it does not apply |
| MS | Missing step | reasoning incomplete, a needed step absent |
| CS | Contradictory step | later steps inconsistent with earlier ones |
| QM | Question misinterpretation | question misread, e.g. a stated constraint ignored |
| CC | Confusing concept | two similar but different concepts mixed up |
| NO | Nonsensical output | illogical or irrelevant output |

Measured distribution of the first error, greedy decoding (temperature 0), classified by GPT-4o; types under 1% not
shown. The classification prompt is not given; Fig. 12 (p.22) shows its output form: a statement of where the first
wrong step arises, then the type name.

| Test set | Model | QM | FC | CA | CC | MS |
|---|---|---|---|---|---|---|
| MATH (Fig. 3, p.5) | LLaMA3-8B | 48.6 | 25.3 | 15.3 | 8.6 | 2.2 |
| | Mistral-7B | 48.8 | 24.9 | 17.2 | 6.8 | 2.3 |
| | DeepSeekMath-7B | 41.0 | 26.5 | 16.8 | 12.0 | 3.7 |
| | Qwen2-Math-7B | 40.1 | 26.3 | 18.1 | 12.9 | 2.7 |
| GSM8K (Fig. 7, p.14) | LLaMA3-8B | 41.6 | 6.3 | 24.5 | 20.9 | 6.8 |
| | Mistral-7B | 41.0 | 5.0 | 26.6 | 21.4 | 5.9 |
| | DeepSeekMath-7B | 35.4 | 8.4 | 24.7 | 22.0 | 9.4 |
| | Qwen2-Math-7B | 34.9 | 8.9 | 24.5 | 22.9 | 8.9 |

Findings: the distribution is stable across models but differs by dataset (FC is common on MATH, rare on GSM8K).
Temperature (Fig. 4a p.5, LLaMA3-8B on MATH): total error count rises from ≈125 at 0.3 to ≈130 at 0.5, ≈145 at 0.7,
≈190 at 0.9 and ≈235 at 1.1; above 0.7 the rise is mostly NO (≈0 up to 0.7, ≈30 at 0.9, ≈70 at 1.1), while the
non-NO total grows more slowly (≈125, ≈130, ≈145, ≈160, ≈165).

### 3.3 Error augmentation (p.4, prompt Fig. 13 p.23)

1. The student (LLaMA3-8B) solves training questions at temperature 0.7; its wrong solutions are kept.
2. An error-type distribution is determined per question (the paper's wording) from the error analysis. How a per-question
   distribution is estimated (samples per question, smoothing, fallback to the global one) is not stated.
3. For each injection one type is sampled from that distribution and the teacher (GPT-4o) writes a wrong solution.
4. Teacher-injected and self-generated wrong solutions form one pool. Their proportion in the main 88.90k set is not
   stated (LEMMA-Hard uses half and half). The number of injections per question is not stated.

Injection prompt structure: role of a careless student; inputs are the question and the ground-truth solution; five
bulleted instructions: (i) understand both; (ii) introduce a `{Error_Type}`, followed by `{Error_Description}`
(presumably the Table 12 definition; not stated), making it plausible rather than blatant, written as natural student
reasoning, without naming the type; (iii) keep reasoning after the wrong step to a final answer that disagrees with
the ground truth, never repairing the error; (iv) end with a fixed answer sentence and stop; (v) explain inside
`<explanation>` tags why it is wrong and why it is that type. No check that the produced error is of the requested
type is described.

### 3.4 Truncation and the two corrections (p.4–5, prompts p.24–25)

The text says the teacher annotates the first wrong step and the trajectory is cut there. In the prompts this happens
in-line: the correction prompt tells the teacher to reproduce the previous attempt up to and including the first wrong
step, then stop following it. Both prompts share role (a diligent student), five numbered instructions, and a list of
13 transition phrases to choose from, all of the kind "But, wait, let's pause and examine this more carefully."
(others: verify the calculation, something seems off, double-check, rethink, could I have missed something).

| | Fix & Continue (Fig. 14, p.24) | Fresh & Restart (Fig. 15, p.25) |
|---|---|---|
| Inputs | question, previous attempt, ground-truth solution | question, previous attempt (no ground-truth field) |
| Step 2 | name the earlier error, correct it, continue from the corrected step to an answer matching the ground truth | reflect on the earlier solution and solve by a different approach |
| Trajectory | prefix + first wrong step → transition → what was wrong → corrected step → continuation → answer | prefix + first wrong step → transition → critique → new solution from the start → answer |
| Motivation | local repair | a correct but brute-force prefix makes local repair fragile (Fig. 9, p.16) |

Shared instructions: format the final answer as the question requires; never mention the previous attempt or ground
truth, so the text reads as a student catching their own mistake; if the previous attempt is in fact correct, output
it unchanged. Each error receives each strategy once (p.6).

### 3.5 Smoothing and filtering (p.5–6)

The teacher then smooths the whole trajectory, adding logical transitions. No separate smoothing prompt, and no
definition of "model-aware" (the abstract's word), is given. Finally only trajectories with a correct final answer are
kept.

### 3.6 Pipeline figure (Fig. 1 p.1, Fig. 2 p.4)

Question → two branches: student model gives self-generated errors; teacher model gives deliberately introduced
errors → cut at first error → flawed trajectory pool → two branches: Fix & Continue revision, Fresh & Restart
revision → revised (incorrect-then-correct) trajectory pool → smoothing by the teacher → training corpus.

## 4. Hyperparameters

| Item | Value |
|---|---|
| Student for self-generated errors | LLaMA3-8B, temperature 0.7 |
| Teacher | GPT-4o; alternative LLaMA-3.1-Nemotron-70B (also used for all MetaMath questions) |
| Synthesis sampling | nucleus, temperature 0.7, top_p 1.0 |
| Source data | MATH and GSM8K training sets, 14.97k solutions; MetaMath questions for the larger variant |
| Trajectories | 88.90k (mean 623.38 tokens); with MetaMath 403.59k (544.14); LEMMA-Hard 90.7k |
| Base models fine-tuned | LLaMA3-8B, Mistral-7B-v0.1, DeepSeekMath-7B, Qwen2-Math-7B |
| Training | LLaMA-Factory; learning rate 1e-5; warmup ratio 0.03; cosine schedule; gradient accumulation 8; 3 epochs |
| LoRA or full fine-tuning | not stated |
| Batch size per device, maximum training length, optimiser, weight decay | not stated |
| Evaluation | at most 1,024 generated tokens; temperature 0 for Pass@1; 0.7 and top_p 1.0 for Majority@32 |
| Hardware | 8 × A100; about 5 hours for LLaMA3-8B |
| Ratio of error-containing to clean trajectories | not stated; all 88.90k are described as error-corrective |
| Whether the original 14.97k solutions are mixed in | not stated |

Our arithmetic: 88.90k / 14.97k ≈ 5.9 kept trajectories per question, about three errors per question at two
corrections each.

## 5. Results (accuracy %, Pass@1 unless noted)

Table 2 (p.7). Avg. covers GSM8K, MATH, ASDIV, Mathematics, MAWPS, SVAMP, College-Math.

| Method (samples) | LLaMA3-8B GSM8K / MATH / Avg. | DeepSeekMath-7B GSM8K / MATH / Avg. |
|---|---|---|
| SFT (14.97k) | 65.5 / 19.3 / 49.1 | 68.1 / 35.2 / 58.4 |
| RFT, correct self-generated only (86.52k) | 67.3 / 21.1 / 50.9 | 73.3 / 39.3 / 62.3 |
| MetaMath (394.99k) | 79.2 / 34.1 / 59.4 | 79.4 / 42.0 / 65.6 |
| GPTAug, teacher's correct solutions (88.62k) | 72.1 / 31.8 / 58.3 | 77.8 / 45.5 / 65.2 |
| ISC self-correction (86.78k) | 70.8 / 33.4 / 57.0 | 66.3 / 36.8 / 60.1 |
| S3C-Math (927k) | 82.9 / 33.1 / – | 82.5 / 41.4 / – |
| RefAug (29.94k) | 75.9 / 32.6 / 59.6 | 75.5 / 39.5 / 62.6 |
| RefAug-90k | 77.4 / 34.2 / 60.1 | 76.7 / 42.5 / 63.8 |
| LEMMA (88.90k) | 79.2 / 38.3 / 62.4 | 80.4 / 50.6 / 70.1 |
| LEMMA with MetaMath (403.59k) | 86.4 / 42.3 / 65.5 | 83.0 / 51.7 / 71.4 |

LEMMA's other LLaMA3-8B columns: ASDIV 84.2, Mathematics 39.2, MAWPS 88.8, SVAMP 82.6, College-Math 24.7.
(RefAug-90k is 89.92k in the tables, 89.82k in the text and Table 7; GPTAug is 88.62k in tables, 88.92k on p.16.)

Table 3 (p.7), LLaMA3-8B, MathChat follow-up QA rounds 1/2/3 and error correction: SFT 63.2/37.2/28.3, 66.1;
MetaMath 80.3/49.4/40.1, 66.6; ISC 78.4/46.8/39.3, 78.5; RefAug-90k 71.7/47.0/38.6, 83.8; DART-Math-Uniform
82.4/50.3/37.1, 80.6; LEMMA 83.4/49.4/43.4, 84.7. On error correction LEMMA leads RefAug-90k by only 0.9.

Table 4 (p.8), LLaMA3-8B GSM8K/MATH/Avg.: DART-Math-Hard 90.7k 81.7/42.6/62.1; LEMMA 79.2/38.3/58.7;
DART-Math-Hard 585k 81.1/46.6/63.8; LEMMA-Hard 90.7k 83.1/44.8/63.9. (In Tables 4 and 6 "Avg." is the mean of
GSM8K and MATH only, which is why LEMMA shows 58.7 here and 62.4 in the seven-set tables.)

Table 6 (p.16), LLaMA3-8B GSM8K/MATH: LEMA (errors pooled from several models) 75.1/32.8; Mistake tuning 68.9/25.5;
LLM2LLM 76.8/33.1; LEMMA 79.2/38.3.

Table 8 (p.20), teacher swapped to Nemotron-70B, Avg.: LLaMA3-8B 61.3 (GPT-4o 62.4, GPTAug 58.3); DeepSeekMath 67.9
(70.1, 65.2).

Table 9 (p.20), Avg.: Mistral-7B SFT 40.5, RefAug-90k 57.3, LEMMA 60.2 (ISC collapses to 24.3, unexplained);
Qwen2-Math-7B SFT 68.0, GPTAug 73.7, LEMMA 77.2 (GSM8K 87.4, MATH 62.9).

Table 10 (p.21), Majority@32 for LEMMA: LLaMA3-8B GSM8K 90.3, MATH 53.1 (from 38.3); DeepSeekMath MATH 63.6 (from 50.6).

Error counts after fine-tuning (Fig. 6 p.9, LLaMA3 on MATH, base / SFT / LEMMA): QM ≈1,600 / 1,590 / 1,180;
FC ≈830 / 1,160 / 690; CA ≈510 / 630 / 420. Plain fine-tuning raised some error types; LEMMA lowered all.

## 6. Ablations, failure modes, limitations

Table 5 (p.9) and Table 11 (p.21), LLaMA3-8B:

| Variant | GSM8K | MATH | Avg. (7 sets) |
|---|---|---|---|
| LEMMA | 79.2 | 38.3 | 62.4 |
| without error augmentation (self-generated errors only; size not stated) | 72.8 | 28.8 | 56.7 |
| same, regenerated to 90k | 74.3 | 32.1 | 58.6 |
| without Fresh & Restart | 75.2 | 34.4 | 59.8 |
| same, 90k | 75.7 | 34.1 | 59.9 |
| without truncation at the first error | 69.4 | 22.8 | 54.0 |

- No truncation is the worst case (−15.5 MATH) and falls below GPTAug (58.3): a model trained to write full wrong
  solutions before correcting does worse than one trained on correct solutions only.
- Removing Fix & Continue was not run. Only Fresh & Restart was removed.
- Unguided injection was not run. "Without error augmentation" removes the teacher's errors altogether, so the paper
  never isolates sampling by type from simply adding teacher-written errors. Teacher-only errors were not run either.
  The nearest evidence is indirect: LEMA's multi-model errors (32.8 MATH) and the temperature chart.
- Error-to-clean ratio: no ablation.
- Data size (Fig. 5 p.8, LLaMA3-8B, MATH): LEMMA ≈29 at ≈9k, ≈36 at ≈45k, 38.3 at 89k, still rising; RefAug ≈26 to
  ≈34. On Mathematics, ISC peaks near ≈33 and drops to ≈32. Nothing below ≈9k.
- Stated limitations (p.9): under 90k examples; possible later use with rule-based reinforcement learning.
- Unstated weaknesses: no human check of the type classifier or of injection fidelity; no significance tests or
  seeds; a showcased output (Fig. 11 example 1, p.19) derives 25√2, then 50, then answers 25, so the learned
  reflection can be cosmetic while the answer is right. That is our "form without substance" failure.

## 7. Recipe for our case

**(a) Taxonomy-conditioned synthetic wrong workings**

1. Estimate P(code | question type) from the human training rows and sample injected codes from it, never uniformly.
   *Supported by the paper* (sampling from the measured per-question distribution); using our 75 codes as the
   taxonomy is *our extrapolation*.
2. Reuse the injection prompt's structure: question, correct solution, code name plus definition, plausible rather
   than blatant, do not name the error, carry the error through to a wrong final answer, then a tagged explanation of
   why the solution is wrong and why it is that type. *Supported by the paper* (p.23). Asking the explanation to also
   name the first wrong line is *our extrapolation*; the paper's injection prompt does not ask for it.
3. Generate at temperature about 0.7, not higher. *Supported* (Fig. 4).
4. Mix with real examples rather than replace them, and give more synthetic rows to (question type, code) cells where
   Qwen3.8 fails, in the spirit of n = f × k_max. *Supported for a solver* (Tables 4, 5); *extrapolation* for a
   classifier.
5. Verify each synthetic working blind: a second model that has not seen the intended code must recover it, and the
   final answer must be wrong. *Our extrapolation*; the paper has no such check and does not need one.
6. Add synthetic correct workings so NONE keeps its real share (33.6%). *Our extrapolation* (see c).

Why the earlier catalogue may have failed, read against this paper. (i) It showed each question with about ten codes,
which flattens P(code | question); LEMMA's premise is that injected errors must follow the observed distribution, and
its critique of pooled multi-model errors is that they are errors the target would not make. That fits the −0.182 on
rows with real examples. *Supported in direction only*: the paper never ablates the sampling. (ii) It contained no
correct workings, which fits the −0.045 on no-error rows; the paper is silent. (iii) Rule-based mal-rules yielded
little; LEMMA's pipeline (self-generated and teacher-injected errors together; the split is not stated) yields about
six kept trajectories per question. *Supported* as a feasibility point. (iv) The realism gap (0.328 against 0.609) is not something this paper measures or fixes.

**(b) Diagnosis-chain template** (adapted from τ_sub^b ⊕ RP ⊕ τ^g)

1. Solve the question independently; state the correct answer (the Fresh & Restart half).
2. Check the student's lines in order; list the correct prefix; quote the first wrong line (τ_sub^b).
3. State what a correct step would have been at that line (the Fix half).
4. Name the mechanism: what rule turns the correct step into the student's step.
5. Check that this mechanism, carried forward, reproduces the student's final answer (the Continue half).
6. Map mechanism → family → code; give the two nearest codes and why they fit less; output three ranked codes.
7. If no line is wrong and the answers agree, output NONE.

Stopping at the first wrong line is *supported by the paper* (largest ablation effect) and by our own Stage B result
(0.50 with the right root line, 0.26 without). Writing two chain variants per row (repair-first and solve-first) is
*supported in form* (removing one style cost about 4 points) but *extrapolated* to diagnosis. Keeping only chains
that end at the gold code is *supported* (the paper's answer filter is the same rejection rule). Steps 4–6, and an
extra filter requiring the chain's root line to match the teacher's, are *our extrapolation*; Fig. 11 shows the
answer filter alone admits incoherent chains. The closest literal transfer is different: build trajectories from
Qwen3.8's own wrong diagnoses (cut at its first wrong inference, then reflection, then the corrected diagnosis).
*Our extrapolation*, and the truncation result warns that an uncut wrong prefix in the target is harmful.

**(c) Clean-to-erroneous ratio**

*Not stated in the paper and not ablated.* What it does show: every LEMMA training trajectory is error-containing
(bar attempts that turned out correct), and correct-only training is worse (RFT, the student's own correct
solutions, 50.9; GPTAug, the teacher's correct solutions at matched size, 58.3; against 62.4). That
supports including errors for a solver; it supports no ratio. For a classifier with a NONE class our own measurement
governs: keep the real class share. *Our extrapolation.*

## 8. Pages inspected as images

All of p.1, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25. The supplied PNGs have a
transparent background and render black, so each was flattened to a white-background JPEG before reading. Pages 2
and 10–13 (related work, references) were read as text only.

Verified against page images: 1, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25
