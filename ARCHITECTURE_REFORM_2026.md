# Reformed architecture for K-12 misconception detection

Written 14 November 2026. Design document only: nothing here has been launched, trained, spent or submitted. It builds
on `RESEARCH_SYNTHESIS_K12.md` (the eight papers) and `docs/experiment_review_2026-11-14.md` (our record). A first
draft was reviewed by six independent critics (maths, our record's numbers, the codebase, fidelity to the papers,
feasibility, completeness and leakage); this version carries their corrections.

Numbers marked **measured** come from files in this repository, scored on CPU for this document. Numbers marked
**estimate** are my judgement and have not been run.

**Terms.**

| Term | Meaning |
|---|---|
| MAP@3 | The competition metric. |
| Dev, 301 rows | The development fold. Every overall score in this file is on all 301 rows unless it says "270 rows". |
| The 270 rows | The dev rows Qwen3.8-27B answered before its time limit. The 31 it missed are all error rows (8 seen-pair, 23 hard), so the 270 rows are easier: the submitted system scores 0.775 on them and 0.755 on all 301. |
| No-error, seen-pair, hard rows | The three dev slices: true label NONE (101 rows); the true (question, code) pair appears in the fit rows (94); it does not (106). |
| Memoriser | Three fine-tuned classifiers blended, times the pointwise ranker ($M$ below). The submitted system is the memoriser after label balancing. |
| Reasoner | Qwen3.8-27B reading the photo with thinking on. |
| Sent rows | Rows whose submitted-system top probability is below 0.817. Only these go to the reasoner. 158 of 301 dev rows (52%), of which 135 have a reasoner answer today; 1,593 of 3,498 test rows (46%). |
| Fusion | The product of memoriser and reasoner on sent rows, then Sinkhorn label balancing. |
| Cross-fitted | Settings chosen on four fifths of the dev questions and scored on the other fifth. |

---

## 1. Summary

**The design in one paragraph.** Keep the memoriser exactly as submitted. Improve the reasoner in three steps, each
with its own paid dev test: show it a shortlist and one human example per shortlisted code while it reads the photo;
fine-tune it on its own gold-matching reasoning chains; scale the adapter back toward the zero-shot weights. Combine
the two systems with the existing product fusion. Keep the current zero-shot answers in the fusion throughout, so the
independent signal is added to and never replaced.

**Where this departs from the request, and why.**

| Requested | What this design does | Evidence |
|---|---|---|
| 1a. Verbatim transcription and line-by-line error diff remove the 45% missing-error gap | The reasoner's prompt already says to preserve mistakes and never correct them. v2 adds an output field for the per-line check and a second look at the root line. Treated as hygiene. | Paper 1: an explicit "do not fix" prompt moved its faithfulness score from 0.75 to 0.75. Our symbolic line checker covered 40% of lines (experiment 75). The 55% figure mixes silent fixes, dropped lines and misreads; the split is unmeasured. |
| 1b. Cropped-image verification on uncertain rows | Specified as its own arm (A5), not built in Module 1. | No evidence either way: never run. Whether Qwen3.8-27B returns usable line boxes is unverified. |
| 2. Replace soft averaging with a hard seen-versus-unseen router | The hard router is built, tested and reported, but the default stays the product fusion. | **Measured**, 270 rows: a router told the true slice scores 0.807; the fusion scores 0.804 cross-fitted and 0.814 at its in-sample setting. The three are not distinguishable. The router as requested, with test-time gates, scores 0.762 to 0.791. "Unseen pair" depends on the true label, so it cannot be observed at test time. |
| 3. Product-of-experts or focal debiasing, plus forward loss correction | Only the reweighting form is wired in, as a paired arm in the reasoner's fine-tuning. The losses and the noise matrix are specified and unit-tested but have no caller. | The memoriser is frozen and the reasoner has no 76-way head. Papers 5 and 6 measure gains only where the bias is absent. We have 301 paired labels for 76 classes. |
| 4. One human example per candidate code | Built, and tested as a paired arm. Same-question examples are a separate arm. | Paper 8 has no arm without examples. In our record, examples in the prompt raised seen-pair rows and lowered hard rows (experiments 48 and 60). |
| 5a. `N=8` gold-filtered rollouts | One rollout per fit row at the only settings validated on dev, plus rationalisation; the filter also checks the root line. One round only. | Paper 3 (STaR) found more and hotter sampling counterproductive. Our validated setting is temperature 1.0, which is itself hotter than STaR's; the root-line filter is the guard. |
| 5b. LEMMA protocol | Not used, apart from its warning about right answers with wrong reasoning. | LEMMA trains a solver, never a classifier of error types. Our two synthetic-error tests lost (experiments 95, 97). Counter-evidence I accept: in LEMMA, training on the model's own correct solutions alone (50.9) is worse than adding error-and-reflection trajectories (62.4). |
| 5c. Weight interpolation | Kept, with a fixed default $\alpha=0.5$. | Paper 2's only non-oracle rule. |

**What is achievable.**

| | Dev, 301 rows | Public (estimate) |
|---|---|---|
| Submitted system (the previous best file) | 0.755 measured | 0.73183 actual |
| Current fusion with the zero-shot reasoner | 0.781 measured, cross-fitted (+0.026 [+0.002, +0.050]) | **0.76266 actual** (submitted 14 Nov 07:21 UTC from another session, with 1,392 of the 1,593 sent test rows answered) |
| Full design | 0.78 to 0.836 estimate. Stages are adopted only if they pass, so the floor is today's fusion. | about 0.76 to 0.81 |
| Requested targets (0.95 / 0.90 / 0.72) | 0.853 | |

The fusion's dev gain of +0.026 came through as +0.031 on the public board, so dev sits 0.018 above public for this system. The estimate for the full design adds to 0.76266 its dev gain over today's fusion (0 to +0.055), scaled by 0.87 for the lower send rate on test (section 8) and by one half to all of it, the range past dev gains have kept. Reaching 0.85 public would need about 0.87 on dev, which means hard rows at 0.77 or higher with the other two slices at
their targets. Claude Opus with the photo scores 0.715 on hard rows. Nothing in the papers or our record shows a route
to that, and I do not expect this design to reach it.

**Constraints that bind this plan.** Under `AGENTS.md` nothing below can start without new explicit permission: no
pods, no paid API calls, no Kaggle submissions. RunPod spend stood at $83.65 of the $88 cap at 07:32 UTC on 14 November with one test-scoring pod still running, so about $3.50 to $4 will remain. The launcher refuses a pod when less than $2 remains and the watchdog stops pods $1 short of the cap, so at most about $3 of pod time is left. Only Module 1c could start on that, and not finish. Every stage in section 7 needs a cap increase. The GPU stages in section 7 total about $46 to $62.

---

## 2. End-to-end data flow

```
                         photo x_i, question q_i, answer key
                                       |
        +------------------------------+-------------------------------+
        |                                                              |
        v                                                              |
 [T] Transcriber (Qwen3-VL-8B LoRA)            unchanged               |
        |  machine transcript                                          |
        v                                                              |
 [M] MEMORISER                                 unchanged               |
     text 32B 0.35, text 8B 0.30, image 8B 0.35  -> blend              |
     x pointwise ranker v1 ^0.5   -> M_i  (76 probabilities)           |
        |                                                              |
        +--> Sinkhorn(0.75) = the submitted system (the 0.73183 file)  |
        |                                                              |
        v                                                              |
 [G] SEND FLAG   c_i = top probability of the submitted system         |
        |                                                              |
        |   c_i >= 0.817 --> not sent: F_i = M_i                       |
        |   (still re-balanced with the other rows at the end,         |
        |    so its top 3 can differ from the submitted file)          |
        |                                                              |
        |   c_i <  0.817   (52% of dev rows, 46% of test rows)         |
        v                                                              v
 [S] SHORTLIST   top 8 of the submitted system  U  the 3 codes     photo x_i
        |        of the zero-shot pass  U  NONE                        |
        |        (8 to 11 codes, mean 8.9; shuffled; no scores)          |
        v                                                              |
 [E] EXEMPLAR INDEX  (keyed by code; fit rows only for dev numbers)    |
        |  one human example card per shortlisted error code           |
        v                                                              v
 [R] REASONER  Qwen3.8-27B, thinking on, reads the PHOTO  <------------+
     prompt v2: shortlist + cards, per-line check field, second look
     weights: W_alpha = W_0 + alpha * (s/r) B A        [alpha = 0 is zero-shot]
        |  three ranked codes -> rank shares R_i
        |  the zero-shot answers R0_i are kept as a separate factor
        v
 [F] FUSION   F_i = norm( M_i * R0_i^(v0*s_i) * R_i^(v*s_i) )       default: product
        |     hard-router mode exists behind --mode hard (section 4.2)
        v
     Sinkhorn(lambda) toward the training label mix
        |
        v
     top 3 codes -> submission.csv          (staged only; never submitted without a yes)


 TRAINING SIDE (fit rows only; the adapter validated on dev is the one used for test)

 fit rows --> zero-shot pass (v1 prompt)  --> simulated shortlist + cards (never the row itself)
          --> one rollout each with the adopted prompt, validated settings, thinking text saved
          --> keep chain iff code = gold AND root line within one of the teacher's
          --> misses: rationalise with the gold code marked; store under the unmarked prompt
          --> arm: weight chains by the leave-one-row-out question prior, or not
          --> LoRA fine-tune Qwen3.8-27B from the base weights  --> scale adapter by alpha
```

---

## 3. Notation and the system that exists today

- $\mathcal{C}$: the 76 labels (75 codes and NONE). Row $i$ has photo $x_i$, question $q_i$, topic $t_i$, true label $y_i$.
- $\mathrm{norm}(\cdot)$ rescales a row to sum to 1.
- Blend: $B_i=\mathrm{norm}\big(\prod_{m=1}^{3}(P^{(m)}_i)^{w_m}\big)$ with $w=(0.35,\,0.30,\,0.35)$ for text 32B, text 8B, image 8B.
- Memoriser: $M_i=\mathrm{norm}\big(B_i\cdot(P^{\text{rk}}_i)^{0.5}\big)$, with $P^{\text{rk}}$ the ranker's probabilities.
- Sinkhorn balancing toward a target mix $\pi$ (`scripts/fuse_candidate.py::sinkhorn`): start with $b=0$; repeat $b\leftarrow b+\log\big(\pi\,/\,\overline{\mathrm{norm}(P\,e^{b})}\big)$ for a fixed number of iterations, the bar being the mean over rows; return $\mathrm{Sink}_\lambda(P)=\mathrm{norm}(P\,e^{\lambda b})$. On dev: 30 iterations, $\pi$ from the 1,199 fit rows. On test: 50 iterations, $\pi$ from all 1,500 rows. $\pi$ has 0.5 added to every class count.
- Submitted system: $\mathrm{Sub}_i=\mathrm{Sink}_{0.75}(M)_i$. Public 0.73183.
- Confidence and send flag: $c_i=\max_k \mathrm{Sub}_{ik}$; $s_i=\mathbb{1}[c_i<\tau]\cdot\mathbb{1}[\text{row } i \text{ has a reasoner answer}]$, $\tau=0.817$.
- Reasoner output: three ranked codes, as rank shares $S_{ik}\in\{0.6,\,0.25,\,0.15,\,0\}$ and $R_i=\mathrm{norm}(S_i+\epsilon)$.
- Current fusion (`scripts/fuse_vlm.py`): $F_i=\mathrm{norm}\big(M_i\cdot R_i^{\,v\,s_i}\big)$, then $\mathrm{Sink}_\lambda(F)$ over all rows, with $(\epsilon,v,\lambda)=(0.01,\,1,\,0.5)$ chosen on dev.

**Measured.** Source for the 270-row block: `reports/q38_fuse.json`, with the submitted system at its fixed setting.
The experiment review quotes a variant with every answered row sent and a cross-fitted base (0.772 and 0.805).

| System | Rows | All | No error | Seen pair | Hard |
|---|---|---|---|---|---|
| Submitted, fixed setting | 301 | 0.755 | 0.916 | 0.871 | 0.500 |
| Fusion, cross-fitted; unanswered rows keep the memoriser | 301 | 0.781 | 0.929 | 0.855 | 0.576 |
| Submitted, fixed setting | 270 | 0.775 | 0.916 | 0.872 | 0.502 |
| Reasoner alone, zero-shot | 270 | 0.759 | 0.924 | 0.709 | 0.608 |
| Fusion, cross-fitted | 270 | 0.804 | 0.929 | 0.853 | 0.600 |
| Per-row best of submitted and reasoner (an upper bound) | 270 | 0.874 | 0.954 | 0.924 | 0.723 |

Two facts about where the fusion's gain comes from, **measured** on the 135 sent rows:

- The two systems' top answers agree on 39% of sent rows. Almost all of the fusion's gain is on the rows where they disagree.
- By slice the gain is on hard rows. On seen-pair rows the memoriser alone (0.872) already beats the fusion (0.853).

---

## 4. The five pillars, as designed

### 4.1 Pillar 1: a front end that reads the photo and does not rewrite the student

**4.1.1 Measure what is lost (CPU, free).** Two audits, both against the teacher's annotated error span $e_i$, after
normalising formatting (the categories of paper 1, Fig. 17).

- *Transcriber*: the 200 dev error rows, using `runs/pod/tx/tx_dev.json`. These are the only out-of-fold machine transcripts; the transcriber was trained on the fit rows and no fit-row transcript exists.
- *Reasoner*: the `transcription` field of the 270 zero-shot answers already on disk. Paper 1 finds that untrained general models over-correct, and the smallest model in each family does so least. The reasoner is an untrained 27B model, so "it reads the photo" is not by itself evidence that it reads faithfully.

Each lost span gets one fate: *silently fixed* (the line is present, $e_i$ is absent, and the line now evaluates
correctly or matches the answer key); *dropped* (no line aligns to the line of $e_i$); *misread* (the line is present
and still wrong, but differs from $e_i$). Also report the two flip rates on final answers, with $t$ the teacher's final
answer, $m$ the machine's and $a^*$ the answer key:

$$\text{flip-to-right}=\frac{\#\{t\neq a^*,\ m=a^*\}}{\#\{t\neq a^*\}},\qquad \text{flip-to-wrong}=\frac{\#\{t=a^*,\ m\neq a^*\}}{\#\{t=a^*\}}.$$

Report counts as well as shares: about 90 spans are lost, so each fate share has a standard error near 5 points.
Split by line position; paper 1 finds 46.7% of events in its two late rubric components against 40% if uniform, and
whether that maps to line position is our assumption.

**Decision rule.** Compare both on the same answered dev error rows. If the reasoner keeps the annotated error text
on fewer of them than the transcriber does, direct sight is not fixing Pillar 1; A5 is then run in Module 1c beside
A1 to A3 and stays exploratory.

**Measured (`scripts/audit_transcript_faithfulness.py`, `reports/transcript_faithfulness.json`).**

| | Transcriber | Reasoner |
|---|---|---|
| Span present, the strict definition behind the recorded 55% | 111 of 200 (0.555); 0.582 on the 165 rows both cover | 0.612 on those 165 rows |
| Span kept, tolerant matching (4 rows with an empty span left out) | 143 of 196 (0.730) | 132 of 165 (0.800) |
| Paired on the 165 common rows | kept by the transcriber only: 9 | kept by the reasoner only: 20 (p = 0.061) |
| Lost spans read by hand: error corrected | 11 (5.6% of error rows) | 6 (3.6%) |
| Lost spans read by hand: changed but not corrected | 34 (17.3%) | 16 (9.7%) |
| Lost spans read by hand: same content, other formatting | 6 | 10 |
| Final answer flipped from wrong to right | 4 of 121 | 0 of 89 |

- The rule does not fire: the reasoner keeps the error text at least as often as the transcriber.
- Silent correction accounts for about 6% of error rows for the transcriber, not 45%. About a quarter of the gap under the strict definition is formatting. The larger loss is text that is changed without being corrected, on 17% of error rows.
- Lost spans are as common in the first half of a working (27%) as in the second (27%), so paper 1's late skew does not show here.
- Caveats. The automatic rules decide the fate of only 30% of lost spans, so the fate rows come from a hand reading by one reviewer, from transcripts without photos, not blind to which system wrote the page. The script has no unit tests yet.

**4.1.2 Reasoner prompt v2.** The v1 prompt (`scripts/claude_label.py::system_prompt`) already says: transcribe line
by line "preserving the student's own symbols and mistakes exactly; never correct them", check each line against the
question, the answer key and the previous line, and respect crossed-out work. So the current zero-shot run is already
paper 1's "mitigated" condition. v2 changes three things:

1. A new answer field `line_checks`, placed between `transcription` and `root_line`: for every line containing "=", the left side and right side as written, whether they are equal, and whether the line follows from the previous one. v1 asks for this check as a procedure; v2 makes it output. It has one consumer: a gate feature, "some line was flagged unequal".
2. `root_line` becomes a label, `L<n>` or `none`. Today 167 of 170 non-NONE answers already begin that way, but the format is not required. A new field `root_line_text` holds the line copied character by character.
3. A second look: before writing the answer, the model re-reads the photo at its chosen root line and corrects its transcription if the copy differs. This happens in thinking, because the answer fields are written in schema order.

Expected effect of v2 wording on its own: about zero, possibly slightly negative (paper 1's two strongest models lost
0.03 under the stronger wording).

**4.1.3 Direct sight.** The reasoner always receives the photo, never our transcript. This is how the current fusion
already works. Higher resolution is not available: **measured**, 94.7% of training photos are at most 1,024 pixels on
the long side, the size we send.

**4.1.4 Arm A5: cropped-image verification.** A second request on sent rows carrying the full photo and an enlarged
crop of the root line. The crop comes from a bounding box the model is asked for in the first request, or from
equal-height horizontal bands when the box is missing or malformed. The request asks for a character-by-character
copy of the crop and whether it matches the first transcription; a mismatch triggers one re-diagnosis. It needs the
server's image limit raised from 2 to 3 for rows with a question diagram. Untested.

### 4.2 Pillar 2: decoupled systems with a gate

**4.2.1 Fusion modes.** The gate score $g_i\in[0,1]$ is defined only for sent rows; $g_i=0$ whenever $s_i=0$.

- `product` (default). $F_i=\mathrm{norm}\big(M_i\cdot R_i^{\,v\,s_i}\big)$, then $\mathrm{Sink}_\lambda$ over all rows. With a second reasoner factor: $F_i=\mathrm{norm}\big(M_i\cdot (R^{0}_i)^{v_0 s^0_i}\cdot R_i^{\,v\,s_i}\big)$, where $R^0$ is the zero-shot answer, $s^0_i$ and $s_i$ each require an answer from their own reasoner, and a missing factor is 1. $v_0$ takes the same grid as $v$ plus 0. In both confirmatory comparisons the candidate is this two-factor form and the reference is $\mathrm{norm}\big(M_i\cdot(R^0_i)^{v s^0_i}\big)$, each with its own cross-fitted setting.
- `hard` (the router as requested). Each row takes one branch, and balancing is applied per branch:

$$\hat P_i=\begin{cases}R_i, & s_i=1\ \text{and}\ g_i\ge\tfrac12 \quad\text{(reasoner alone; the memoriser breaks ties among codes outside its three)}\\[2pt] \mathrm{Sub}_i, & \text{otherwise}\quad\text{(memoriser alone)}\end{cases}$$

- `soft`. $F_i(\beta)=\mathrm{norm}\big(M_i^{\,1-\beta g_i s_i}\cdot R_i^{\,v\,s_i}\big)$, then $\mathrm{Sink}_\lambda$. $\beta=0$ is `product`.

Before balancing, a row that was not sent is $M_i$ in `product` and `soft`, and $\mathrm{Sub}_i$ in `hard`. In `product` and `soft` the final balancing runs over all rows, so an unsent row's order can differ from the submitted file (61 of 166 unsent dev rows; the top code on 2).

**4.2.2 Gate score.** Fitted and applied on sent rows only, from test-time facts. With $\hat k^M_i,\hat k^R_i$ the top codes of the submitted system and the reasoner, and $\mathcal{C}_{\text{fit}}(q)$ the codes seen on question $q$ in the training rows:

$$z_i=\Big(c_i,\ H(M_i),\ \mathbb{1}[\hat k^M_i=\hat k^R_i],\ \mathbb{1}[\hat k^R_i\neq\text{NONE}\wedge\hat k^R_i\notin\mathcal{C}_{\text{fit}}(q_i)],\ \mathbb{1}[q_i\ \text{seen in training}],\ \mathbb{1}[\hat k^M_i=\text{NONE}],\ \mathbb{1}[\text{a line flagged unequal}]\Big),$$

$g_i=\sigma(\theta^{\top}z_i)$, with $H$ the entropy. $\theta$ is fitted by logistic regression on the label "row is
hard", cross-fitted over question folds. Its intercept is not calibrated for test: on dev $\mathcal{C}_{\text{fit}}$
comes from 1,199 rows and 13% of rows are on unseen questions; on test it comes from 1,500 rows and 4.3% are.

**4.2.3 What was measured (`scripts/eval_router.py`, `reports/router_benchmark.json`).** All 301 dev rows; rows
without a reasoner answer keep the memoriser. The last column is the paired difference from the cross-fitted product
fusion, with its 95% range.

| System | All | No error | Seen pair | Hard | Against the product fusion |
|---|---|---|---|---|---|
| Submitted | 0.755 | 0.916 | 0.871 | 0.500 | −0.026 [−0.050, −0.002] |
| Product fusion, cross-fitted (the default) | 0.781 | 0.929 | 0.855 | 0.576 | reference |
| Hard router, TRUE slice as the gate (cannot be deployed) | 0.787 | 0.916 | 0.871 | 0.590 | +0.006 [−0.017, +0.030] |
| Hard router, learned cross-fitted gate | 0.758 | 0.919 | 0.762 | 0.601 | −0.023 [−0.047, +0.001] |
| Hard router, confidence below 0.4 | 0.770 | 0.929 | 0.872 | 0.527 | −0.012 [−0.032, +0.010] |
| Hard router, confidence below 0.6 | 0.767 | 0.939 | 0.798 | 0.576 | −0.014 [−0.035, +0.007] |
| Hard router, reasoner names a new code and confidence below 0.6 | 0.765 | 0.914 | 0.801 | 0.591 | −0.016 [−0.037, +0.005] |
| Soft fusion, all settings and the gate nested inside the folds | 0.782 | 0.929 | 0.855 | 0.577 | +0.001 [−0.005, +0.007] |

- The learned gate is a reasonable detector of hard rows (cross-fitted AUC 0.816; precision 0.73, recall 0.87 on sent rows). The hard router built on it still loses, because every seen-pair row it misroutes costs more than a hard row gains: seen-pair falls from 0.855 to 0.762.
- Even with the true slice as the gate, the hard router is not distinguishable from the product fusion.
- On the 270 answered rows the same table reads 0.775 / 0.804 / 0.810 / 0.778 for submitted, product, true-slice router and learned-gate router.
- Where the fusion's gain comes from: the two systems' top answers agree on 39% of sent rows; 96% of the gain is on the rows where they disagree.

**4.2.4 Rule for leaving the default.** `hard` or `soft` replaces `product` only if it passes the rule of section 6.
On current outputs it does not.

**4.2.5 What the send flag is for.** It chooses which rows are worth reasoner compute, using $c_i$ alone. Sending
half the rows kept the whole gain in-sample.

### 4.3 Pillar 3: debiased training and noisy-teacher correction

**4.3.1 Leave-one-row-out question prior.** For fit row $i$, with $n_{q,k}$ the number of fit rows of question $q$
with label $k$ and $n_q=\sum_k n_{q,k}$; $n_{t,k}$, $n_t$ the same for the topic; $\pi^{\text{all}}$ the fit label mix
(0.5 added per class); $\lambda_b,\lambda_t>0$:

$$\pi^{(-i)}_{t_i}(k)=\frac{n_{t_i,k}-\mathbb{1}[k=y_i]+\lambda_t\,\pi^{\text{all}}(k)}{n_{t_i}-1+\lambda_t},\qquad b_i(k)=\frac{n_{q_i,k}-\mathbb{1}[k=y_i]+\lambda_b\,\pi^{(-i)}_{t_i}(k)}{n_{q_i}-1+\lambda_b}.$$

The row's own label is removed at both levels, so $0<b_i(k)<1$ for every row and class. For a row outside the fit
set: $b(k)=\big(n_{q,k}+\lambda_b\,\pi_{t}(k)\big)/\big(n_q+\lambda_b\big)$ with
$\pi_t(k)=\big(n_{t,k}+\lambda_t\pi^{\text{all}}(k)\big)/\big(n_t+\lambda_t\big)$.

Why leave-one-out matters: with the row's own label counted, the formula gives the true label a prior of at least 0.5 on a one-row question (mean 0.64 on the 112 one-row fit questions), so the row's own label, not the question, would drive its weight.

**4.3.2 Losses (papers 5 and 6).** For a model with 76-way logits and $p_i=\mathrm{softmax}(f_M(x_i))$:

- Product of experts: $\mathcal{L}^{\text{PoE}}_i=-\log\dfrac{p_i(y_i)\,b_i(y_i)}{\sum_k p_i(k)\,b_i(k)}$. At test time $f_M$ is used alone.
- Debiased focal loss: $\mathcal{L}^{\text{DFL}}_i=-\big(1-b_i(y_i)\big)^{\gamma}\log p_i(y_i)$. $\gamma=1$ is the Reweight baseline of paper 5.
- Using $f_M$ alone at test time is the "logit subtraction" of the request: $\mathrm{softmax}\big(\log(p_i\circ b_i)-\log b_i\big)=p_i$, valid because $b_i>0$.

**These two losses have no caller in this design.** The memoriser is frozen and the reasoner has no 76-way head.
They are implemented and unit-tested in `src/k12/debias.py` so that a later retrain of a memoriser member can use
them. That retrain is not proposed: the true label is among a question's training codes on about half of dev rows,
and both papers report accuracy losses where the bias still holds, from about zero for a strong encoder under product
of experts to −11 points for weak ones. Experiment 76 was a CPU proxy applied to already-trained models; the record
gives no cause for its collapse, and reading it as the over-compensation paper 5 describes is our interpretation.

**4.3.3 What is wired in: chain weights, as an arm.** For kept chain $o_i$ (thinking and answer tokens), with raw
weight $u_i=\big(1-b_i(y_i)\big)^{\gamma}$:

$$w_i=\max\big(\kappa\,u_i,\ w_{\min}\big),\quad \kappa\ \text{chosen so that}\ \tfrac{1}{|\mathcal{K}|}\textstyle\sum_{i\in\mathcal{K}}w_i=1,\qquad \mathcal{L}(\phi)=-\frac{1}{|\mathcal{K}|}\sum_{i\in\mathcal{K}}w_i\sum_{t}\log p_{\theta_0+\Delta(\phi)}\big(o_{i,t}\mid x_i,\text{prompt}_i,o_{i,<t}\big).$$

$\gamma=0$ gives $w_i=1$, which is plain STaR. With $b=0.7$ and $\gamma=1$ the raw weight is 0.30; with $b=0.01$ it is 0.99. $w_{\min}=0.2$ applies to the final weight.

The two settings pull in opposite directions, so they are run as a paired arm on the same chains:

- $\gamma=0$: the fine-tune learns the annotators' habits. Expected: seen-pair rows up for the reasoner alone, hard rows at risk.
- $\gamma=1$: rows the question prior already predicts are down-weighted. Expected from papers 5 and 6: hard rows protected or up, seen-pair and no-error rows flat or down. The floor and the fusion with the memoriser are the protection.

Rationalised chains (4.5.2) are the least trustworthy and, being misses, tend to have the largest weights. Their total weight is capped at their share of rows: if their mean weight under the common $\kappa$ exceeds 1, rationalised chains and the rest each get their own scale, chosen so that each group's mean weight is 1.

**4.3.4 Forward loss correction for Claude-labelled rows (paper 7).** $T_{ij}=p(\tilde y=j\mid y=i)$: rows are the
human label, columns Claude's. Estimated from paired rows, with Claude's answer encoded as $c_n$ (0.7 / 0.2 / 0.1 on
its three codes):

$$\hat T_{ij}=\frac{\sum_n\mathbb{1}[y_n=i]\,c_n(j)+\lambda_T\,\Pi_{ij}}{\sum_n\mathbb{1}[y_n=i]+\lambda_T},$$

$$\Pi_{ij}=a\,\mathbb{1}[i=j]+b\,\mathbb{1}[j=\text{NONE},\,i\neq\text{NONE}]+f\,\frac{\mathbb{1}[\mathrm{fam}(i)=\mathrm{fam}(j),\,i\neq j]}{|\mathrm{fam}(i)|-1}+\frac{u}{76},\qquad a+b+f+u=1,$$

with $a,b,f,u$ fitted on the pooled error-row pairs, $f$ folded into $u$ when the family has one member and $b$ into
$u$ for the NONE row. Every entry of $\hat T$ is then positive. For a classifier with prediction $\hat p$ and a
Claude soft target $\tilde q$:

$$\mathcal{L}^{\rightarrow}=-\sum_j\tilde q_j\,\log\Big(\sum_i\hat T_{ij}\,\hat p_i\Big)=-\tilde q^{\top}\log\big(\hat T^{\top}\hat p\big),\qquad \frac{\partial\mathcal{L}^{\rightarrow}}{\partial z_k}=\hat p_k-r_k,\quad r_k=\sum_j\tilde q_j\,\frac{\hat p_k\,\hat T_{kj}}{\sum_i\hat p_i\,\hat T_{ij}}.$$

Human-labelled rows use $T=I$. Order of training, following paper 7's Clothing1M result: forward-corrected training
on Claude rows first, then plain cross-entropy on human rows. Predictions use $\hat p$.

**This also has no caller.** It needs a classifier retrained with it, and a cross-fitted dev score would need one
training per dev fold. **Measured**: the only blind Claude answers on human-labelled rows are the 301 dev rows; 16 of
76 classes have five or more pairs; the largest off-diagonal cell holds 3 rows; 77% of Claude's off-diagonal mass is
outside the true code's family. Claude's NONE skew is mostly a different label mix in the pool (implied true NONE
rate 0.75), which $T$ does not address. A usable $\hat T$ needs blind Claude answers on the 1,199 fit rows, about $11,
nearly all that is left under the Claude limit. Not recommended before the reasoner stages are tested.

### 4.4 Pillar 4: examples keyed by error code

**4.4.1 Index.** For each code $k$, $E_k=\{j\in\text{fit}:y_j=k\}$. Every fit error row has a root entry with a step
and an explanation (**measured**: 794 of 794); 792 have a span; 121 root steps are ranges such as `L1-L3`. Each entry
stores the question text, the teacher transcript, the root step and span, and the explanation. No test row ever enters the index, and no dev row enters it when dev is scored. A row is never a card for itself: every lookup uses $E_k\setminus\{i\}$, in every mode. For
the final test run the index is rebuilt from all 1,500 rows.

**4.4.2 Which example (cross-question mode).** For row $i$ and shortlisted error code $k$:

$$e(i,k)=\arg\max_{j\in E_k,\ q_j\neq q_i}\Big[\cos\big(\phi(a_j),\,\bar\phi_k\big)-\mu\,\cos\big(\psi(q_j),\,\psi(q_i)\big)\Big],$$

where $a_j$ is the annotator's explanation, $\phi$ a TF-IDF vector of it, $\bar\phi_k$ the mean over $E_k$ (so the
first term prefers the most typical example of the mechanism), $\psi$ a TF-IDF vector of the question text, and
$\mu\ge0$ penalises a question that resembles the test question. Ties go to the shorter working. If the set is empty
the code gets its description only; this covers the 7 codes with no fit example and the 11 whose examples all sit on
one question.

The penalty is a precaution. Its basis is our own reading of three worked errors in paper 8, where the predicted
code's example shared the test item's story, and experiment 79. $\mu$ is fixed at 0.5 before any run. A3 with $\mu=0$ (arm A3′) is an exploratory comparison of the penalty; it picks a different example for 239 of 1,069 shortlisted codes.

**4.4.3 Same-question mode.** When $(q_i,k)$ has a fit example other than the row itself, that example is used. The
0.50 → 0.82 step in our record was observed for a fine-tuned classifier with the pair in its training rows; whether
a same-question card in a prompt reproduces it is what arm A4 tests. The record says what to expect: with
same-question examples first, Claude's seen-pair rows rose from 0.812 to 0.855 and its hard rows fell from 0.715 to
0.645 (experiment 48, **measured** by slice for this document); with four examples, Qwen3-235B gained 0.066 overall
while its hard rows fell from 0.448 to 0.407 (experiment 60).

**4.4.4 Card.** One per shortlisted error code, order shuffled per row, no scores:

```
[k] ALG-06 | <name> | <description>
Example from another student (different question):
  Question: <text>
  Working:  L1 | ...   L2 | ...
  Root line: L2 | <span, or the whole line when the span is empty>
  Why this is ALG-06: <annotator's explanation>
```

NONE gets one correct working from another question in the same topic, with the line "no major error". The prompt
also lists the remaining code names and says the answer may be a code outside the shortlist. Ten cards add about
1,400 to 2,600 prompt tokens to the present 5,400, inside the server's 32,768-token limit.

**4.4.5 Shortlist.** $\mathcal{S}_i=\text{top-8}(\mathrm{Sub}_i)\cup\{\text{the three codes of the zero-shot pass}\}\cup\{\text{NONE}\}$.
**Measured** recall of the true code on hard rows: 0.849 for the top 8 alone (106 rows); on the 83 answered hard rows
0.855 for the top 8 and 0.904 for the union; on the 68 of those that are sent, where the shortlist is actually used,
0.88. So about 12% of sent hard rows cannot be helped by the list.

**4.4.6 Arms.** Same model, same settings, same rows.

| Arm | Prompt | Purpose |
|---|---|---|
| A0 | v1, full taxonomy (already run on 270 rows) | Reference |
| A1 | v2: `line_checks` field and second look, full taxonomy | Explains A3; not judged on its own |
| A2 | A1 + shortlist, descriptions only | Explains A3 |
| A3 | A2 + cross-question cards | **The confirmatory comparison, against A0** |
| A3′ | A3 with $\mu=0$ | Exploratory: tests the similarity penalty |
| A4 | A3 with same-question cards where they exist | Exploratory; adopted only if hard rows on sent rows are not below A3's |
| A5 | A1 + cropped root-line second request | Exploratory |

### 4.5 Pillar 5: reasoning fine-tuning and weight interpolation

**4.5.0 Shortlists and cards for fit rows.** The memoriser was trained on the fit rows, so its predictions there are
in-sample and would put the gold code in the shortlist almost always. No fit-row memoriser output exists, and
out-of-fold output would need four more trainings of three classifiers and the ranker. Instead the fit-row shortlist
is simulated to match dev:

- the row's three zero-shot codes (from a v1 pass over the fit rows, which is also needed as the reference in 4.5.8);
- codes seen on the question in the other fit rows, and same-family distractors, up to 8 to 10 codes, shuffled;
- the gold code is included at the dev-measured rate for the row's type: 0.88 for error rows whose pair appears in no other fit row (the leave-one-row-out equivalent of a hard row), and always otherwise (67 of 67 sent dev rows).

In same-question mode, gold cards are dropped at random until the share of gold codes with a same-question card
equals the share of distractor codes with one, within 0.05. This is the balancing the Stage B candidate lists used in training, where before balancing the truth carried an example on 0.917 of rows against 0.624 for distractors. As there, the cue stays informative at dev and test time.

If the adopted prompt is A1 (no shortlist), none of this is needed.

**4.5.1 Chains.** For each fit row, one rollout from the untrained model with the adopted prompt at the validated
settings: thinking on, effort medium, temperature 1.0, top-p 0.95, top-k 20. Every dev number for the reasoner was
measured there. This is not low temperature in STaR's sense. STaR found temperatures 0.5 and 0.7 worse than rationale
generation alone, on a model with 5-way labels and a 20% chance of a lucky pass; with 76 labels a blind pass is about
1.3% per rollout, so the mechanism is weaker here but untested. A lower-temperature rollout is a separate option that
would need its own dev pass first.

Keep chain $o_i$ if and only if, with the teacher's root step $[a_i,b_i]$ ($a_i=b_i$ for a single line) and the
model's root-line label $\hat n_i$:

$$\mathbb{1}\big[\hat y_i=y_i\big]\cdot\mathbb{1}\big[\,y_i=\text{NONE}\ \ \text{or}\ \ a_i-1\le\hat n_i\le b_i+1\ \ \text{or}\ \ \mathrm{sim}\big(\text{span}_i,\ \widehat{\text{line text}}_i\big)\ge\tau_s\big]=1 .$$

The first factor is STaR's filter. The second is ours. The text-similarity alternative is needed because the model
numbers its own transcription, whose line count equals the teacher's on only 58% of rows. The filter answers the
failure STaR reports, where the answer is right and the reasoning is not, and our own Stage B result, where accuracy
was 0.50 when the root line was right and 0.26 when it was wrong.

$\mathrm{sim}$ is the token-set Jaccard similarity between the teacher's span and the model's `root_line_text` after formatting normalisation, with $\tau_s=0.6$ fixed in advance; it is false when the span is empty. The line test is false when the model's root line is not an `L<n>` label.

**Measured** on the 169 answered dev error rows: the code is right on 58%; code and root line both pass on 51% by the line test alone. On no-error rows the code is right on 91%. That is about 400 error chains and 370 no-error chains from the fit rows. The 34% cap of 4.5.3 keeps about 210 of the no-error chains, so about 610 chains are trained on before rationalisation, and about 720 if rationalisation lifts error chains to the 60% target.

**4.5.2 Rationalisation for misses (STaR).** Re-run the row with the gold code marked `(CORRECT)` in the shortlist.
Keep the chain only if it then answers the gold code and passes the root-line check. Store it under the prompt
without the marker. Drop any chain that mentions the marker. The paper's own caveats: rationalisation added 3.7
points on one task (confounded) and 0.6 on another; rationalised chains can be template matching; and it says bias
amplification is "made worse by rationalization". So the fine-tune is also run without rationalised chains, as a third run, and their weight is capped (4.3.3).

**4.5.3 Balance.** At most one chain per row. No-error chains are capped at the training share of NONE (34%).

**4.5.4 Training format.** Each example is rendered with the model's own chat template and the serving settings
(thinking enabled, effort medium, the same image limits). The assistant turn is the template's thinking block
holding the saved reasoning text, followed by the JSON answer. The client today stores only the length of the
reasoning text, so a `--save-reasoning` flag is required before any rollout. Two checks: the rendered prompt has the
token count the server reported for that row; and the untrained model, given the first 200 chain tokens, continues
without re-opening a thinking block. The loss is 4.3.3. STaR's gradient is the case $w_i=1$:

$$\nabla J=\sum_i\mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\big[\mathbb{1}(\hat y_i=y_i)\cdot\nabla\log p_M(\hat y_i,\hat r_i\mid x_i)\big].$$

**4.5.5 Adapter.** LoRA on the language model's attention and MLP projections, vision tower frozen:
$W=W_0+\tfrac{s}{r}BA$, $B$ initialised to zero so that training starts at the zero-shot model. Proposed settings,
none taken from a paper for this model class: rank 16, AdamW, learning rate $1\times10^{-4}$, cosine schedule with 3%
warm-up, weight decay 0.1, gradient clipping at 1, 2 epochs. One round, restarting from the base weights. STaR used
16 to 46 rounds and reports no single-round result on its two language tasks.

**4.5.6 Interpolation (WiSE-FT).** For LoRA, interpolating the weights is scaling the update, done by multiplying
the $B$ matrices by $\alpha$:

$$\theta_\alpha=(1-\alpha)\,\theta_0+\alpha\,\theta_1\ \Longleftrightarrow\ W_\alpha=W_0+\alpha\,\tfrac{s}{r}BA .$$

This identity holds for any served base with an unmerged adapter. What a base mismatch breaks is the $\alpha=1$
endpoint: an adapter trained on 4-bit weights and served on the FP8 weights is not the model that was trained.
Checking at $\alpha=0$ tests nothing, since that is the zero-shot model by construction. So:

- train on the full-precision base on an 80 GB card, within the approved $1.79 per hour, so that FP8 serving is a near match; or
- if training on 4-bit weights, verify at $\alpha=1$: decode 30 dev rows greedily with the adapter on the training base and on the served base, and require the same first code on at least 90%.

Before any training spend, a 15-minute check (about $0.30): load a dummy rank-16 adapter in the server on the FP8
weights and answer four rows. Serving an adapter for this model is untested here.

**4.5.7 Choosing $\alpha$.** Default $\alpha=0.5$, the only non-oracle rule in paper 2. Another value from
$\{0.25,\,0.75,\,1\}$ replaces it only if it beats 0.5 on a paired bootstrap of the fused score, with the choice made
inside the question folds. Paper 2 found the optimum under shift at a smaller $\alpha$ (0.35) than in distribution
(0.60 to 0.70). Per-slice monitor:

$$\mathrm{MAP@3}_g(\theta_\alpha)\ \ge\ (1-\alpha)\,\mathrm{MAP@3}_g(\theta_0)+\alpha\,\mathrm{MAP@3}_g(\theta_1).$$

With 68 sent hard rows the slice figures are noisy (paired standard error about 0.027), so this is a monitor and not
a pass mark.

**4.5.8 The comparison that matters.** The reference is the fusion with the zero-shot reasoner under the adopted
prompt, not the zero-shot reasoner alone. Paper 2's Table 13 suggests the extra gain may be small: output fusion with
an independent model came within about one point of weight interpolation. An arm whose reasoner-alone score rises
while its fused score does not is not adopted.

**4.5.9 Which adapter scores the test set.** The one trained on the 1,199 fit rows and validated on dev. It is not
retrained on all 1,500 rows, because $\alpha$ and the arm would then carry over unvalidated.

---

## 5. File-by-file map

No existing file changes behaviour by default. Every new path is behind a flag whose default reproduces today's
output.

**Step 0, before any existing file is edited (needs your go-ahead, since I commit only when asked).** The Module 1 files written so far add files only and change no existing one; the reproduction test already passes. Several files on the 0.73183 path
are untracked or carry uncommitted edits (`scripts/fuse_candidate.py`, `scripts/fuse_vlm.py`,
`scripts/build_vlm_requests.py`, `src/k12/pointwise_ranker.py` untracked; `src/k12/pod_train.py`,
`src/k12/text_diagnose.py` modified), and the prediction bundles are git-ignored. Commit the working tree, tag it
`submitted-0.73183`, and record a manifest of sha256 hashes for the four test bundles and the submission file.

| File | Status | What changes |
|---|---|---|
| `src/k12/slices.py` | new | `slice_masks`, a copy of `fuse_candidate.unseen_pairs`, and `slice_report`: overall, no-error, seen-pair, hard, with row counts. A test asserts the copy and the original give identical masks. |
| `src/k12/router.py` | new | Gate features, cross-fitted gate, and `fuse(M, R, sent, mode, ...)` for the three modes of 4.2.1. `mode='product'` equals the `fuse_vlm.py` formula. |
| `src/k12/exemplars.py` | new | Code-keyed index and card rendering. Starts from `Exemplars`, `root_line` and `why` in `scripts/build_candidate_dataset.py` and `few_shot` in `scripts/claude_label.py`, moved rather than rewritten. |
| `src/k12/prompts.py` | new | v2 system prompt, the key-list sentence and the answer schema with `line_checks` and `root_line_text`. v1 stays where it is. |
| `src/k12/debias.py` | new | Leave-one-row-out prior, chain weights. PoE and DFL losses: library and tests only, no caller. |
| `src/k12/noise.py` | new | $\hat T$ estimator and forward loss: library and tests only, no caller. |
| `src/k12/rft.py` | new | Chain filter, rationalisation prompt, NONE cap, simulated fit-row shortlists, training-file writer. |
| `scripts/build_shortlist.py` | new | Reads the memoriser bundle and the zero-shot answers; writes the shortlist and the list of sent rows for dev or test. Replaces the hand-made `ids_least_confident.json`. |
| `scripts/eval_router.py` | new | The three fusion modes on 301 and 270 rows, cross-fitted, four-slice report, bootstrap ranges. CPU. |
| `scripts/audit_transcript_faithfulness.py` | new | Both audits of 4.1.1. CPU. |
| `scripts/build_vlm_requests.py` | flags added | `--prompt v1|v2` selects the system prompt, the key-list sentence and the schema together; `--shortlist FILE`; `--cards none|cross|same`; `--mu`. Until you confirm, the same function lives in a separate new script, `scripts/build_vlm_requests_v2.py`. Default writes the same bytes as today (sha256 of the dev request file `c1ae4272…b36d`). |
| `scripts/fuse_vlm.py` | flags added | `--mode product|hard|soft`; `--rows FILE` for the evaluation set; `--zero-shot FILE` for the second factor; `--merged-out` and `--selected-ids` in place of the hard-coded Qwen3.8 paths. Prints the four-slice report on 301 and 270 rows. |
| `scripts/score_vlm_dev.py` | small | Uses `k12.slices` (same numbers, including its hard A/B/C groups). |
| `scripts/vlm_zero_shot_client.py` | flags added | `--save-reasoning`; `--temperature`, `--top-p` (hard-coded today); `--model` already exists for a named adapter. Defaults unchanged. |
| `scripts/build_rft_dataset.py` | new | Rollouts → kept chains → weighted training file. |
| `scripts/train_reasoner_lora.py`, `scripts/scale_adapter.py` | new | Pod-side fine-tune; adapter scaled by $\alpha$. |
| `scripts/pod_run_arms.sh`, `pod_run_rollouts.sh`, `pod_serve_adapter.sh` | new | Tracked launch scripts with pinned versions (vLLM 0.33.0, transformers 5.20.2), FP8 only, thinking pass only, adapter serving. The existing scripts under `runs/deep/q38/` stay as they are. |
| `scripts/fuse_candidate.py`, `scripts/evaluate_claude.py`, `src/k12/core.py`, `pointwise_ranker.py`, `text_diagnose.py`, `pod_train.py`, `pod_predict.py`, `vision.py`, `transcriber.py` | untouched | The memoriser and the 0.73183 path. The submission is rebuilt from saved bundles, not by re-running these. |
| `tests/test_slices.py`, `test_router.py`, `test_exemplars.py`, `test_prompts.py`, `test_debias.py`, `test_noise.py`, `test_rft.py` | new | Section 6. Tests that need `scripts/` add it to the path; tests that need git-ignored bundles skip when they are absent. |
| `tests/test_reproduce_submission.py` | new | Rebuilds `fuse(geo(TEST), ranker_test, w=0.5, lam=0.75, target=marginals(all 1,500 train rows), iters=50)` from `runs/pod/{text32-full,text-full,full-distill}/test.npz` and the ranker's test bundle, writes it with `write_submission`, and asserts the CSV sha256 is `5354bd61…570f`. **Measured**: this reproduces the file byte for byte; 30 iterations or fit-row marginals do not. |

---

## 6. Tests, reporting and decision rules

**Four-slice report.** Every evaluation prints overall, no-error, seen-pair and hard MAP@3 with row counts, on all
301 rows (rows without a reasoner answer keep the memoriser) and on the 270 rows, plus the paired bootstrap range
over questions against a named reference. Every staged test file also records the send rate and median confidence
next to the dev values.

**Decision rules.**

- *Two confirmatory comparisons, in this order.* (1) A3 against A0. (2) The fine-tuned reasoner at $\alpha=0.5$ against the adopted zero-shot prompt. Everything else is exploratory and cannot be adopted on its own dev score.
- *Pass mark.* Fused, cross-fitted, all 301 rows: observed gain at least +0.025 with the 95% range above zero. A range above zero needs about that much, because the paired standard error is 0.012 to 0.014. Slice conditions: the hard-row range is not wholly below zero; the no-error change is not below −0.01.
- *Nested selection.* Any choice among arms, card modes, $\mu$, $\gamma$ or $\alpha$ joins $(\epsilon,v_0,v,\lambda)$ in the grid chosen on four fifths of the questions and scored on the fifth. Fixed in advance and not tuned: $\tau=0.817$, $w_{\min}=0.2$, $\lambda_b=\lambda_t=1$, $\mu=0.5$, $\tau_s=0.6$.
- *A second, unworn check.* The reasoner has never seen the fit rows. The rollouts of Module 2 give the zero-shot reasoner alone, under A0 and under the adopted prompt, on all 1,199 fit rows, with cards built leaving out the row itself and slices defined leave-one-row-out. A paired difference there has a standard error near 0.010, against about 0.030 on the sent dev rows. It is produced in Module 2, so it is read after the Module 1c dev comparison and before any fine-tuning spend: an adoption of A3 stands only if the fit-row paired difference is also above zero. It covers the prompt comparison only. Nothing fused can be scored on fit rows, because the memoriser is in-sample there, and the fine-tune has no unworn rows, because the fit rows are its training data.
- *Complementarity monitor.* For every arm and every $\alpha$: top-1 agreement between reasoner and memoriser on sent rows (0.39 today), and the fused gain split by agree and disagree.
- *Why these are strict.* Dev has judged dozens of designs. The two dev gains of +0.019 that passed came through at +0.009 and +0.015 on the public board; a +0.007 gain whose range crossed zero reversed to −0.006.

**Unit tests.**

| Area | Checks |
|---|---|
| Slices | Masks are disjoint and cover all rows; counts are 101 / 94 / 106 on dev and 101 / 86 / 83 on the 270 rows; identical to `unseen_pairs`. |
| Router | `product` reproduces `reports/q38_fuse.json` (0.8037 cross-fitted, 0.8142 in-sample, choice (0.01, 1, 0.5)); any row with $s_i=0$ is $M_i$ before balancing in `product` and `soft` and $\mathrm{Sub}_i$ in `hard`; `hard` with $g=1$ and $s=1$ equals the reasoner's order before balancing; outputs are row-stochastic with shape (rows, 76); gate features raise an error for a row with no reasoner answer; the gate for a row is never fitted on its own question. |
| Exemplars | No card's source row equals the prompt row in any mode; no test row in the index in any build and no dev row in the index used for dev numbers; in `cross` mode every shortlisted error code with a fit example on another question gets exactly one card and the rest get the description only; card text never marks which code is true; a range step and an empty span render. |
| Prompts | v1 request bytes unchanged (hash); v2 key-list sentence and schema agree; shortlist order differs across rows and carries no scores. |
| Debias | $b_i$ sums to 1 and $0<b_i(y_i)<1$ on every fit row; weights have mean 1 and minimum at least $w_{\min}$; $\gamma=0$ gives all ones; rationalised chains' weight share is at most their row share. |
| Noise | $\hat T$ has shape (76, 76), rows sum to 1, every entry positive; $T=I$ gives plain cross-entropy and gradient $\hat p-\tilde q$; finite-difference gradient check; a test that fails if $T$ and $T^{\top}$ are swapped; no dev row used when dev is scored. |
| Chains | Wrong code or wrong root line is dropped; a range step and a renumbered transcription are handled; a rationalised chain is stored without the marker; the NONE cap holds; the share of training prompts whose shortlist contains the gold code is within 0.03 of the dev share on sent rows. |
| Reproduction | The rebuilt submission is byte-identical to the staged 0.73183 file. |

---

## 7. Stages, cost and pass marks

Costs are **estimates** from measured throughput: one RTX 6000 Ada at $1.09 per hour gives about 155 output tokens
per second, 270 thinking answers in 56 minutes after 12 minutes of setup; sent rows average about 2,400 output
tokens. Longer v2 answers are assumed to cost 20 to 50% more; Module 1c measures the real figure and later stages
are re-quoted from it. Each GPU stage needs explicit permission and a cap increase.

| Stage | What | Needs | Cost | Pass mark |
|---|---|---|---|---|
| Step 0 | Commit, tag, manifest | Your go-ahead | none | Reproduction test passes (it does today) |
| Module 1a | `slices`, `router`, `exemplars`, `prompts`, `build_shortlist`, tests, `eval_router.py` on existing outputs | CPU | none | **Done as new files** (section 9). Tests pass; the router report reproduces the recorded fusion numbers. Flags on existing scripts wait for your confirmation. |
| Module 1b | Both faithfulness audits (4.1.1) | CPU | none | **Done** (4.1.1). The decision rule does not fire. |
| Module 1d | `debias` and `noise` libraries with unit tests, no caller | CPU | none | Not built yet |
| Module 1c | A0 on the 31 dev rows it missed; arms A1, A2, A3 and A3′ on the 158 sent dev rows; A4 and A5 if wanted | One pod, about 4 to 5 hours | about $5 | A3 against A0, section 6 |
| Module 2 | Serving check with a dummy adapter. Zero-shot v1 pass on fit rows; one rollout per fit row with the adopted prompt; rationalisation for misses | One pod, about 12 to 15 hours | about $14 to $16 | At least 45% of error rows (357) yield a kept chain before rationalisation and 60% (476) after. Between 450 and 475 the stage continues but is reported as below its mark; stop below 450. The dev yield was 51%, so the 45% mark is not safe. Unworn check of section 6 read here. |
| Module 3 | LoRA fine-tune, three runs on the same rollouts: $\gamma=0$, $\gamma=1$, and $\gamma=0$ without rationalised chains; dev pass at $\alpha=0.5$ for each, other values only if needed | One 80 GB pod | about $18 to $30, the least certain figure | Fine-tuned against zero-shot, section 6 |
| Module 4 | Test scoring of sent rows with the chosen system | Two pods | about $10 to $12 | Staged only; submission needs its own yes |
| Not proposed | Blind Claude answers on fit rows for $\hat T$; retraining a memoriser member with PoE or the forward loss | Claude API; several pods | about $11; not costed | |

---

## 8. Expected effect by slice

Dev, MAP@3, all 301 rows, fused system. The first two rows are measured. The rest are estimates; each overall range
is the slice-weighted range, (101 × no-error + 94 × seen-pair + 106 × hard) / 301.

| System | No error | Seen pair | Hard | Overall | Basis |
|---|---|---|---|---|---|
| Submitted | 0.916 | 0.871 | 0.500 | 0.755 | Measured |
| + zero-shot reasoner (today) | 0.929 | 0.855 | 0.576 | 0.781 | Measured, cross-fitted |
| + v2 wording, per-line field (A1) | 0.91–0.93 | 0.85–0.87 | 0.55–0.59 | 0.76–0.79 | Estimate. Paper 1: wording is worth about nothing. |
| + shortlist (A2) | 0.91–0.93 | 0.85–0.87 | 0.55–0.63 | 0.76–0.81 | Estimate. Unknown size: paper 8's +21 points is for an oracle topic and its mechanism is not established. |
| + cross-question cards (A3) | 0.91–0.94 | 0.85–0.88 | 0.54–0.64 | 0.76–0.82 | Estimate. Could be negative. |
| + same-question cards (A4) | 0.91–0.94 | 0.86–0.90 | 0.50–0.61 | 0.75–0.81 | Estimate. The record predicts seen-pair up and hard down. |
| + fine-tuned reasoner, $\alpha=0.5$ | 0.91–0.95 | 0.86–0.90 | 0.56–0.67 | 0.77–0.836 | Estimate. See the note below. |
| Requested targets | 0.95 | 0.90 | 0.72 | 0.853 | Arithmetic |
| Claude Opus with the photo | 0.934 | 0.812 | 0.715 | 0.819 | Measured |

**Where a gain can come from.** Seen-pair rows offer little: the memoriser alone (0.871) is already above the fusion (0.855), so restoring that is worth about +0.005 overall, and beyond it the reasoner would have to beat the memoriser
on its own ground. No-error rows are at 0.92 to 0.93 against Claude's 0.934. A real gain has to come from hard rows,
and fine-tuning a classifier has not lifted hard rows above the untrained model in this project (the 32B model: 0.392 untrained, 0.357 trained). The one fine-tuned component that moved them is the pointwise ranker (+0.058 [+0.014, +0.102] fused).

**Dev-to-test differences that scale the gain.**

| | Dev | Test |
|---|---|---|
| Rows sent to the reasoner | 52% | 46% |
| Median confidence of the submitted system | 0.80 | 0.86 |
| Rows on a question absent from the training rows the models saw | 13% | 4.3% |
| Label mix distance to the training rows | 0.109 (a stratified fold) | not known; 0.322 expected for a random split |

All of the fusion's gain is on sent rows. Today's measured gain used the 135 answered sent rows (45% of dev), level with the 46% test rate, so it needs no send-rate scaling. Once the 23 unanswered sent rows are scored, dev is at 52%, and dev gains measured from then on scale by about 0.87 before other losses. Past dev gains kept about half to three quarters of their size on the public board.

**Risks, in order.**

1. *The reasoner stops disagreeing with the memoriser.* The fusion works because the two disagree on 61% of sent rows. This design feeds the memoriser's top 8 into the reasoner, may add same-question cards, and then fine-tunes on the labels the memoriser learned. Each step raises agreement. Every earlier member that resembled existing ones added nothing (experiments 59, 111 to 113, ranker v2). Mitigations: the agreement monitor, the zero-shot answers kept as a separate factor, and the rule that an arm must raise the fused score.
2. *The shortlist and cards lower hard rows.* Examples in the prompt lowered hard rows in both earlier reasoning-model tests. 12% of sent hard rows have the true code outside the list. Mitigations: shuffled order, no scores, an explicit way to answer outside the list, A2 measured apart from A3.
3. *Fine-tuning teaches the reasoner the question shortcut*, as it taught every earlier model, from about 610 to 720 chains made of rows the model already solves. Stage B trained on 9,705 rows and failed. Mitigations: the $\gamma=1$ arm, the root-line filter, interpolation, the fixed $\alpha$.
4. *Dev cannot resolve the expected steps.* The estimates above are +0.01 to +0.02 per stage; the pass mark the data can certify is +0.025. Power to detect a true +0.02 is under 40%. A stage can therefore be real and still fail its mark. The unworn fit-row check is a partial remedy, for the prompt comparison only.
5. *Serving.* Adapter serving for this model, and the match between training and serving weights, are both unverified (4.5.6).
6. *Budget.* About $46 to $62 in pods beyond the current cap.

---

## 9. Status: what exists now

Built on 14 November 2026 as new files only. No existing file was edited, nothing was launched, and no test photo or
test row is read by any of them except the saved test prediction bundles in the reproduction test.

| File | What it does |
|---|---|
| `src/k12/slices.py`, `tests/test_slices.py` | The three slices and the four-slice report. |
| `src/k12/router.py`, `tests/test_router.py` | Product, hard and soft fusion; gate features; cross-fitted gate. |
| `scripts/eval_router.py` → `reports/router_benchmark.json` | The benchmark of 4.2.3 on 301 and 270 rows. Reproduces `reports/q38_fuse.json` exactly (0.8037037 cross-fitted, 0.8141975 in-sample). |
| `src/k12/exemplars.py`, `tests/test_exemplars.py` | Code-keyed example index and cards. |
| `src/k12/prompts.py`, `tests/test_prompts.py` | Prompt v2, its schema, and the shortlist block. |
| `scripts/build_shortlist.py` → `data/v10/shortlist_dev.json` | Dev shortlists and their recall by slice. |
| `scripts/build_vlm_requests_v2.py` → `data/v10/q38_dev_A{1,2,3,4}_requests.jsonl` | Ready-to-send dev requests for arms A1 to A4 on the 135 sent rows that have a zero-shot answer. Not sent anywhere. |
| `scripts/audit_transcript_faithfulness.py` → `reports/transcript_faithfulness.json` | The audits of 4.1.1. |
| `tests/test_reproduce_submission.py` | Rebuilds the 0.73183 file byte for byte, with two negative controls. |

**Tests.** `PYTHONPATH=src:scripts python -m unittest discover -s tests`: 96 tests pass (22 that existed before, 74
new), none skipped. Two reviewers mutated the new code to check that the tests catch defects: 48 of 49 mutations were
caught, and the survivor does not change behaviour.

**Prompt sizes (text characters, 135 rows).** A0 17,500 at the median; A1 18,800; A2 23,900; A3 and A4 31,400.
A3 is roughly 9,000 to 10,000 prompt tokens against the server's 32,768 limit.

**Things the reviewers flagged that are still open.**

- In A3, a code shown with its description only is the true code in 4 of 11 cases, against 10% for codes shown with a card. This comes from the fit labels and the zero-shot answer, not from dev labels, but it is a visible pattern.
- In A4, 26% of same-question cards are the true code against 7% of different-question cards. That is the seen-pair prior entering the prompt, by design.
- 94 of 1,193 A3 cards use an example whose question depends on a diagram that the card does not show.
- Cards show the teacher's root step as written, and 121 of 794 are ranges such as `L2-L3`, while v2 asks the model for a single `L<n>`.
- Whether the server accepts the v2 answer schema (a pattern on `root_line`, an unbounded `line_checks` list) is untested until a pod run. The existing client drops the whole schema for every later row if the server rejects it; smoke-test four rows first.
- `router.fuse()` does not yet pass the second reasoner factor through, although `product_fusion` supports it.
- The 23 sent dev rows with no zero-shot answer have no requests yet; they need the A0 pass of Module 1c first.
