# Research synthesis for K-12 misconception detection

Written 14 November 2026. Covers the eight PDFs in `research-papers-k12/` (201 pages) and maps each one to the
bottlenecks measured in `docs/experiment_review_2026-11-14.md`. Nothing was launched, spent or submitted to produce
this file. The only computation was read-only CPU scoring of prediction files already in the repository (section 4).

**How to read this file.** Sections 1 to 5 are the synthesis. Section 6 holds one verified digest per paper, with every
equation in LaTeX. The full per-paper notes (about 3,000 words each, with page references) are in
`docs/research_notes/`. In the digests, `[paper]` marks something the paper shows and `[extrapolation]` marks our own
inference from it. Sections 1 to 5 were reviewed by an independent critic against the notes and corrected.

**Terms.** MAP@3 is the competition metric. "Dev" is the 301-row development fold. "The 270 rows" are the dev rows
Qwen3.8-27B answered; the 31 it missed are all error rows, so the 270 rows are easier than dev as a whole.
"No-error", "seen-pair" and "hard" rows are the three dev slices (101, 94 and 106 rows): the true label is NONE; the
true (question, code) pair appears in the fit rows; it does not. "Submitted system" is the pipeline behind the 0.73183 file, the best file when this work began; the fusion described here was submitted during the session and scored 0.76266. "Sent rows" are those where the submitted system's top probability is below 0.817. "Fusion" is the current
product of the submitted system with Qwen3.8-27B's rank shares on sent rows, followed by label balancing.

---

## 1. How the papers were extracted

| Step | What was used | Note |
|---|---|---|
| Tool check | `pymupdf`, `pdfplumber`, `pdf2image`, `pdftotext`, `pdftoppm`, `mutool`: not installed. `pypdf` and `Pillow`: installed. macOS `sips`: available. | Nothing was downloaded or installed. |
| Text | `pypdf`, one text block per page | Loses maths layout, so it was used only to locate things. |
| Page images | Each page split with `pypdf` and rendered with `sips` at 1,900 px, then flattened onto white | The first renders had transparent backgrounds that displayed black; all 201 pages were re-flattened. |
| Extraction | One agent per paper read the full text and transcribed equations, algorithms and tables from the page images | Eight notes in `docs/research_notes/`. |
| Verification | A second, independent agent per paper re-read every equation, hyperparameter and table value against the page images and fixed the note in place | 76 corrections across the eight notes; each note ends with the list of pages checked. |

Corrections that changed a conclusion:

- Paper 1: "larger models over-correct more" is not monotone. The smallest model in each family over-corrects least; the two largest are within 0.5 points of each other.
- Paper 3: the gain from rationalisation is confounded, because the share of self-generated examples grew in the same run.
- Paper 5: the in-domain cost of Learned-Mixin +H is tied by the paper to how useful the bias is, not to the entropy weight; no sweep is reported.
- Paper 6: a question-only bias model cannot reach probability 1 on every row; it converges to the in-sample frequency $n_{q,y}/n_q$, which is still 1 on single-code questions.
- Paper 7: with an estimated matrix the forward correction's gain over plain cross-entropy is +0.2 to +2.0 points at low asymmetric noise, −1.1 to +1.3 at symmetric noise, and +11.9 to +24.0 at high asymmetric noise on five of six settings.
- Paper 8: the gain from restricting to the true topic cannot be attributed to removing out-of-topic codes alone.

---

## 2. The papers at a glance

| # | Paper | What it is | Bottleneck it speaks to | What it gives us | Main limit for us |
|---|---|---|---|---|---|
| 1 | Seong et al. 2026, "When VLMs 'Fix' Students" (arXiv:2604.22774v2) | An evaluation metric (PINK) for transcripts that silently correct student errors; 15 untrained vision models | Error text present in only 55% of our transcripts | A way to measure over-correction; evidence that a "do not fix" prompt nets to zero | The one fix tested is that prompt; no training or decoding method is proposed. Errors are injected, not real. |
| 2 | Wortsman et al. 2022, WiSE-FT (arXiv:2109.01903v3) | Interpolate zero-shot and fine-tuned weights | Fine-tuning trades hard rows for seen-pair rows | The interpolation formula and evidence that it beats tuning less | Image classification only (CLIP, ALIGN, JFT, BASIC); "optimal $\alpha$" is chosen on the test set |
| 3 | Zelikman et al. 2022, STaR (arXiv:2203.14465v2) | Fine-tune on self-generated rationales whose answer is right | Teaching a thinking model the annotators' conventions | Algorithm 1; the restart-from-base rule; rationalisation | Finds that more, hotter samples hurt; 7,000+ rows, many rounds, 6B text model |
| 4 | Pan et al. 2025, LEMMA (arXiv:2503.17439v2) | Train a solver on "wrong step → reflection → correct solution" | Synthetic errors carry half the signal of real ones | Type-conditioned error injection; a chain anchored on the first wrong step | Trains a solver, never a classifier of error types |
| 5 | Clark et al. 2019, "Don't Take the Easy Way Out" (arXiv:1909.03683v1) | Train the main model in an ensemble with a frozen bias-only model | Our models answer "the usual code for this question" | Bias Product, Learned-Mixin +H | Every test set removes or reverses the bias; ours keeps it on many rows |
| 6 | Karimi Mahabadi et al. 2020 (arXiv:1909.06321v3) | Product of Experts and Debiased Focal Loss | Same as paper 5 | The two loss formulas | Bias-free-set gains of +0.8 to +10.1 points for BERT; in-domain within about one point for BERT, −3.9 to −10.7 for the weaker encoder |
| 7 | Patrini et al. 2017 (arXiv:1609.03683v2) | Forward and backward loss correction with a noise transition matrix | Claude's labels follow different conventions from the annotators' | The forward loss; counting $T$ from paired labels | Class-conditional noise only; no case with as few pairs per class as ours |
| 8 | Otero, Druga, Lan 2024 (arXiv:2412.03765v1) | A 55-misconception algebra benchmark, GPT-4-turbo with one example per misconception | Rows whose exact pair had one teacher example in training score 0.82 against 0.50 with none (a training-data effect, not an in-context one) | A level for description-plus-example prompting (53% over 55 codes); a hint that narrowing candidates matters | No arm without examples, so it shows no effect of the example itself |

---

## 3. What each bottleneck gets from the papers

### 3.1 Hard rows: 0.500 for us, 0.715 for Claude, 0.608 for Qwen3.8-27B

No paper addresses this directly. The nearest evidence:

- **Paper 8** scores 53.0% top-1 over all 55 codes and 73.8% in a second experiment restricted to the true topic, which is an oracle. The paper does not say whether the candidate list was cut or how topics were sampled, and per-topic gains do not match the share of out-of-topic errors (one topic has 3% of its predictions out of topic and gains 12 points). So the +21 points is an upper-end hint, not a measured effect of narrowing. Our equivalent is the shortlist, whose recall on sent hard rows is 0.88 (section 4.3).
- **Our reading of paper 8's nine worked errors**, not a claim of the paper: in at least three, the predicted code's example shares the test item's story. The authors say only that the model matches patterns. This is consistent with the mechanism behind our −0.052 result for cross-question examples (experiment 79), not proof of it.
- **Papers 3 and 4** bear on training the reasoning model (section 3.5).
- **Papers 5 and 6** would raise hard-row accuracy only if the model already has the skill and the question prior is masking it. Neither paper tests that condition; in both, the main model could already solve the task from the remaining input.

### 3.2 The transcript holds the annotated error text only 55% of the time

- **Paper 1** confirms the phenomenon across model families: 42.1% to 66.2% of transcripts have at least one over-correction event. That rate includes grader noise (60.4% of events are 0 to 5 points out of 20) and is not comparable with our 8-of-173 final-answer check.
- **It weakens the planned fix.** An explicit "transcribe exactly, do not fix" prompt cut the event share from 51% to 47% on average, and the faithfulness score moved from 0.75 to 0.75. On the strongest two models the score fell by 0.03. The authors conclude that prompting alone is not enough. Our reasoner's prompt already contains that instruction ("preserving the student's own symbols and mistakes exactly; never correct them"), so our current zero-shot run is already the paper's mitigated condition.
- **It is about untrained general models**, and the smallest in each family over-corrects least. Our transcriber is a fine-tuned 8B model, the one class the paper does not cover. Our reasoner is an untrained 27B model, the class it does cover. So "the reasoner reads the photo" is not by itself evidence that it reads faithfully.
- **Where the damage is.** The two late rubric components (late calculation, final value) hold 46.7% of events against 40% if uniform: a moderate skew, by rubric component, not by line position.
- **It does not test** any decoding setting, any fine-tuned transcriber, any verification step, or real student errors.
- **What our record adds.** Our 55% figure mixes three causes the paper's taxonomy separates: a token silently fixed, a line dropped or rewritten, and an ordinary misread. The split is unmeasured. It can be measured for free on the 200 dev error rows (the only rows with out-of-fold machine transcripts), and on the reasoner's own transcription field for the 270 rows it answered.
- Reading the suspect line directly from the image is our extrapolation. Its support comes from our record, not the paper: Claude loses 0.048 overall and 0.078 on hard rows when it reads our transcript instead of the photo.

### 3.3 Fine-tuned models memorise the usual codes per question

- **Papers 5 and 6 give the losses** (section 6). Both train the main model against a bias-only model and use the main model alone at test time.
- **Both papers evaluate only where the bias is absent or reversed.** The in-domain cost depends on the model. For BERT in paper 6 it is −0.96 to +0.32 points on the language-inference tasks under product of experts or focal debiasing; on the fact-verification task it is +0.47 for product of experts and −2.92 for focal. For the weaker encoder it is −3.9 to −10.7, and up to −11.3 on paper 5's synthetic data. On rows where the shortcut is right (paper 5, Table 8), the recurrent model falls from 97–100% to 71–80% under Reweight and Bias Product and to 47–67% under Learned-Mixin +H. BERT falls to 68–84% under Reweight and 54–71% under Bias Product, and keeps 92–98% only under Learned-Mixin +H. Reweight is the form our design wires in.
- **How much of our data keeps the bias.** About 65% of dev rows are not hard rows. The question prior itself scores 0.497 MAP@3 on dev; its top label is right on about a third of rows, and the true label is among the question's fit labels on about half.
- **Experiment 76.** Our CPU proxy subtracted the prior from already-trained models at inference, and no-error and seen-pair rows collapsed. The record gives no cause. Paper 5's "Dependent" experiment shows that over-compensation makes a model anti-correlated with the bias (22.5% against 33% for an unbiased model); reading experiment 76 that way is our interpretation.
- **Neither paper cross-fits the bias model.** With 2.78 fit rows per question, an in-sample question prior gives the true label a probability near 1 on single-code questions, which removes the gradient for exactly those rows. A leave-one-row-out prior is required (formula in section 6, papers 5 and 6).
- **Size of the prize.** Paper 6's bias-free-set gains for BERT run from +0.8 to +2.5 points (language-inference hard sets) to +4.4 to +9.8 (two adversarial sets; +10.1 with a setting tuned on the test set, which cost 1.75 points in domain). Paper 5's closest analogue to our prior, a question-type bias on a visual question-answering set, gains +12.9 with Learned-Mixin +H (its entropy weight chosen on the test set) and +9.5 with Learned-Mixin, but only +0.9 with Reweight and +0.8 with Bias Product, the two forms our design implements. On a 35% hard share the whole range is +0.003 to about +0.045 overall, before any in-domain cost; for the forms we implement it is the low end. No setting has a bias model that can memorise the training set, a 1,500-row training set, or a test set where the bias holds on most rows.

### 3.4 Claude's labels plateau because its conventions differ from the annotators'

- **Paper 7 gives the forward loss**, $-\log\big(T^{\top}\hat p\big)_{\tilde y}$ with $T_{ij}=p(\tilde y=e^j\mid y=e^i)$, and shows on Clothing1M that $T$ can be counted from paired clean and noisy labels.
- **The correction alone was worth +0.9 points there** (68.94 → 69.84). The large step (80.38 against 75.19) came from fine-tuning the forward-trained network on clean labels afterwards, in a single run with no control.
- **The paper's one many-class estimate failed at high noise.** On CIFAR-100 at 60% asymmetric noise, an anchor-point estimate from 500 images per class made forward correction worse than no correction (15.9 against 17.1). In the other CIFAR-100 settings the estimated matrix was within −0.4 to +0.7 points of plain cross-entropy, against +4.6 to +6.1 with the true matrix. We would count the matrix from paired labels instead, which the paper does not test at our sample size.
- **Our position (section 4.5).** The only blind Claude answers on human-labelled rows are the 301 dev rows: about 4 pairs per class, and 16 of 76 classes with five or more. Claude's fit-row outputs were produced with the gold code in the prompt and cannot be used. Our disagreement rate (26% overall, 36% on error rows) sits between the paper's low and high noise settings.
- **Most of Claude's NONE skew is label shift, not noise.** Claude says NONE on 93.1% of human NONE rows and on 3.5% of human error rows. Under the class-conditional model the pool's implied true NONE rate is $(0.709-0.035)/(0.931-0.035)=0.75$. The pool simply contains more correct work. Forward correction does not address that; prior rescaling does, and our earlier prior correction did not survive blending (experiment 73).

### 3.5 Fine-tuning a thinking model on about 1,000 gold rows without losing zero-shot strength

- **Paper 3 (STaR) supports** the filter (keep a chain only if its answer equals the gold answer), retraining from the original weights every round, and rationalisation with the hint removed from the stored example.
- **Its caveats on rationalisation.** It added 3.7 points on one task (confounded) and 0.6 on another; rationalised chains can be template matching; and the paper says bias amplification is "made worse by rationalization".
- **Paper 3 argues against `N=8` rollouts.** More and higher-temperature sampling was "counterproductive" because it keeps correct answers reached by incorrect reasoning; temperatures 0.5 and 0.7 were worse than rationale generation alone. The paper used one rollout per question at a low temperature whose value it does not state.
- **How that applies to us.** The mechanism depends on the chance of a lucky pass: 20% for the paper's 5-way task, about 1.3% per rollout for our 76 labels. Our only validated setting for Qwen3.8-27B is temperature 1.0, hotter than anything the paper tried. No low-temperature run exists in our record.
- **Paper 3's regime is far from ours.** It used 7,000 to 10,000 rows, 16 to 46 full retrains and a 6B text model with no built-in reasoning, with 5-way labels on one language task and free numeric answers on the other. It reports no result after a single round on its two language tasks (only arithmetic: 2-digit sums from under 1% to 32% after one round with rationalisation).
- **Paper 4 (LEMMA).** Its largest ablation is cutting the trajectory at the first wrong step (38.3 → 22.8 on MATH without it). The hazard that removes is on the output side, a solver learning to write wrong continuations, and does not arise when the wrong working is our input. What transfers is the skeleton of a chain anchored on the first wrong line. The paper gives no ratio of clean to erroneous examples and never isolates type-conditioned sampling. It also contains counter-evidence for a gold-filter-only recipe: training on the model's own correct solutions alone scores 50.9 against 62.4 with error-and-reflection trajectories.
- **Paper 2 (WiSE-FT) supports** fine-tuning normally and then scaling back, over tuning less (a lower rate, fewer epochs, early stopping and L2-to-init all land on or below the interpolation curve). For LoRA the interpolation is adapter scaling: $(1-\alpha)W_0+\alpha\big(W_0+\tfrac{s}{r}BA\big)=W_0+\alpha\tfrac{s}{r}BA$.
- **Paper 2 suggests the gain over what we already do may be small.** Output-ensembling the zero-shot model with a separately trained model came within 0.6 and 1.0 points of weight interpolation, and with a stronger partner exceeded it. It does not test interpolation plus an independent ensemble, which is our case.
- **Choosing $\alpha$.** The paper's "optimal" value is picked on the evaluation set. Its only non-oracle rule is fixed $\alpha=0.5$: within 0.0 to 0.5 points of the optimum on CLIP's averaged columns and +0.6 to +2.8 in distribution on seven CLIP datasets, but −1.9 and −2.5 against the fine-tuned endpoint for two other model families.

### 3.6 Combiners over existing outputs are exhausted

No paper offers a new combiner. Paper 2's diversity analysis explains why the Qwen3.8 fusion worked where the earlier
combiners (experiments 12 to 17, 66 to 71, 104 to 114) failed: the zero-shot and fine-tuned models disagree often and
each is more confident where it is better. That is a property of the members, not of the mixing rule. **Measured**:
our two systems' top answers agree on only 39% of sent rows, and almost all of the fusion's gain is on the rows where
they disagree.

---

## 4. Measurements made for this synthesis

All read-only, on files already in the repository, on CPU. Scripts are in the session scratchpad and are not part of
the pipeline. Scores are MAP@3. "In-sample" means the setting was chosen on the same rows and is optimistic.

**4.1 Routing against multiplying (270 rows).**

| System | All | No error | Seen pair | Hard |
|---|---|---|---|---|
| Submitted system | 0.775 | 0.916 | 0.872 | 0.502 |
| Qwen3.8-27B alone | 0.759 | 0.924 | 0.709 | 0.608 |
| Fusion, cross-fitted | 0.804 | 0.929 | 0.853 | 0.600 |
| Fusion, in-sample setting | 0.814 | 0.929 | 0.861 | 0.627 |
| Router with the TRUE slice known: no-error and seen-pair → submitted, hard → Qwen3.8 | 0.807 | 0.916 | 0.872 | 0.608 |
| Per-row best of (submitted, Qwen3.8), an upper bound | 0.874 | 0.954 | 0.924 | 0.723 |

- A router told the true slice is not distinguishable from the fusion: −0.007 [−0.037, +0.022] against the in-sample fusion, +0.004 [−0.025, +0.033] against the cross-fitted one. On hard rows the fusion is 0.627 in-sample and 0.600 cross-fitted against 0.608 for Qwen3.8 alone, so there is no evidence that the memoriser helps or hurts there.
- Twelve hand-built gates from test-time facts were tried in a first probe; the learned gate and the full benchmark on all 301 rows are in section 4.9. The router as requested (gated rows to Qwen3.8 alone, the rest to the submitted system alone) scored 0.762 to 0.791, which is 0.013 to 0.042 below the cross-fitted fusion. A hybrid (gated rows to Qwen3.8, the rest left on the fusion) scored 0.006 to 0.048 below the in-sample fusion; the best gate was "confidence below 0.4", −0.006 [−0.015, +0.006].
- The true slice cannot be observed at test time, because "unseen pair" is defined by the true label. The best observable proxy identified hard rows with 76% precision.

**4.2 The same fusion on all 301 rows.** With the 31 unanswered rows left on the submitted system: 0.781 cross-fitted (0.929 / 0.855 / 0.576) against 0.755 for the submitted system, +0.026 [+0.002, +0.050].

**4.3 Shortlist recall on hard rows.** Submitted system top 3 / 5 / 8: 0.698 / 0.792 / 0.849 (106 rows). On the 83
answered hard rows: 0.855 for its top 8, 0.904 for the union with Qwen3.8's three codes. On the 68 of those that are
sent: 0.88.

**4.4 What Qwen3.8-27B gets right, by slice (270 rows).**

| Measure | No error | Seen pair | Hard |
|---|---|---|---|
| Top-1 accuracy | 0.911 | 0.628 | 0.530 |
| Truth among its three codes | 0.941 | 0.814 | 0.699 |

On the 169 answered error rows, its top code is right on 58%, and both the code and the root line (within one line of
the teacher's) are right on 51%.

**4.5 Paired Claude and human labels.** 301 pairs (dev only). Top-1 agreement 0.738. 60 human classes appear; 16
have five or more rows; 7 have ten or more. The largest off-diagonal cell holds 3 rows. On human error rows Claude
gives the same code 64%, a different code in the same family 7%, another family 25.5%, and NONE 3.5%.

**4.6 Human examples per code (fit rows only).** 68 of 75 codes have at least one; 10 have exactly one; 7 have none
(`ALG-02`, `ANG-02`, `GEO-01`, `INT-01`, `NOT-02`, `PRO-03`, `STA-03`). 97.8% of shortlisted error codes have an
example on another question. For 95% of hard rows the true code has such an example (median 23).

**4.7 Examples in the prompt, by slice (experiment 48, re-scored).** Claude Opus with four fit-row examples,
same-question first: no-error 0.944 (zero-shot 0.934), seen-pair 0.855 (0.812), hard 0.645 (0.715); overall 0.811
against 0.819.

**4.8 What the two readers do to the annotated error text** (`reports/transcript_faithfulness.json`; dev error rows).

| | Transcriber | Qwen3.8-27B's own transcription |
|---|---|---|
| Span present, the strict definition behind the recorded 55% | 0.555 (200 rows); 0.582 on the 165 common rows | 0.612 on the 165 common rows |
| Span kept, tolerant matching | 143 of 196 (0.730) | 132 of 165 (0.800) |
| Lost spans read by hand: error corrected | 11 (5.6% of error rows) | 6 (3.6%) |
| Lost spans read by hand: changed but not corrected | 34 (17.3%) | 16 (9.7%) |
| Lost spans read by hand: same content, other formatting | 6 | 10 |

The hand reading is by one reviewer, from transcripts without photos, not blind to the system. Silent correction is
about 6% of error rows for the transcriber. The larger loss is text changed without being corrected. Lost spans are
equally common in the first and second half of a working.

**4.9 Router benchmark on all 301 rows** (`reports/router_benchmark.json`; rows without a reasoner answer keep the
submitted system; differences are against the cross-fitted product fusion).

| System | All | No error | Seen pair | Hard | Difference |
|---|---|---|---|---|---|
| Submitted | 0.755 | 0.916 | 0.871 | 0.500 | −0.026 [−0.050, −0.002] |
| Product fusion, cross-fitted | 0.781 | 0.929 | 0.855 | 0.576 | reference |
| Hard router, true slice as the gate (cannot be deployed) | 0.787 | 0.916 | 0.871 | 0.590 | +0.006 [−0.017, +0.030] |
| Hard router, learned cross-fitted gate (AUC 0.816) | 0.758 | 0.919 | 0.762 | 0.601 | −0.023 [−0.047, +0.001] |
| Hard router, best of three hand gates (confidence below 0.4) | 0.770 | 0.929 | 0.872 | 0.527 | −0.012 [−0.032, +0.010] |
| Soft fusion, everything nested inside the folds | 0.782 | 0.929 | 0.855 | 0.577 | +0.001 [−0.005, +0.007] |

96% of the fusion's gain is on the sent rows where the two systems' top answers disagree.

---

## 5. What this means for the five planned pillars

| Pillar as requested | What the papers and our data support | What they do not support | Decision carried into the architecture |
|---|---|---|---|
| 1. Verbatim transcription, line-by-line error diff, and image verification on uncertain rows | Over-correction is real and moderately skewed toward late-calculation and final-value components (paper 1). The photo is worth 0.048 to Claude (our record). | A "do not fix" prompt alone: no net gain in the paper, and our prompt already has it. Line-by-line symbolic checks: our checker covered 40% of lines (experiment 75). The "45% lost to auto-correction" reading: the 55% figure also includes dropped lines and misreads, and the split is unmeasured. That an untrained 27B model reads more faithfully than our fine-tuned transcriber: paper 1 points the other way. | The split is now measured (section 4.8): silent correction is about 6% of error rows, and the reasoner keeps the error text at least as often as the transcriber. Make the per-line check an output field. Specify cropped-image verification as its own arm. |
| 2. Replace soft fusion with a hard seen-versus-unseen router | The two model types are complementary (per-row best 0.874). | Hard routing: even a router told the true slice only matches the fusion, and none of twelve hand-built observable gates beat it. Earlier routers and adaptive fusions also failed (experiments 25, 70, 96, 107). A learned gate with a cross-fitted AUC of 0.816 still loses 0.023 as a hard router. | Build the router and report it, but keep the product fusion as the default. The send flag decides which rows get the expensive model. |
| 3. Product-of-experts or focal debiasing, plus forward loss correction | The loss formulas (papers 5, 6, 7). Counting $T$ from paired labels (paper 7). Bias-free-set gains from small to moderate. | Any gain where the bias still holds on most rows; the in-domain cost for our case is unmeasured. A $T$ estimated from about 4 pairs per class. Forward correction fixing the NONE skew, which is label shift. | Wire in only the reweighting form, as a paired arm in the reasoner's fine-tuning. Implement and test the losses and the matrix, with no caller, so that a later retrain of a memoriser member as a hard-row specialist could use them; that retrain is not proposed. |
| 4. One human example per candidate code in the reasoning prompt | Examples exist for 98% of shortlisted codes. Narrowing candidates may help (paper 8, mechanism not established). | That an example per candidate helps: paper 8 has no arm without examples. In our record, examples raised seen-pair rows and lowered hard rows for both reasoning models tested (experiments 48, 60); a text matcher lost 0.052 (experiment 79). | Build the code-keyed index and test it as the first confirmatory comparison (cards against no cards, same model, same rows). Same-question cards are a separate arm. The 0.50 → 0.82 step was a training-data effect for a fine-tuned classifier and may not transfer to a prompt. |
| 5. Self-generated reasoning chains (`N=8`, gold-filtered) and weight interpolation | Gold filtering, restart from base, rationalisation with the hint removed (paper 3). A chain anchored on the first wrong line (paper 4). Train normally then scale the adapter (paper 2). | `N=8` hot rollouts: paper 3 found this counterproductive. A STaR result on its language tasks after one round or under 7,000 rows. A gold-filter-only recipe: paper 4's correct-only baseline is its weak one. An oracle-free rule for $\alpha$ other than 0.5. | One rollout per fit row at the validated settings (temperature 1.0, which is not low temperature in the paper's sense), with a root-line filter as the guard; rationalisation with a control arm without it; one round; fixed $\alpha=0.5$; judged against the existing fusion, not against the untrained model alone. |

**Three findings that should change the plan's priorities.**

1. The cheapest untested lever is not in any of the eight papers: the reasoning model has only ever been run zero-shot with the full taxonomy and no examples. A shortlist with 0.88 recall on sent hard rows, plus example cards, is an untested lever of unknown size, and our record warns that examples can lower hard rows.
2. Routing by slice is not a source of gain on current outputs. The headroom (0.874 per-row best) is real, but a router told the true slice does not reach it. It has to come from a better hard-row model that still disagrees with the memoriser.
3. The debiasing papers give small to moderate gains on bias-free sets with an in-domain cost that is unmeasured for our case, and the noise-correction paper has no case with as few paired labels as ours. Both belong after the reasoning-model experiments.

---

## 6. Verified per-paper digests

Each digest below was checked against the page images. Full notes: `docs/research_notes/paper1_vlm_ocr_overcorrection.md`,
`paper2_wise_ft.md`, `paper3_star.md`, `paper4_lemma.md`, `paper5_bias_ensembles.md`, `paper6_poe_debiased_focal.md`,
`paper7_forward_loss_correction.md`, `paper8_misconception_benchmark.md`.

### 6.1 Paper 1: over-correction in multi-line handwritten maths OCR (PINK)

**Citation.** Jin Seong*, Wencke Liermann*, Minho Kim, Jong-hun Shin, Soojong Lim (ETRI, Republic of Korea). "When VLMs 'Fix' Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR." arXiv:2604.22774v2 [cs.CY], 26 May 2026. Venue not stated. An evaluation paper: no model is trained.

#### Method (pp3-5, Fig. 2)

A VLM transcribes the photo with a fixed OCR prompt. One LLM grader scores both the model transcript and the ground-truth transcript (which keeps the student's error) on a rubric of $R=5$ components worth 20 points each (total 100): (1) Core Formula/Principle Identification, (2) Condition/Boundary Application, (3) Calculation Accuracy (Early-Mid), (4) Calculation Accuracy (Late), (5) Final Value Verification.

For component $r\in\{1,\dots,R\}$, with $S_{\text{model},r}$ and $S_{\text{oracle},r}$ the grader's scores for the model and ground-truth transcripts:

$$\Delta S_r = S_{\text{model},r} - S_{\text{oracle},r} \tag{1}$$

An over-correction event is any $\Delta S_r>0$. Penalised component score:

$$S'_r=\begin{cases}S_{\text{model},r}, & \Delta S_r\le 0,\\ S_{\text{oracle},r}, & 1\le \Delta S_r\le 10\ \text{(Minor, revert)},\\ 0, & \Delta S_r>10\ \text{(Major, nullify)}.\end{cases} \tag{2}$$

$$S_{\text{penalized}}=\sum_{r=1}^{R}S'_r \tag{3}$$

$$S_{\text{PINK}}=\frac{\sum_{n=1}^{N}S_{\text{penalized},n}}{\sum_{n=1}^{N}S_{\text{oracle},n}} \tag{4}$$

$N$ is the number of items. Fig. 2 writes the minor case as $0<\Delta S\le 10$; the text says major is "11 points or more". Since $S'_r\le\min(S_{\text{model},r},S_{\text{oracle},r})$, PINK $\le 1$ and lost content ($\Delta S_r<0$) also lowers it, so PINK mixes reading loss with the penalty. Worked example: student $(1-1)-0=0$, model $(1+1)-0=2$; oracle 70, model 100, $\Delta S_4=\Delta S_5=+15$, penalised 60; dataset illustration $60.4/71.3=84.7$ (×100).

Taxonomy (pp13-15): Type 1 "Simple-Level (Local Fix)", a token swap (InternVL3-8B writes $\tan^{-1}$ for the student's $\sin^{-1}$; oracle 34, model 99, penalised 27); Type 2 "Critical-Level (Global Rewrite)", erroneous steps dropped and a correct summary written (GPT-4o; oracle 14, model 100).

#### Exact settings

- Data: FERMAT, 609 problems, perturbations injected into correct solutions and hand-copied by 43 annotators, 2,244 samples; oracle mean 71.3; 500 clean samples score 95.1 (p5).
- 15 untrained VLMs: Qwen2.5-VL 7B/32B/72B, InternVL3 8B/38B/78B, Ovis2 8B/34B, Llama-3.2 11B/90B, Pixtral 12B/124B, GPT-4o, GPT-4o Mini, Gemini 2.5 Flash.
- Grader: GPT-5 (`gpt-5-2025-08-07`); second grader Qwen3-Next-80B-A3B-Thinking. Threshold $T=10$.
- Grading prompt (Fig. 16, p20): rubric as JSON, "character by character" analysis with justification, integer total 0-100, JSON-only output with per-component `score` 0-20.
- OCR prompt (Fig. 18, p22): transcribe to LaTeX "exactly as it appears in the image", fields `**Question:**` / `**Answer:**`. Mitigated prompt (Fig. 19): adds "IMPORTANT: You must transcribe EXACTLY what is written, including any errors or mistakes made by the student. Do NOT correct or fix any mathematical errors." plus a five-bullet "CRITICAL INSTRUCTIONS FOR FAITHFUL TRANSCRIPTION" block with the "2 + 2 = 5" example.
- Not stated: temperature, sampling, max tokens, image resolution, quantisation, grader reasoning settings, attention layer/head.

#### Headline results

- Share of transcripts with at least one event: 42.1% (Llama-3.2 11B) to 66.2% (Qwen2.5 32B) (Fig. 14, p17).
- PINK / BLEU (Table 1, p7): Gemini 2.5 Flash 0.935 / 0.614; Qwen2.5 72B 0.920 / 0.733; Qwen2.5 32B 0.919 / 0.734; GPT-4o 0.907 / 0.735; InternVL3 38B 0.905 / 0.771; InternVL3 8B 0.873 / 0.642; Qwen2.5 7B 0.832 / 0.565; Llama-3.2 11B 0.676 / 0.407. GPT-4o is 3rd on BLEU, 6th on PINK; Gemini 10th and 1st; on raw rubric scores they are 4th and 5th.
- Table 1's two Pixtral labels are swapped (Table 4 and Figs. 3, 5, 15 give the 124B model 0.85, rank 10, and the 12B model 0.72, rank 14); over-correction rates are 56.1% (124B) and 43.9% (12B).
- Scale: the smallest model in each family over-corrects least (Qwen2.5 7B 54.6 → 32B 66.2 / 72B 65.7; InternVL3 8B 50.8 → 38B 58.8 / 78B 58.5; Ovis2 53.6 → 62.5; Llama 42.1 → 50.1; GPT-4o Mini 49.6 → GPT-4o 62.3). Not monotone at the top of the two three-member families.
- Location (Fig. 3, p6; our sums of printed cells): 5,317 / 7,238 / 7,270 / 9,588 / 7,787 events for components 1-5, total 37,200; the last two hold 46.7% against 40% if uniform.
- Size (Table 2, p12): 0-5 points 22,483 (60.4%); 5-10 7,038 (18.9%); 10-15 3,548 (9.5%); 15-20 4,131 (11.1%). Major events are 20.6%.
- Human study (pp8, 18): 3 annotators, 200 Ovis2-8B transcripts. Votes PINK 330 (55.0%), BLEU 237 (39.5%), neither 33 (5.5%); item majority 119 / 69 / 12 tied, $p=3.27\times10^{-4}$. Weighted kappa 0.69; grader vs mean expert Pearson 0.78.
- BLEU, edit distance and MathBERTa agree with each other (Kendall $\tau$ 0.75-0.88) and not with PINK (0.47-0.52) (Table 9).
- 60.9% of BLEU discrepancies on Ovis2-8B involve formatting (8.2% formatting only, 52.7% mixed) (p2).

#### Ablations, failure modes, limits

- Prompt mitigation (Table 10, p18; six open models, Qwen3-80B grader, $N=2{,}207$). PINK original → mitigated, event share: Qwen2.5-32B .88 → .85, 62% → 53%; Qwen2.5-72B .87 → .84, 61% → 50%; Ovis2-34B .85 → .85, 59% → 54%; Qwen2.5-7B .73 → .74, 46% → 46%; Pixtral-12B .59 → .61, 41% → 38%; Llama-3.2-11B .57 → .64, 39% → 42%; average .75 → .75 (+.01), 51% → 47%. Authors: "prompt-level intervention alone is insufficient".
- Threshold (p12): rank $\tau$ against $T=10$ is 0.905 at $T=5$, 0.981 at $T=15$, about 0.66 at $T=0$ (plot).
- Graders (pp15-17): GPT-5 vs Qwen3-80B per-sample $r=0.78$ (QWK 0.78), penalty $r=0.74$ (QWK 0.73), per-model PINK $r=0.99$, rank $\tau=0.90$. Five repeat gradings ($N=1{,}803$): PINK mean 0.872, CV 0.41%, QWK 0.926. Prompt paraphrases: std ≤ 0.006 (grading), ≤ 0.015 (OCR).
- Mechanism (pp14, 16): one attention example. InternVL3-8B shows "negligible activation" on the handwritten "sin" when writing "tan"; "Qwen2.5-VL 8B" (not one of the 15 models) attends to the strokes.
- Stated limits (p9): one axis only, prompting fails, proprietary grader, one dataset, small human study.
- Our added limits: any $\Delta S_r>0$ counts, so the 42-66% includes grader noise; no noise floor reported; models that drop content have fewer chances to exceed the oracle, which confounds the scale trend.

#### Recipe for our pipeline

- [paper] Keep "do not fix" wording but expect little: −4 points of event share on average, no net PINK gain; no harm for 7B-12B models (+.01 to +.07).
- [extrapolation] Put the Fig. 19 additions in the training prompt of the next transcriber LoRA.
- [paper] Do not move to a larger untrained transcriber for faithfulness.
- [extrapolation] Keep greedy decoding as reference; use sampling only as a detector.
- [extrapolation] Cut context pressure: re-read the final-answer line from a crop without preceding lines; try "final answer first"; drop the question text from the transcriber prompt.
- [extrapolation] Upweight loss on annotated error spans; oversample rows whose error is in the last two lines.
- [extrapolation] Final-answer flip rate on out-of-fold transcripts: flip-to-right $=\#\{t\ne a^*, m=a^*\}/\#\{t\ne a^*\}$, flip-to-wrong $=\#\{t=a^*, m\ne a^*\}/\#\{t=a^*\}$ ($t$ teacher, $m$ machine, $a^*$ correct answer).
- [extrapolation] Classify each lost error span as local fix, dropped line or ordinary misread; [paper] normalise formatting first with the Fig. 17 category lists.
- [paper] PINK as defined, $T=10$, minor and major reported separately; [extrapolation] its transfer to short K-12 workings, plus grading the teacher transcript twice as a noise floor. Needs a judge model.
- [paper] Split every measure by line position (46.7% of events in the last two components).
- [extrapolation] Verify suspect lines: isolated-crop re-read, two-choice question on the crop, sampling disagreement, low attention mass on image tokens.

#### What the paper does not show

No training or decoding method; no fine-tuned model; no real student errors (injected perturbations only); no error-free samples in the evaluation; no breakdown by FERMAT error category; no per-model major-event rate; no decoding settings; no verification method; nothing on reasoning over a correct transcript, memorised question-label pairs or label conventions.

---

### 6.2 Paper 2: WiSE-FT, robust fine-tuning of zero-shot models

**Citation.** Wortsman\*, Ilharco\*, Kim, Li, Kornblith, Roelofs, Gontijo-Lopes, Hajishirzi, Farhadi, Namkoong, Schmidt. "Robust fine-tuning of zero-shot models." arXiv:2109.01903v3, 21 June 2022. Code: github.com/mlfoundations/wise-ft. Venue not stated in the PDF.

**Claim.** Fine-tuning a zero-shot model raises reference accuracy but lowers accuracy under shift; interpolating zero-shot and fine-tuned weights with one coefficient recovers both, at no extra training or inference cost.

#### Method (pp. 5–6, 21)

- Zero-shot head: image encoder $g$, text encoder $h$, captions $s_j$ = "a photo of a $\{c_j\}$"; $\mathbf{W}_{\text{zero-shot}}\in\mathbb{R}^{d\times k}$ has columns $h(s_j)$ and $f(x)=g(x)^\top\mathbf{W}_{\text{zero-shot}}$ (80 CLIP prompts ensembled; temperature folded into $\mathbf{W}$).
- Step 1, standard fine-tuning of $f(x,\theta)=g(x,\mathbf{V}_{\text{enc}})^\top\mathbf{W}_{\text{classifier}}$, $\theta=[\mathbf{V}_{\text{enc}},\mathbf{W}_{\text{classifier}}]$:
$$\arg\min_\theta\Big\{\sum_{(x_i,y_i)\in\mathcal{S}^{\text{tr}}_{\text{ref}}}\ell(f(x_i,\theta),y_i)+\lambda R(\theta)\Big\}$$
with $\ell$ cross-entropy, $R$ a regulariser. End-to-end changes all of $\theta$; linear classifier (LC) fixes $\mathbf{V}_{\text{enc}}$.
- Step 2 (Eq. 1), $\alpha\in[0,1]$, $\theta_0$ zero-shot, $\theta_1$ fine-tuned:
$$\mathrm{wse}(x,\alpha)=f\big(x,(1-\alpha)\cdot\theta_0+\alpha\cdot\theta_1\big).$$
Algorithm 1: load both state dicts, assert equal keys, set every key to `(1-alpha)*theta_0[key] + alpha*theta_1[key]`.
- Output-space ensemble: $(1-\alpha)\cdot f(x,\theta_0)+\alpha\cdot f(x,\theta_1)$ on unnormalised scores; identical to wse for LC; twice the compute.
- Observation 1 (Eq. 2, empirical, ImageNet and five shifts, all $\alpha$):
$$\mathrm{Acc}_{\mathcal{D},f}((1-\alpha)\theta_0+\alpha\theta_1)\ge(1-\alpha)\mathrm{Acc}_{\mathcal{D},f}(\theta_0)+\alpha\,\mathrm{Acc}_{\mathcal{D},f}(\theta_1).$$
Observation 2: some $\alpha$ beats $\max\{\mathrm{Acc}(\theta_0),\mathrm{Acc}(\theta_1)\}$.
- Proposition 1 (p. 51): if $f(\theta)=f(\theta_0)+\nabla f(\theta_0)^\top(\theta-\theta_0)$ on the whole path, then $f((1-\alpha)\theta_0+\alpha\theta_1)=(1-\alpha)f(\theta_0)+\alpha f(\theta_1)$. The paper adds that in practice the two differ.
- Effective robustness: $\rho(f)=\mathrm{Acc}_{\text{shift}}(f)-\beta(\mathrm{Acc}_{\text{ref}}(f))$, $\beta$ the fitted trend.
- Prediction diversity (p. 47): $\mathrm{PD}=\frac1N\sum_i\mathbb{1}[d_f\vee d_g]$, where $d_f$ = "$f$ right and $g$ wrong", $d_g$ the reverse.
- EMA (pp. 28–29): $\mu_t\leftarrow\beta\mu_{t-1}+(1-\beta)\theta_t$; WiSE-FT with coefficient $1-\beta^T$ on the debiased EMA equals the EMA initialised at $\theta_0$.

#### Hyperparameters (pp. 45–46)

End-to-end CLIP: AdamW ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$), rate $3\times10^{-5}$, cosine schedule, 500 warm-up steps, 10 epochs, weight decay 0.1, clipping at global norm 1, largest batch fitting 8 GPUs (512 for ViT-B/16), CLIP's crop augmentation; head initialised from the zero-shot classifier scaled by the pre-trained temperature, no temperature during training, encoder outputs normalised. Default model ViT-L/14@336px. End-to-end label smoothing: not stated. ALIGN: 40,000 steps, batch 512, maximum rate $5\times10^{-6}$, zero-shot weights temperature-calibrated first. JFT ViT-H/14: 20k iterations, sigmoid cross-entropy. 16-shot: 10 epochs, rate $10^{-5}$. LC: 64 random configurations per regulariser, batch in $\{64,128,256\}$, rate $10^{-\beta}$ with $\beta\sim U[0,6]$. $\alpha$ grid $\{0,0.05,\dots,1\}$. "Optimal $\alpha$" maximises each column on the evaluation data (an oracle); the recommendation is $\alpha=0.5$; fixing it costs 0.0–0.5 pp on averaged columns and up to 2.4 pp on one dataset (Table 3, p. 22).

#### Results

- Table 1 (p. 7), ViT-L/14@336px, ImageNet / average of five shifts: zero-shot 76.6 / 73.4; LC 85.2 / 72.6; end-to-end 86.2 / 68.6; WiSE-FT LC $\alpha=0.5$ 83.7 / 75.9; WiSE-FT end-to-end $\alpha=0.5$ 86.8 / 76.9; optimal 87.1 / 77.4.
- Table 4 (p. 24): shift peak 77.4 at $\alpha=0.35$; ImageNet peak 87.1 at 0.60–0.70; $\alpha=0.4$ gives 86.2 / 77.3, i.e. fine-tuned ImageNet accuracy with +8.7 pp on shifts.
- Table 8 (p. 37), zero-shot → fine-tuned → $\alpha=0.5$: ViT-B/16 68.3/58.1 → 81.3/53.8 → 81.7/63.0; ALIGN 76.4/74.6 → 88.2/75.0 → 86.3/78.4; JFT ViT-H 72.9/65.3 → 85.4/69.8 → 82.9/72.1; BASIC-L 85.6/84.1 → 87.5/76.3 → 87.9/83.4 (shift optimum 85.0 at $\alpha=0.10$–$0.20$, p. 44).
- Table 6 (p. 27): YTBB-Robust 95.8 / 66.5 / 89.7 / 96.0 (zero-shot / fine-tuned / 0.5 / optimal); iWildCam OOD F1 15.5 / 39.9 / 46.1 / 46.4.
- Table 2 (p. 9): reference gains over end-to-end at $\alpha=0.5$ of +0.6 to +2.8 pp on seven datasets.
- Low data, LC only (Figure 17, p. 33, read off): gain over the better endpoint at $\alpha=0.5$ is about +1.0, +4.7, +2.2, 0.0, −0.8 for $k=1,5,10,25,50$ per class; smaller $k$ favours smaller $\alpha$.

#### Ablations, failure modes, limits

- Robustness is hyperparameter-sensitive and invisible from reference accuracy: rates $3\cdot10^{-5}$ and $3\cdot10^{-6}$ differ by 0.3 pp on ImageNet and up to 8 pp on shifts (p. 7).
- Lower rate, fewer epochs, early stopping and $\lambda\lVert\theta-\theta_0\rVert_2^2$ all land on or below the WiSE-FT curve; only weight averaging (WiSE-FT, EMA) reaches it.
- LC alternatives (Figure 13, read off): WiSE-FT about 75.7, distillation about 75.0, regularise-to-zero-shot about 74.7, weight decay / L1 / label smoothing about 73.
- Output-ensembling zero-shot CLIP with an independent NoisyStudent EfficientNet-B6 gives 86.2 / 75.9 against WiSE-FT's 86.8 / 76.9 (Table 13, p. 46).
- Zero-shot and LC predictions are diverse (PD about 0.10–0.18 against 0.04–0.07 for two half-split classifiers); each is more confident where it is better.
- $\alpha=0.5$ costs reference accuracy for ALIGN (−1.9), JFT (−2.5), BASIC-M (−0.6), CLIP LC (−1.5); it ends below zero-shot on YTBB-Robust (−6.1) and BASIC-L shifts (−0.7). For JFT and BASIC-M the reference optimum only ties the fine-tuned model.
- Interpolating networks trained from a random initialisation fails; a shared start is required.

#### Recipe for our project

1. [extrapolation] Interpolation is adapter scaling: $(1-\alpha)W_0+\alpha(W_0+\frac{s}{r}BA)=W_0+\alpha\frac{s}{r}BA$. Exact for LoRA/rsLoRA; not for DoRA (merge first). The serving base must be the training base.
2. [paper for CLIP; extrapolation for a VLM] Start exactly at the zero-shot model ($B=0$; same prompt, format and thinking setting).
3. [paper] Fine-tune normally (AdamW, cosine with warm-up, weight decay 0.1, clip 1), then scale back; do not protect hard rows with a tiny rate or early stopping. [extrapolation] LoRA rate, rank and epochs.
4. [extrapolation] Supervision target for a thinking model is outside the paper and needs its own dev gate.
5. [paper] Default $\alpha=0.5$. [extrapolation] Coarse grid $\{0,0.25,0.5,0.75,1\}$, selection on fused dev MAP@3, $M(\alpha)=\sum_g\pi_gM_g(\theta_\alpha)$; expect the hard-row optimum at smaller $\alpha$ (Table 4: 0.35 against 0.60–0.70).
6. [paper] Barrier monitor per slice: $M_g(\theta_\alpha)\ge(1-\alpha)M_g(\theta_0)+\alpha M_g(\theta_1)$. [extrapolation] Fallback $p(c)\propto p_{\theta_0}(c)^{1-\alpha}p_{\theta_1}(c)^{\alpha}=\mathrm{softmax}((1-\alpha)z_0+\alpha z_1)$, re-derived and correct.

#### Not shown by the paper

LoRA or any adapter; generative, sequence or reasoning models (its stated limitation is image classification only); end-to-end $\alpha$ sweeps at about 1,000 examples; long-tailed labels or unseen label-context pairs; end-to-end rates above $3\cdot10^{-5}$; a held-out protocol for choosing $\alpha$; any guarantee for end-to-end interpolation.

---

### 6.3 Paper 3: STaR, the Self-Taught Reasoner

**Citation.** Eric Zelikman\*, Yuhuai Wu\*, Jesse Mu, Noah D. Goodman. "STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning." Stanford / Google Research. arXiv:2203.14465v2, 20 May 2022 (30 pages).

**Idea.** Turn (question, answer) data with no rationales into rationale training data: the model writes rationale + answer for every training question, only rationales whose answer equals the gold answer are kept, and the *original* pre-trained model is fine-tuned on them; the tuned model regenerates the data and the loop repeats. "Rationalisation" adds failed questions by giving the gold answer as a hint.

#### Method (pp. 3–4)

Pre-trained LLM $M$; data $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{D}$; few-shot prompt set with rationales $\mathcal{P}=\{(x^p_i,r^p_i,y^p_i)\}_{i=1}^{P}$, $P\ll D$ (e.g. $P=10$), prepended to every question. Output: rationale $\hat r_i$, then answer $\hat y_i$.

Algorithm 1 (rationalisation parts: lines 4, 6 and $\cup\,\mathcal{D}^{\text{rat}}_n$; Algorithm 2, p. 26, omits them):

1. $M_0\leftarrow M$
2. for $n=1\dots N$:
3. $(\hat r_i,\hat y_i)\leftarrow M_{n-1}(x_i)\ \ \forall i\in[1,D]$
4. $(\hat r^{\text{rat}}_i,\hat y^{\text{rat}}_i)\leftarrow M_{n-1}(\text{add\_hint}(x_i,y_i))\ \ \forall i\in[1,D]$
5. $\mathcal{D}_n\leftarrow\{(x_i,\hat r_i,y_i)\mid \hat y_i=y_i\}$
6. $\mathcal{D}^{\text{rat}}_n\leftarrow\{(x_i,\hat r^{\text{rat}}_i,y_i)\mid \hat y_i\neq y_i\wedge\hat y^{\text{rat}}_i=y_i\}$
7. $M_n\leftarrow\text{train}(M,\ \mathcal{D}_n\cup\mathcal{D}^{\text{rat}}_n)$ — always from the original $M$, "to avoid overfitting"; $M_{n-1}$ only generates data.

Policy-gradient reading: $p_M(y\mid x)=\sum_r p(r\mid x)\,p(y\mid x,r)$ and

$$J(M,X,Y)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\,\mathbb{1}(\hat y_i=y_i) \tag{1}$$

$$\nabla J(M,X,Y)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\big[\mathbb{1}(\hat y_i=y_i)\cdot\nabla\log p_M(\hat y_i,\hat r_i\mid x_i)\big] \tag{2}$$

The indicator is the filter. STaR approximates $J$ by greedy decoding (lower variance, possibly biased exploration) and several gradient steps on the same batch. Rationalisation samples from $p(r\mid x,y)$ instead of $p(r\mid x)$, "an off-policy estimate" of Eq. 1 (p. 8); no importance weights appear.

Hint format: CQA appends `(CORRECT)` to the gold option in the answer list (the rationalisation few-shot examples carry the same marker); arithmetic writes the gold sum after `Target:`. The stored example uses the unhinted prompt, "as if the model had come up with the rationale without the hint."

#### Hyperparameters (pp. 5, 6, 8, 27)

- GPT-J, 6B: 28 layers, embedding 1024, 16 heads of dimension 256, FFN 16384, vocabulary 50.4K.
- Adam, no weight decay, lr $10^{-6}$ (best of $10^{-7}$–$10^{-4}$), 100-step warm-up then constant; batch 8 sequences of length 1024 with packing; one TPU-v3 node; GPT-J repository fine-tuning script (whether all weights are updated is not stated).
- Steps: 40 in iteration 1, +20% per iteration, $S_n=40\cdot1.2^{\,n-1}$; GSM8K capped at the 30th iteration "after 7912 steps" ($40\cdot1.2^{29}=7912.5$). Arithmetic with rationalisation: 300 steps, +20 per iteration.
- Iterations: arithmetic 16 (with rationalisation), 28 plotted without, 36 in the extra-digit run; GSM8K 36 without rationalisation plus 10 with; CQA not stated ("until we see performance saturate").
- One rationale per question; "low sampling temperature", value not stated. Maximum rationale length not stated.
- Few-shot prompts: CQA text says 10, Appendix B prints 7; arithmetic 10 per digit length (1–5); GSM8K Appendix I prints 6.
- Data: arithmetic 50,000 generated, 10,000 sampled per iteration; CQA 9,741 / 1,221 / 1,285; GSM8K 7,473 / 1,319.

#### Results

| CQA dev (Table 1, p. 7) | Acc. % | Train data used % |
|---|---|---|
| GPT-3 direct fine-tuned (30× larger) | 73.0 | 100 |
| Few-shot direct GPT-J | 20.9 | ~0 |
| Few-shot CoT GPT-J | 36.6 | ~0 |
| Few-shot CoT LaMDA 137B | 55.6 | ~0 |
| GPT-J direct fine-tuned | 60.0 | 100 |
| STaR without rationalisation | 68.8 | 69.7 |
| STaR with rationalisation | 72.5 | 86.7 (78.2 + 8.5) |

| GSM8K test (Table 2, p. 8) | Acc. % | Train data used % |
|---|---|---|
| Few-shot direct / CoT GPT-J | 3.0 / 3.1 | ~0 |
| GPT-J direct fine-tuned | 5.8 | 100 |
| STaR without rationalisation | 10.1 | 25.0 |
| STaR with rationalisation | 10.7 | 28.7 (0.5 rationalised) |

Arithmetic (p. 6): 89.5% overall after 16 iterations; no-rationale baseline 76.3%; few-shot 2-digit <1%, 32% after one iteration. Figure 4 (read off): with rationalisation at iteration 16 about 100/96/91/85/75% for 1–5 digits; without, 5-digit sums take off only after about iteration 23. Figure 5: unseen 9- and 10-digit sums reach roughly 60% and 10%, noisily. Keeping few-shot prompts in fine-tuning: 60.9→68.8 without, 69.9→72.5 with rationalisation (p. 8).

#### Ablations, failure modes, limits

- Rationalisation: +3.7 on CQA, +0.6 on GSM8K ("does not substantially improve performance"); removes the stage-wise delay on arithmetic.
- Rationalised chains can be template matching (Appendix D). Showing hints in training taught the model that the answer equals the hint (A.6).
- More, higher-temperature sampling is "counterproductive": more correct answers despite incorrect reasoning; 0.5 and 0.7 were worse than no rationalisation; 10 samples cost about 10× (p. 9).
- CQA is about 20% correct by chance, about 30% by heuristics; Appendix A shows kept rationales with correct answers and bad reasoning. High-chance settings "yield many poor rationales"; filtering them is open (p. 9).
- Bias amplification, "made worse by rationalization" (Appendix G). Few-shot accuracy must be above chance; GPT-2 failed.

#### Recipe for our project

1. Keep a rollout iff its first code equals the human code **[paper]**; applying this to $N=8$ sampled rollouts **[extrapolation]**, against the paper's finding on more/hotter sampling.
2. Yield: $P(\text{row kept})=1-(1-p_i)^N$, $\mathbb{E}[\text{chains}]=Np_i$; by concavity the slice-mean formula is an upper bound. For $N=8$: $p=0.59\to0.999$, $0.30\to0.942$, $0.10\to0.570$, $0.05\to0.337$ **[extrapolation]**.
3. Lucky passes: $1-(1-1/K)^N$ = 0.10 for $K=76,N=8$ (0.013 for one rollout), 0.83 for $K=5$; require at least $m=2$ of 8 passes, low temperature **[extrapolation]**, motivated by p. 9 **[paper]**.
4. NONE dominance: cap chains per row at 1–2 and hold $n_{\text{NONE}}\le0.51\,n_{\text{error}}$ **[extrapolation]**; the risk is **[paper]**.
5. Rationalise rows with no pass using the `(CORRECT)` marker, keep only if $\hat y^{\text{rat}}=y$, store under the unhinted prompt **[paper]**; drop chains that mention the hint or cite no line of the working **[extrapolation]**.
6. Retrain from base weights every round **[paper]**; LoRA **[extrapolation]**.
7. Steps grow as $S_1\cdot1.2^{\,n-1}$ **[paper]**; $S_1$ chosen on dev **[extrapolation]**.
8. MAP@3 reward $R=\sum_{k=1}^{3}\mathbb{1}[\hat y^{(k)}=y]/k$ as a chain weight **[extrapolation]**, untested.
9. Dev gate against untrained Qwen3.8-27B: hard >0.608, NONE ≥0.924, seen-pair >0.709 **[extrapolation]**.
10. Cost: $8\times1{,}199=9{,}592$ rollouts, about \$47; one rollout per row about \$6 **[extrapolation]**.

#### What the paper does not show

LoRA or adapters; vision inputs; models with native thinking; datasets under 7,000 rows; a dominant class or a 76-way label space; ranking metrics (top-1 only); CQA or GSM8K after a single outer iteration; self-generated versus teacher-distilled rationales; variance across seeds; kept fraction per iteration; out-of-distribution gains beyond arithmetic.

---

### 6.4 Paper 4: LEMMA, learning from errors

**Citation.** Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H. Vicky Zhao, Conghui He, Lijun Wu. "LEMMA: Learning from Errors for MatheMatical Advancement in LLMs." arXiv:2503.17439v2 [cs.LG], 30 May 2025, 25 pp. Code: github.com/pzs19/LEMMA.

**What it is.** A data-construction recipe for fine-tuning 7–8B text solvers on "wrong prefix cut at the first wrong step → reflection phrase → correct solution". It trains a solver to self-correct; it never trains or evaluates an error-type classifier.

#### Core formulation (p.3, p.5, p.8)

$$\tau=(s_1,s_2,\ldots,s_n,\hat a)$$

$$\tau^{b}=(s^{g}_{1},\ldots,s^{g}_{k},s^{b}_{1},\ldots,s^{b}_{m},\hat a^{b})$$

$$\tau^{b}_{sub}=(s^{g}_{1},\ldots,s^{g}_{k},s^{b}_{1}),\qquad \tau^{r}=\tau^{b}_{sub}\oplus RP\oplus\tau^{g}$$

$s^g$ correct steps, $s^b$ incorrect steps, $\hat a$ predicted answer, $RP$ reflection phrase, $\tau^g$ a correct trajectory, $\oplus$ concatenation. Training set $\{\tau^r\}_{i=1}^{N}$. No loss equation or algorithm box is given. LEMMA-Hard allocation:

$$n=f\times k_{max},\qquad k_{max}=20$$

with $f$ the student's failure rate on a question and $n$ the trajectories synthesised for it; half of the errors self-generated, half teacher-injected (p.8).

#### Pipeline (p.4–6; prompts p.23–25)

1. Student (LLaMA3-8B) solves training questions at temperature 0.7; wrong solutions kept.
2. Errors are classified by GPT-4o into a 12-type taxonomy (CA, CO, CV, HA, UC, OP, FC, MS, CS from Li et al. 2024d; QM, CC, NO added). An error-type distribution is determined "for each question"; how is not stated.
3. Teacher (GPT-4o) injects an error of a type sampled from that distribution. Injection prompt (Fig. 13): careless-student role; question + ground-truth solution; `{Error_Type}` + `{Error_Description}`; plausible not blatant; do not name the type; continue to a wrong final answer without repairing; fixed answer sentence; `<explanation>` of why it is wrong and why it is that type.
4. Each error is corrected once by each strategy. Both prompts: diligent-student role, five numbered instructions, 13 transition phrases, copy the attempt up to and including the first wrong step, never mention the attempt or ground truth, output unchanged if the attempt is actually correct. **Fix & Continue** (Fig. 14; has the ground-truth field): name the error, correct it, continue. **Fresh & Restart** (Fig. 15; no ground-truth field): reflect, then solve by an alternative approach.
5. Teacher smooths the trajectory; only trajectories with a correct final answer are kept.

Measured first-error distribution, greedy decoding (types under 1% omitted): MATH (Fig. 3, p.5) LLaMA3-8B QM 48.6 / FC 25.3 / CA 15.3 / CC 8.6 / MS 2.2; Qwen2-Math-7B 40.1 / 26.3 / 18.1 / 12.9 / 2.7. GSM8K (Fig. 7, p.14) LLaMA3-8B QM 41.6 / CA 24.5 / CC 20.9 / MS 6.8 / FC 6.3. Stable across models, different across datasets. Temperature (Fig. 4a): total errors ≈125 (0.3), ≈130 (0.5), ≈145 (0.7), ≈190 (0.9), ≈235 (1.1); nonsensical output ≈0 up to 0.7, ≈30 at 0.9, ≈70 at 1.1.

#### Hyperparameters (p.6, p.15)

| Item | Value |
|---|---|
| Student / teacher | LLaMA3-8B at T=0.7 / GPT-4o (alt. LLaMA-3.1-Nemotron-70B, also used for MetaMath questions) |
| Synthesis sampling | nucleus, T=0.7, top_p 1.0 |
| Data | MATH + GSM8K train (14.97k) → 88.90k trajectories (mean 623.38 tokens); with MetaMath 403.59k (544.14); LEMMA-Hard 90.7k |
| Training | LLaMA-Factory, lr 1e-5, warmup ratio 0.03, cosine, gradient accumulation 8, 3 epochs |
| Evaluation | ≤1,024 new tokens; T=0 for Pass@1; T=0.7, top_p 1.0 for Majority@32 |
| Hardware | 8 × A100, ≈5 h for LLaMA3-8B |
| Not stated | LoRA vs full, batch size, optimiser, weight decay, max length, self/teacher error split, injections per question, error-to-clean ratio |

#### Headline results (Pass@1 %, seven-set Avg. unless noted)

| Method (samples), Table 2 p.7 | LLaMA3-8B GSM8K / MATH / Avg. | DeepSeekMath-7B |
|---|---|---|
| SFT (14.97k) | 65.5 / 19.3 / 49.1 | 68.1 / 35.2 / 58.4 |
| RFT (86.52k) | 67.3 / 21.1 / 50.9 | 73.3 / 39.3 / 62.3 |
| GPTAug (88.62k) | 72.1 / 31.8 / 58.3 | 77.8 / 45.5 / 65.2 |
| RefAug-90k (89.92k) | 77.4 / 34.2 / 60.1 | 76.7 / 42.5 / 63.8 |
| LEMMA (88.90k) | 79.2 / 38.3 / 62.4 | 80.4 / 50.6 / 70.1 |
| LEMMA + MetaMath (403.59k) | 86.4 / 42.3 / 65.5 | 83.0 / 51.7 / 71.4 |

- MathChat (Table 3, p.7): LEMMA 83.4 / 49.4 / 43.4 follow-up rounds, 84.7 error correction; RefAug-90k 83.8 on error correction (gap 0.9).
- Table 4 (p.8; Avg. = mean of GSM8K and MATH only): LEMMA 79.2 / 38.3 / 58.7; DART-Math-Hard 90.7k 81.7 / 42.6 / 62.1; LEMMA-Hard 90.7k 83.1 / 44.8 / 63.9; DART-Math-Hard 585k 81.1 / 46.6 / 63.8.
- Table 6 (p.16): LEMA 75.1 / 32.8; Mistake tuning 68.9 / 25.5; LLM2LLM 76.8 / 33.1.
- Nemotron teacher (Table 8, p.20): Avg. 61.3 (LLaMA3-8B), 67.9 (DeepSeekMath).
- Other bases (Table 9, p.20): Mistral-7B 60.2 (SFT 40.5); Qwen2-Math-7B 77.2 (SFT 68.0).
- Majority@32 (Table 10, p.21): LLaMA3-8B GSM8K 90.3, MATH 53.1; DeepSeekMath MATH 63.6.
- Error counts, LLaMA3 on MATH, base / SFT / LEMMA (Fig. 6, p.9): QM ≈1,600 / 1,590 / 1,180; FC ≈830 / 1,160 / 690; CA ≈510 / 630 / 420.

#### Ablations, failure modes, limits (Table 5 p.9, Table 11 p.21; LLaMA3-8B)

| Variant | GSM8K | MATH | Avg. |
|---|---|---|---|
| LEMMA | 79.2 | 38.3 | 62.4 |
| w/o error augmentation | 72.8 | 28.8 | 56.7 |
| same, 90k | 74.3 | 32.1 | 58.6 |
| w/o Fresh & Restart | 75.2 | 34.4 | 59.8 |
| same, 90k | 75.7 | 34.1 | 59.9 |
| w/o truncation | 69.4 | 22.8 | 54.0 |

- No truncation is worst (−15.5 MATH) and falls below correct-only GPTAug (58.3).
- Data size (Fig. 5, p.8): LEMMA on MATH ≈29 at ≈9k, ≈36 at ≈45k, 38.3 at 89k; nothing below ≈9k.
- Stated limitations: under 90k examples; rule-based RL left for later.
- Unstated weaknesses: no human check of the GPT-4o type classifier or of injection fidelity; no seeds or significance tests; Fig. 11 example 1 (p.19) derives $25\sqrt2$, then 50, then answers 25, so the answer filter admits incoherent reflections.

#### Recipe for our project

- Sample injected codes from a measured distribution per question type, never uniformly. [paper] Using our 75 codes as the taxonomy. [extrapolation]
- Injection prompt: question, correct solution, code name + definition, plausible not blatant, do not name the error, carry it to a wrong final answer, tagged explanation of why it is wrong and why that type. [paper] Also naming the first wrong line. [extrapolation]
- Generate at temperature ≈0.7, not higher. [paper]
- Mix synthetic with real rows; more rows where the model fails, $n=f\times k_{max}$. [paper, for a solver] For a classifier. [extrapolation]
- Blind verification that a second model recovers the intended code and the answer is wrong. [extrapolation]
- Synthetic correct workings so NONE keeps its real share. [extrapolation]
- Diagnosis chain anchored on the first wrong line. [paper: largest ablation effect] Mechanism naming, forward check, code mapping, root-line filter. [extrapolation]
- Two chain styles per row (repair-first, solve-first). [paper in form; extrapolation to diagnosis]
- Keep only chains ending at the gold label. [paper: answer filter]
- Trajectories built from our model's own wrong diagnoses, cut at the first wrong inference. [extrapolation]

#### What the paper does NOT show

- Any result for classifying error types; the type is never a label and never verified.
- Whether type-conditioned sampling beats unguided teacher injection (not ablated; teacher-only errors and removal of Fix & Continue also not run).
- Any error-to-clean ratio or ablation of it; whether the 14.97k originals are mixed in.
- Realism of injected errors versus real ones; children's errors (these are LLM errors, QM 35–49%).
- Anything below ≈9k samples, for vision models, thinking models, or models above 8B.

---

### 6.5 Paper 5: ensemble debiasing with a bias-only model (Clark et al.)

**Citation.** Christopher Clark, Mark Yatskar, Luke Zettlemoyer. "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases." arXiv:1909.03683v1, 9 Sep 2019, 14 pp. Code: github.com/chrisc36/debias.

**Idea.** Train a bias-only model $h$ on known shortcut features, freeze it, then train the main model $f$ inside an ensemble with it; gradients flow through $f$ only (Fig. 1, p. 2). "During evaluation $f$ is used alone" (p. 3).

#### Method (pp. 3-5)

Examples $x_1..x_n$, labels $y_i \in \{1..C\}$; $h(x_i) = b_i = \langle b_{i1},\dots,b_{iC}\rangle$ (bias probabilities); $f(x_i,\theta) = p_i$. Loss is computed on the combined $\hat p_i$ (cross-entropy implied, not printed).

- **Reweight baseline (4.1):** no ensemble; example weight $1 - b_{i y_i}$.
- **Bias Product (3.2.3):** $$\hat p_i = \mathrm{softmax}(\log p_i + \log b_i) \;\propto\; p_i \circ b_i$$ Justification, with bias features $x^b$ and the rest $x^{-b}$ conditionally independent given $c$: $$p(c\mid x) \propto p(c\mid x^{-b})\,\frac{p(c\mid x^{b})}{p(c)} \quad (5)$$ The $p(c)$ factor is dropped (uniform classes on MNLI; pointer outputs on QA; class priors unwanted on VQA). A view $x^{-b}$ usually cannot be built, hence the trained product.
- **Learned-Mixin (3.2.4):** $$\hat p_i = \mathrm{softmax}(\log p_i + g(x_i)\log b_i),\quad g(x_i) = \mathrm{softplus}(w\cdot h_i),\quad \mathrm{softplus}(x)=\log(1+e^x)$$ $w$ a learned vector, $h_i$ the main model's last hidden layer; $g=1$ recovers Bias Product. Failure: the model absorbs the bias into $p_i$ and sets $g=0$.
- **Learned-Mixin +H (3.2.5):** add $$R = w\,H\big(\mathrm{softmax}(g(x_i)\log b_i)\big),\quad H(z) = -\sum_j z_j\log z_j$$ with $w$ here a scalar hyperparameter.
- **Bias model:** trained first on the same training set; footnote 2 flags overfitting but says the model is usually simple enough. Cross-fitting is never mentioned (in-sample is our reading).
- **VQA (5.2):** each answer treated as a two-class problem (no formula printed; our derivation: $\hat p_{ij} = \sigma(\mathrm{logit}\,p_{ij} + g\,\mathrm{logit}\,b_{ij})$). Footnote 4: a learned $\sigma(\alpha)$ is added to the bias probabilities because some are zero.
- **QA:** separate bias models for start and end tokens; $g$'s hidden state is a ReLU layer on question and passage embeddings, max-pooled.

#### Hyperparameters

Entropy weight $w$ (Table 7, p. 12): Synthetic Indicator 0.01, Excluder 0.005, Dependent 0.005; VQA-CP 0.36; HANS 0.03 (recurrent and BERT); Adversarial SQuAD 2.0 (both bias models); TriviaQA-CP Location 0.4, Person 0.2. **Selected on the test set** (4.2). Co-attention NLI model: variational dropout 0.2, 30 epochs, Adam, batch 32, learning rate decayed by 0.999 every 100 steps, 200-d LSTMs, 50-d fully connected. Modified BiDAF: dropout 0.2, Adam, batch 45, same decay; SQuAD 200 FC / 100 LSTM; TriviaQA 256 FC / 128 LSTM, highway connections, recurrent dropout 0.2. BERT base uncased: defaults. No early stopping or further tuning. Not stated: initial learning rates, QA epochs, VQA hyperparameters, the grid for $w$. Bias models: first hypothesis token (synthetic); one of 65 question types (VQA-CP); linear classifier on five word-overlap features (HANS); TF-IDF sentence scores, plain and filtered (SQuAD); NER tags (TriviaQA-CP).

#### Results, main model alone (out-of-domain / in-domain)

| Setting | None | Reweight | Bias Product | Learned-Mixin | LM +H |
|---|---|---|---|---|---|
| Synthetic Indicator (T2, p. 7) | 69.36 / 86.49 | 75.44 / 82.74 | 76.27 / 81.32 | 76.29 / 81.35 | 76.77 / 77.65 |
| Excluder | 68.06 / 83.56 | 70.36 / 83.29 | 77.33 / 80.41 | 77.80 / 78.86 | 77.90 / 78.57 |
| Dependent | 63.23 / 87.90 | 69.81 / 85.50 | 71.85 / 84.98 | 75.75 / 77.70 | 75.79 / 76.65 |
| VQA-CP v2 (T3, p. 7; no in-domain) | 39.18 | 40.06 | 39.93 | 48.69 | 52.05 |
| HANS / MNLI, co-attention (T4) | 50.58 / 78.73 | 52.85 / 77.03 | 53.69 / 76.63 | 51.65 / 78.05 | 53.35 / 74.50 |
| HANS / MNLI, BERT | 62.40 / 84.24 | 69.19 / 83.54 | 67.92 / 82.97 | 64.00 / 84.29 | 66.15 / 83.97 |
| SQuAD AddSent / dev, filtered TF-IDF (T5, p. 9) | 42.54 / 80.61 | 41.55 / 80.59 | 47.17 / 78.63 | 42.25 / 80.39 | 51.84 / 75.94 |
| SQuAD AddSent / dev, TF-IDF | 42.54 / 80.61 | 42.74 / 80.51 | 44.41 / 78.22 | 42.00 / 80.46 | 48.30 / 74.14 |
| TriviaQA-CP Location CP / dev (T6) | 41.23 / 59.27 | 40.14 / 59.18 | 44.42 / 60.02 | 41.15 / 61.64 | 47.77 / 57.74 |
| TriviaQA-CP Person | 39.69 / 55.26 | 39.96 / 55.38 | 40.58 / 55.20 | 41.31 / 56.08 | 44.37 / 54.83 |

Unbiased-training upper bound on synthetic MNLI: 78.94 / 78.94 / 78.86. Prior VQA-CP work: 41.17, 42.33. Table 8 (p. 13, mean of 8 runs), co-attention on HANS rows where the shortcut is right (lexical / subsequence / constituent): None 97.83 / 99.67 / 97.28, LM +H 67.18 / 61.05 / 47.13; rows where it is wrong: 1.37 / 3.68 / 3.68 to 27.41 / 56.82 / 60.53. BERT with LM +H keeps 91.98 / 98.20 / 97.98 on entailment.

#### Ablations, failure modes, limits

- Bias Product over-compensates when independence fails (Dependent, p. 6): on rows whose indicator is 1 it gives the bias-indicated class 22.5% against 33% for an unbiased model; Learned-Mixin gives 34.5%, with $g = 0.00 \pm 0.0001$ when the indicator is turned off and $1.91 \pm 0.285$ otherwise.
- Without the penalty the gate collapses: mean $g$ 0.13 against 5.16 (AddSent, filtered bias); 0.25 against 5.01 (TriviaQA Location dev). Plain Learned-Mixin gains nothing on SQuAD or TriviaQA Location.
- Gate tracks reliability: Spearman 0.77 on VQA-CP test between $g$ and $\sum_j s_{ij}b_{ij}/\sum_j b_{ij}$; Fig. 2 (p. 8) 4.65 / 5.96 for a typical answer, 0.00 / 0.48 for an atypical one.
- In-domain cost is the usual outcome (5.6), TriviaQA-CP the exception: ensemble methods lose 2.9 to 11.3 points on synthetic MNLI; SQuAD dev loses 2.0 to 6.5 F1 where the method helps.
- Reweight needs bias-free examples (Excluder gain 2.3); it is best for BERT on HANS (69.19). BERT runs had high variance.
- Section 5.6: where the bias already answers correctly the main model has little to learn, which effectively shrinks the training set.

#### Recipe for our project

- [extrapolation] Cross-fit the bias model: $b_i(c) = \dfrac{n^{(-i)}_{q_i,c} + \alpha\,\pi^{(-i)}_{t_i}(c)}{n^{(-i)}_{q_i} + \alpha}$, floor at $\epsilon$, renormalise. In-sample counts give $b_{iy_i}=1$ on single-row questions, so $\partial\mathcal{L}_i/\partial z_{ij} = \hat p_{ij} - \mathbb{1}[j=y_i] = 0$. Softening zeros is [paper] (footnote 4); the floor and topic prior are ours.
- [paper] Loss forms, in order of risk: Reweight $1-b_{iy_i}$; Bias Product $-\log\mathrm{softmax}(z_i+\log b_i)_{y_i}$; Learned-Mixin +H. [extrapolation] Applying them to 1,500 rows and 76 unbalanced codes; dropping $p(c)$ leaves the main model without the class prior.
- [extrapolation] Choose $w$ on cross-fitted dev hard rows across 0.005 to 2.0.
- [paper] direction, [extrapolation] size: use the debiased model only as a hard-row specialist in fusion; alone it gives up the bias-aligned rows. Optionally re-attach $\mathrm{softmax}(\log p_i + \lambda\log b_i)$ [extrapolation].
- [paper] The method is training-time only; subtracting a prior from an already-trained model reproduces the Dependent anti-correlation failure.

#### What the paper does not show

The ensemble $\hat p_i$ at test time; any test set where the bias still holds for most rows; an honest (non-test) choice of $w$ or a sweep, or any link between $w$ and the in-domain drop; cross-fitted bias predictions; training sets near 1,500 rows (stated sizes 60,133 and 52,953); a bias model that is too strong; in-domain VQA accuracy; whether the main model has the skill to solve rows once the shortcut is removed.

---

### 6.6 Paper 6: Product of Experts and Debiased Focal Loss

**Citation.** Rabeeh Karimi Mahabadi, Yonatan Belinkov, James Henderson. "End-to-End Bias Mitigation by Modelling Biases in Corpora." arXiv:1909.06321v3, 23 April 2020. Code: `github.com/rabeehk/robust-nli`.

**Idea.** A bias-only model $f_B$ sees only shortcut features $x_i^b$ (NLI hypothesis, FEVER claim, hand-built syntactic features for HANS) and is trained beside the base model $f_M$. Its predictions reshape the base model's loss. $f_B$ is dropped at test time: $\hat y=\arg\max_k f_M^k(x)$.

#### Equations (p. 3-5)

$\mathcal{D}=\{x_i,y_i\}_{i=1}^N$; $f_M:\mathcal{X}\to\mathbb{R}^{|\mathcal{Y}|}$ (parameters $\theta_M$); $f_B:\mathcal{X}^b\to\mathbb{R}^{|\mathcal{Y}|}$ (parameters $\theta_B$); $f^j$ is the $j$th element; $\sigma(u^j)=e^{u^j}/\sum_{k=1}^{|\mathcal{Y}|}e^{u^k}$.

Bias-only loss (Eq. 1):
$$\mathcal{L}_B(\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\log\big(\sigma(f_B^{y_i}(x_i^b;\theta_B))\big)$$

PoE (Eq. 2):
$$f_C(x_i,x_i^b)=\log\sigma(f_B(x_i^b))+\log\sigma(f_M(x_i)),\qquad \mathcal{L}_C(\theta_M;\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\log\sigma(f_C^{y_i}(x_i,x_i^b))$$
$$\sigma(f_C^{y_i})=\frac{\sigma(f_B^{y_i}(x_i^b))\,\sigma(f_M^{y_i}(x_i))}{\sum_{k}\sigma(f_B^{k}(x_i^b))\,\sigma(f_M^{k}(x_i))}$$
$$\nabla_{\theta_M}\mathcal{L}_C=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{|\mathcal{Y}|}\big(\delta_{y_ik}-\sigma(f_C^{k}(x_i,x_i^b))\big)\nabla_{\theta_M}\log\sigma(f_M^{k}(x_i))$$
Limits stated: bias-only model correct $\Rightarrow\sigma(f_C^{y_i})=1$, gradient 0; bias-only uniform ($1/|\mathcal{Y}|$) $\Rightarrow$ plain cross-entropy.

DFL (Eq. 3), $\gamma=0$ is cross-entropy:
$$\mathcal{L}_C(\theta_M;\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\big(1-\sigma(f_B^{y_i}(x_i^b))\big)^{\gamma}\log\sigma(f_M^{y_i}(x_i))$$

RUBi baseline ($\phi$ sigmoid): $f_C=f_M(x_i)\odot\phi(f_B(x_i^b))$, cross-entropy on $f_C$.

Several biases: joint PoE $f_C=\sum_{j=1}^{K}\log\sigma(f_{B_j}(x_i^{b_j}))+\log\sigma(f_M(x_i))$; joint DFL uses $f_B=\frac1K\sum_{j}f_{B_j}(x_i^{b_j})$ in Eq. 3. No classifier on concatenated bias features.

Gradient flow: one-stage joint training. Stated: $\mathcal{L}_B$ "is not back-propagated to any shared parameters of the base model". $\theta_M$ is trained from $\mathcal{L}_C$. Whether $\mathcal{L}_C$ reaches $\theta_B$ is not stated in words; Figure 1 (p. 2) draws no backward arrow from the combination model to the bias-only model. No single total-objective formula is printed.

#### Hyperparameters (p. 4, 7, 8, 12, 13)

- BERT (HuggingFace; base/large not stated): 3 epochs, default parameters, learning rate $2\mathrm{e}{-5}$ (stated for FEVER). InferSent: 20 epochs, no early stopping, BiLSTM 512, classifier 512 and 512 hidden units, Tanh.
- Bias-only classifier: BERT runs, "shallow nonlinear classifier with 768, 384, 192 hidden units with Tanh"; InferSent, "shallow linear classifier with 512 and 512 hidden units"; HANS, 6 and 6 hidden units, Tanh, on five feature groups.
- $\gamma=2$ "through all experiments". HANS-tuned: sweep $\{0.5,1,2,3,4\}$, selected $\gamma=3$ on the test set.
- Transfer: BERT $\gamma,\alpha\in\{0.4,0.6,0.8,2,3,4,5\}$; InferSent $\gamma,\alpha\in\{0.02,0.05,0.1,0.6,2.0,4.0,5.0\}$, $\beta\in\{0.05,0.2,0.4,0.8,1.0\}$. Formulae for $\alpha$ (PoE strength) and $\beta$ (bias-only loss weight) and the selected values are not stated.
- Batch size, optimiser, warm-up, seeds: not stated.

#### Results (in-domain / bias-free set, $\Delta$ vs CE)

| Setting | CE | PoE | DFL ($\gamma=2$) |
|---|---|---|---|
| FEVER dev / symmetric, BERT (T1, p. 5) | 85.99 / 56.49 | 86.46 / 66.25 (+9.8) | 83.07 / 64.02 (+7.5) |
| SNLI test / hard, BERT (T2, p. 6) | 90.53 / 80.53 | 90.11 / 82.15 (+1.6) | 89.57 / 83.01 (+2.5) |
| SNLI test / hard, InferSent | 84.24 / 68.91 | 80.35 / 73.69 (+4.8) | 73.54 / 73.05 (+4.1) |
| MNLI-mm dev / hard, BERT (T3, p. 6) | 84.53 / 77.55 | 84.85 / 79.23 (+1.7) | 84.85 / 78.92 (+1.4) |
| MNLI-mm dev / hard, InferSent | 69.99 / 56.53 | 65.85 / 59.14 (+2.6) | 61.12 / 59.05 (+2.5) |
| MNLI-m dev / hard, BERT (T7, p. 12) | 84.41 / 76.56 | 84.58 / 78.02 (+1.5) | 83.72 / 77.37 (+0.8) |
| MNLI-m dev / HANS, BERT (T4, p. 6) | 84.51 / 61.88 ± 1.9 | 84.19 / 66.31 ± 0.6 (+4.4) | 83.95 / 69.26 ± 0.2 (+7.4) |

DFL $\gamma=3$ (tuned): 82.76 / 71.95 ± 1.4 (+10.1). RUBi: −0.1 to +1.5 everywhere. Clark et al. Reweight (= DFL with $\gamma=1$): 83.54 / 69.19.

Transfer from SNLI, BERT (T6, p. 8): +0.2 to +3.8 on ten sets; MNLI −0.8 / −0.8; MNLI-M 0.0 / −0.5 (DFL / PoE). InferSent (T9, p. 13): mean relative change DFL +2.57%, PoE +3.39%.

#### Ablations, failure modes, limits

- In-domain cost: BERT within about one point except DFL on FEVER (−2.92) and tuned $\gamma=3$ on MNLI (−1.75); InferSent −3.89 to −10.70. The paper says the removed biases "are useful for performing the in-domain task".
- HANS rows where the heuristic is right (T8, p. 12; constituent / lexical / subsequence): CE 98.98 / 96.41 / 99.72; DFL 90.90 / 84.78 / 94.33; PoE 97.24 / 92.16 / 98.58. Heuristic wrong: CE 20.12 / 48.86 / 7.18; DFL 50.20 / 71.06 / 24.28; PoE 36.08 / 59.18 / 14.63.
- $\gamma$ curves (Figures 2, 4, read off plots): InferSent SNLI test about 84 → 73 ($\gamma=2$) → 50 ($\gamma=6$) while hard stays about 70-73 for $\gamma$ 1-3.5; BERT test about 90.8 → 89.6 (2) → 87.5 (3) → 82.5 (7), hard peaks near 83 at $\gamma=2$-3.
- Joint debiasing (T5, p. 7): hypothesis-debiased models lose 1.2-1.5 on HANS; syntactic-debiased models gain only +0.1 to +0.4 on the hard set; joint PoE +1.1 hard / +6.2 HANS, joint DFL +0.8 / +7.2.
- Figure 3 (approximate): loss correlation with the bias-only model, baseline / RUBi / DFL / PoE: SNLI 0.37 / 0.37 / 0.29 / 0.28; MNLI 0.275 / 0.28 / 0.245 / 0.23; HANS 0.725 / 0.72 / 0.52 / 0.645. The lower-correlation-scores-higher claim is made only for MNLI and HANS.
- The bias must be known in advance.

#### Recipe for our project

1. **PoE training logits** [paper]: $z_i=\log\mathrm{softmax}(f_M(x_i))+\log\mathrm{softmax}(f_B(q_i))$, cross-entropy on $z_i$; per-row gradient $\partial\mathcal{L}_i/\partial f_M^k=\mathrm{softmax}(z_i)^k-\delta_{y_ik}$.
2. **Test with $f_M$ alone** [paper]; identity [extrapolation]: $\mathrm{softmax}(z_i-\log b_i)=\mathrm{softmax}(f_M(x_i))$, so $f_M^k\approx\log\frac{P(k\mid x,q)}{P(k\mid q)}+c_i$.
3. **DFL weights** $w_i=(1-b_i^{y_i})^\gamma$ [paper]; slice arithmetic [extrapolation]: $b=0.7\Rightarrow w=0.09$ at $\gamma=2$; $b=0.3\Rightarrow0.49$; $b=0.01\Rightarrow0.98$; mean weight about 0.53 on our dev mix, so renormalise to mean 1. PoE on the same no-error row gives gradient 0.3 against 0.5 for cross-entropy and 0.045 for DFL, if the prior's other 0.3 sits on the competitor. Start at $\gamma\in\{1,2\}$.
4. **Cross-fit the prior, no joint $f_B$** [extrapolation]: $b_i^k=\frac{n_{q_i,k}-\mathbb{1}[k=y_i]+\lambda\pi_k}{n_{q_i}-1+\lambda}$, folded over rows. An in-sample question model converges to $n_{q,y}/n_q$, which counts the row's own label and equals 1 on single-code questions. A frozen precomputed $b$ is the Clark et al. two-stage form, which the paper says "gets similar results" [paper, indirect].
5. **Deploy as a hard-row specialist only** [extrapolation]: fusion member, or $\mathrm{softmax}(f_M+\alpha\log b)$ with $\alpha$ chosen on dev by slice; do not trust its NONE decision. Needs a GPU run and the user's approval.
6. **Risk** [extrapolation]: re-weighting does not add evidence; hard rows may need new evidence.

#### What the paper does not show

- Any gain on a test set where the bias still holds: every headline gain is on sets built so the bias is absent or wrong, and in-domain accuracy falls or stays flat.
- Small-data behaviour, or a bias-only model able to memorise the training set; no cross-fitting.
- The bias-only model's accuracy, the in-domain accuracy of the ensemble $f_C$, or variance outside the HANS experiments.
- Results beyond 3-class NLI and fact verification with BERT and InferSent; nothing multimodal or with 76 classes.

---

### 6.7 Paper 7: forward and backward loss correction

**Paper 7.** Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, Lizhen Qu. "Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach." arXiv:1609.03683v2 [stat.ML], 22 March 2017, 9 pages. Venue is not printed in the PDF.

#### Method (pages 3–5)

- **Noise model.** Class-conditional only: $p(x,\tilde y)=\sum_y p(\tilde y\mid y)\,p(y\mid x)\,p(x)$, with $T\in[0,1]^{c\times c}$, $T_{ij}=p(\tilde y=e^j\mid y=e^i)$ (rows = true class, columns = noisy class; row-stochastic, not necessarily symmetric). $\hat p(y\mid x)$ is the softmax output; $\boldsymbol\ell(\hat p)=(\ell(e^1,\hat p),\dots,\ell(e^c,\hat p))^\top$ is the loss evaluated at every label.
- **Backward (Theorem 1).** For non-singular $T$: $\boldsymbol\ell^{\leftarrow}(\hat p(y\mid x))=T^{-1}\boldsymbol\ell(\hat p(y\mid x))$. Unbiased: $\mathbb E_{\tilde y\mid x}\ell^{\leftarrow}=\mathbb E_{y\mid x}\ell$, so the minimisers coincide. The loss can be negative; for a badly conditioned $T$ the paper suggests to "mix $T$ with the identity matrix before inversion".
- **Forward (Theorem 2).** For cross-entropy,
$$\ell(e^i,\hat p(y\mid x))=-\log\sum_{j=1}^{c}T_{ji}\,\hat p(y=e^j\mid x),\qquad \boldsymbol\ell(\hat p(y\mid x))=-\log T^{\top}\hat p(y\mid x).$$
It is $T^{\top}$ applied to the prediction, inside the log (the introduction and Table 1 loosely say "$T\cdot$"). In general, for a proper composite loss with link $\psi$ and non-singular $T$: $\boldsymbol\ell^{\rightarrow}_\psi(h(x))=\boldsymbol\ell(T^{\top}\psi^{-1}(h(x)))$ and $\operatorname{argmin}_h\mathbb E_{x,\tilde y}\ell^{\rightarrow}_\psi(y,h(x))=\operatorname{argmin}_h\mathbb E_{x,y}\ell_\psi(y,h(x))$. The proof uses the link $\phi=(T^{-1})^{\top}\circ\psi$ (as printed), giving minimiser $\psi((T^{-1})^{\top}p(\tilde y\mid x))=\psi(p(y\mid x))$. The guarantee covers the minimiser only; there is no explicit inversion.
- **Estimating $T$ without clean labels (Theorem 3).** Assume a "perfect example" $\bar x^j$ with $p(y=e^j\mid\bar x^j)=1$ for each class and a network that models $p(\tilde y\mid x)$ accurately. Then $T_{ij}=p(\tilde y=e^j\mid\bar x^i)$. Estimator on any unlabelled set $X'$:
$$\bar x^i=\operatorname*{argmax}_{x\in X'}\hat p(\tilde y=e^i\mid x),\qquad \hat T_{ij}=\hat p(\tilde y=e^j\mid\bar x^i),$$
cost $O(c^2\cdot|X'|)$. In practice: $X'$ = training + validation, a 97th percentile replaces the argmax (plain argmax on CIFAR-100), then rows are normalised.
- **Algorithm 1 (two-stage).** Input: noisy set $\mathcal S$, loss $\ell$. If $T$ is unknown: train $h$ on $\mathcal S$ with $\ell$; obtain unlabelled $X'$; estimate $\hat T$. Then train $h$ on $\mathcal S$ with $\ell^{\leftarrow}$ or $\ell^{\rightarrow}$. If $T$ is known, "for example by cleaning manually a subset of training data", skip estimation. The first network may initialise the second.
- **Theorem 4.** With ReLU networks (linear-odd losses), the Hessians of $\ell$ and $\ell^{\leftarrow}$ are unchanged by noise for any $T$. No such guarantee for forward, and no assurance on minima.

#### Hyperparameters (pages 6–7)

Common: cross-entropy; 10% of training held out, validated on noisy data with the corrected loss; ReLU; He initialisation, otherwise uniform $[-0.05,0.05]$; batch 128. Noise level $N$: true class keeps $1-N$.

- MNIST: 2 dense layers of 128, dropout 0.5; flips 2→7, 3→8, 5↔6, 7→1; AdaGrad, 40 epochs, lr 0.01, $\delta=10^{-6}$; 5 runs.
- IMDB: (a) 50-d embedding, dropout 0.8; (b) 256-d embedding, LSTM 512, 512-d hidden layer, dropout 0.5; top 5,000 words, 400 words per review; 0→1 fixed at 5%, 1→0 at $N$; AdaGrad, 50 epochs; 5 runs.
- CIFAR-10 (pre-activation ResNet-14/32) and CIFAR-100 (ResNet-44, each class flips to the next within its 5-class super-class): SGD, momentum 0.9, lr 0.01, ÷10 after epochs 40 and 80 of 120 (80 and 120 of 150 for CIFAR-100), weight decay $10^{-4}$; single runs.
- Clothing1M: ImageNet-pretrained ResNet-50, 14 classes, 1M noisy plus 50k/14k/10k clean; SGD, lr $10^{-3}$ then $10^{-4}$ for 5 epochs each, momentum 0.9, batch 32; weight decay $5\cdot10^{-2}$ plus augmentation on 50k, $10^{-3}$ on 1M. $T$ is taken from the 50k curated labels and their noisy versions, not from Eqs. 12–13.

#### Results (Table 2 and Table 3, page 8; accuracy %)

| Asymmetric, high noise | cross-entropy | forward, true $T$ | forward, $\hat T$ | backward, true $T$ |
|---|---|---|---|---|
| MNIST ($N=0.6$) | 53.0±0.6 | 97.3±0.0 | 64.9±4.4 | 93.0±0.9 |
| IMDB embedding ($N=0.4$) | 58.1±0.5 | 80.9±1.3 | 73.0±1.2 | 82.1±0.1 |
| IMDB LSTM ($N=0.4$) | 71.4±1.3 | 84.7±0.6 | 84.2±0.9 | 83.6±1.4 |
| CIFAR-10 ResNet-14 ($N=0.6$) | 57.6 | 84.5 | 74.8 | 75.2 |
| CIFAR-10 ResNet-32 ($N=0.6$) | 53.6 | 87.6 | 77.6 | 74.3 |
| CIFAR-100 ResNet-44 ($N=0.6$) | 17.1 | 68.4 | 15.9 | 36.8 |

- Low asymmetric noise, forward with $\hat T$ against cross-entropy: +0.2 to +2.0 points. Symmetric noise: −1.1 to +1.3. No noise: −0.4 to +0.4.
- Clothing1M: cross-entropy on 1M 68.94; backward 69.13; forward 69.84; cross-entropy on clean 50k 75.19; forward model fine-tuned on 50k with cross-entropy (#8, same learning parameters as #7) **80.38**.
- Authors: estimating $T$ costs on average between 0 (IMDB LSTM) and 27 points (MNIST), excluding CIFAR-100, with "a median drop of only 10 points" (pages 7–8).

#### Failure modes and limits

- Backward is unstable: negative losses, coefficients "far by orders of magnitude"; at low asymmetric noise it loses to cross-entropy on ResNet-32 (84.4 against 89.0) and CIFAR-100 (53.8 against 63.5).
- Estimated $\hat T$ is the bottleneck. On CIFAR-100 the estimator "performs very poorly" (500 images per class, sensitive to the percentile) and the forward gain disappears (15.9, below 17.1).
- The printed MNIST $\hat T$ at $N=0.7$ is within 0.01–0.05 of the truth, yet $\hat T$ runs at $N=0.6$ lose 25.6 (backward) and 32.4 (forward) points; the paper does not explain this.
- With a counted $T$ on real noise (Clothing1M) the correction alone adds only +0.9; the large step comes from clean fine-tuning.
- Guarantees hold "in the limit of infinite data" and need a non-singular $T$. Instance-dependent noise is "an open question".

#### Recipe for our project

- [paper] Count $T$ from paired clean/noisy labels (teacher, Claude) instead of anchor points; the paper does this on Clothing1M but gives no formula.
- [extrapolation] Smoothed soft counts, with $c_n$ Claude's answer encoded as the training target (0.7/0.2/0.1) and prior $\pi_{i\cdot}$ of full support:
$$\hat T_{ij}=\frac{\sum_{n}\mathbb 1[y_n=i]\,c_n(j)+\lambda\,\pi_{ij}}{\sum_{n}\mathbb 1[y_n=i]+\lambda}$$
Choose $\lambda\in\{1,3,10,30,100\}$ by leave-one-out log-likelihood; use only the 1,199 fit rows for any dev number. Prior: identity-heavy, or a tied family-block prior (guard $|F(i)|=1$).
- [paper] Hard-label forward loss $-\log(\hat T^{\top}\hat p)_{\tilde y}$; predict with $\hat p$; row-stochastic, non-singular $T$.
- [extrapolation] Soft-target form $\mathcal L=-\tilde q^{\top}\log(\hat T^{\top}\hat p)$, with logit gradient $\hat p_k-r_k$, $r_k=\sum_j\tilde q_j\,\hat p_k\hat T_{kj}/\sum_i\hat p_i\hat T_{ij}$ (re-derived and correct; requires $\sum_j\tilde q_j=1$). A column identical in every row yields zero gradient. Teacher rows use $T=I$. Needs a 76-way softmax head; not applicable as written to the pointwise ranker.
- [paper] Sequence: forward-corrected training on noisy rows, then plain cross-entropy fine-tuning on clean rows (Table 3, #6 → #8).
- [extrapolation] $T$ does not depend on the class prior; estimate the pool prior by $\operatorname{argmin}_{\pi}\lVert\hat T^{\top}\pi-\tilde\mu\rVert_2^2$ (two-class check: $\hat\pi_{\text{NONE}}=(0.709-\beta)/(\alpha-\beta)$); rescale $\hat p_k\,\pi^{\text{teacher}}_k/\hat\pi^{\text{pool}}_k$ at inference; do not stack the earlier prior correction.
- [extrapolation] Unit tests: row sums; $T=I$ reduces to plain cross-entropy with gradient $\hat p-\tilde q$; finite-difference gradient; $T$ against $T^{\top}$ direction; condition number; no dev rows in $\hat T$.

#### What the paper does not show

- No counting formula, smoothing or sample-size analysis for a paired-label $T$; no case with about 13 pairs per class.
- No soft or ranked targets, no label-shift treatment, no instance-dependent noise.
- No control fine-tuning a cross-entropy-pretrained network on Clothing1M (only #6 was fine-tuned), and single runs for all ResNet results.
- No evidence that the gain survives ensembling or blending, and it adds no coverage for classes or questions absent from the noisy set.

---

### 6.8 Paper 8: a benchmark for maths misconceptions

**Citation.** Otero, N., Druga, S., Lan, A. "A Benchmark for Math Misconceptions: Bridging Gaps in Middle School Algebra with AI-Supported Instruction." arXiv:2412.03765v1 [cs.HC], 4 Dec 2024, 37 pp. Data/code: GitHub `creature-ai/math-misconceptions` (ref. 77) plus a Hugging Face copy.

**Benchmark (pp. 5–7, 25–29).** 55 algebra misconceptions and errors ("MaEs"), grades 4–8, from 145 sources (snowball from Google Scholar; 56% US). Exactly 4 typed examples each (220): question, wrong answer (often with working or spoken reasoning), correct answer, bracketed figure description, source link. Researcher comments on answers were removed because they made labelling easier (p. 10); removal was incomplete (Figs. 13, 14). No "no misconception" class. Topics: Number sense 01–05; Number operations 06–22; Ratios 23–28; Properties 31–34 in the text, 29–34 by our check (six codes); Patterns/functions 35–42; Algebraic representations 43–44; Variables 45–48; Equations 49–55.

**Protocol (pp. 7–9, 30).**
1. 100 iterations; in each, for every MaE draw one of its 4 examples as the in-context "training" example and another as the test example (random, not similarity-chosen).
2. One prompt (about 140 fixed words): role sentence (expert middle-school maths tutor); slot 1 all MaE ids with descriptions; slot 2 one labelled example per MaE; slot 3 the unlabelled test items, batched; output rule `UNLABELED ITEM $N DIAGNOSIS: MaE_ID`, nothing else. No reasoning, no system prompt described.
3. Exp. 1: test items from the whole set, 5,500 predictions. Exp. 2: test items only from the same topic as the training MaEs, 1,100 predictions.
4. Exact-match scoring. Metrics are defined in words only (p. 5). Our reading, with $C_{ij}$ the number of test items of true MaE $i$ predicted as $j$:

$$\mathrm{Acc}=\frac{\sum_i C_{ii}}{\sum_{i,j} C_{ij}},\qquad R_i=\frac{C_{ii}}{\sum_j C_{ij}},\qquad P_j=\frac{C_{jj}}{\sum_i C_{ij}}$$

Topic scores are means over the topic's codes; this reproduces every Exp. 1 cell of Fig. 4 from Fig. 18 after truncation (Properties only with MaE29–34; recall 284/600 = 0.473 against 0.471 printed).

**Hyperparameters.** Model "gpt4-turbo", accessed 1 March–25 April 2024; snapshot, temperature, top-p, seed and max tokens not stated. Text only (figures as bracketed descriptions). 100 draws per experiment. Every run has descriptions plus one example per code.

**Results.** Fig. 3 (p. 11): Exp. 1 precision 0.526, recall 0.529; Exp. 2 precision 0.753, recall 0.748.

| Fig. 6 (p. 14) | Exp. 1 | Exp. 2 |
|---|---|---|
| Predictions | 5,500 | 1,100 |
| Correct | 2,913 (52.96%) | 812 (73.82%) |
| Incorrect | 2,587 (47.04%) | 288 (26.18%) |
| Experts agree with model | 687 (12.49%) | 111 (10.09%) |
| Picture needed | 284 (5.16%) | 56 (5.09%) |
| Correct after review | 3,600 (65.45%) | 923 (83.91%) |
| Model truly wrong | 1,487 (27.04%) | 121 (11.00%) |

Exp. 1 review rows sum to 2,458, not 2,587 (129 unaccounted).

| Topic (Fig. 4, p. 11) | Exp. 1 P / R | Exp. 2 P / R |
|---|---|---|
| Number sense | 0.593 / 0.654 | 0.776 / 0.76 |
| Number operations | 0.685 / 0.77 | 0.905 / 0.891 |
| Ratios | 0.215 / 0.191 | 0.286 / 0.333 |
| Properties | 0.462 / 0.471 | 0.776 / 0.783 |
| Patterns, functions | 0.489 / 0.346 | 0.6 / 0.581 |
| Algebraic representations | 0.4151 / 0.37 | 1 / 1 |
| Variables | 0.657 / 0.4975 | 0.688 / 0.67 |
| Equations | 0.493 / 0.468 | 0.731 / 0.7 |

Per-code Exp. 1 recall (Fig. 18 diagonal, p. 36; sums to 2,914): best 100 (MaE05, 20), 98 (19), 96 (11, 16); worst 0 (31), 4 (42), 8 (44), 10 (23), 11 (49), 13 (48); ratio codes 23–28 are 10, 25, 14, 21, 20, 25.

**Ablations, failure modes, limits.**
- Topic constraint is the only ablation: +20.9 points accuracy. In Exp. 1, 1,256 of 5,500 predictions (22.8%, about half the errors) name a code outside the true topic. Algebraic representations has every error out of topic and goes 0.37 to 1.0; Ratios has 431 of 600 predictions in topic with 115 correct and goes 0.191 to 0.333 (about 18% of its errors removed, the smallest share). Not a rule: Number operations has 49 of 1,700 predictions out of topic yet rises 0.77 to 0.891.
- Largest confusions (Fig. 18): 42→23 (52), 35→36 (50), 38→23 (45), 36→35 (38), 41→36 (38), 07→06 (35), 14→15 (35), 30→33 (34), 37→43 (33), 38→37 (32), 02→01 (31), 02→06 (31). Reciprocal pairs: 35/36, 06/07, 14/15, 08/09, 01/02. Function codes 38 and 42 go to ratio code 23, never the reverse. The text (p. 13) names 07/11 and 44/49, but the figure shows 07→11 = 0 and 44→49 = 6.
- Surface pull (our reading of Figs. 13–17): in at least three of nine worked errors the test item shares its story with the predicted code's example (eggs and flour → eggs and cakes; hourly pay → overtime pay; same shaded-fraction worksheet).
- Label ambiguity: experts accepted 26.6% (Exp. 1) and 38.5% (Exp. 2) of "errors"; about 5% of predictions needed the picture. Two reviewers agreed on 90.91%.
- Authors' stated limits: one model; text only; five surveyed teachers; no demographic data in sources; MaEs at different levels of abstraction; chain-of-thought and reconstructing the student's steps proposed, untested (p. 16).

**Recipe for our project.**
- [paper] Description plus one cross-question example per candidate, no reasoning, gives 53% top-1 over 55 codes on clean typed text: a level, not a gain.
- [paper] Items carrying an explanation of the error are easier to label (anecdotal, p. 10).
- [paper] Restricting to the true topic adds 21 points; every topic improves; Ratios stays lowest.
- [paper] Card content: id, description, question, the wrong working, correct answer, bracketed figure description.
- [extrapolation] Gain from a constraint roughly tracks the share of predictions outside the restricted set (22.8% against +20.9 points); it is not an identity (Algebraic representations 0.63 outside, +0.63; Ratios 0.28, +0.14; Number operations 0.03, +0.12). The paper's topic is an oracle; our equivalent is the top-10 shortlist, whose recall is the ceiling.
- [extrapolation] Add a "why this working shows this code" line and a "nearest confusable code and deciding difference" line; use a context-free canonical example or two unlike ones; ask for the student's steps before the label; one test item per prompt.
- [extrapolation] Our earlier negative results (examples in a text matcher; four arbitrary few-shots) are not contradicted; MAP@3 should credit the alternative labels experts accepted.

**What the paper does not show.** No zero-shot, description-only or example-only arm, so no measured effect of the example or the description; no second model; no confidence intervals; no handwriting, photos or reading errors; no top-3 metric; no reasoning run; no explanation of why Exp. 2 has 1,100 predictions, how topics were weighted, or whether the description list was cut to the topic; no test of an example from the same question as the test item.
