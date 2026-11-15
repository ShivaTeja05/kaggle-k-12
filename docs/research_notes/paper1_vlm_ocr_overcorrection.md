# Paper 1: over-correction in multi-line handwritten maths OCR (PINK)

## 1. Citation and summary

Jin Seong*, Wencke Liermann*, Minho Kim, Jong-hun Shin, Soojong Lim (ETRI, Republic of Korea). "When VLMs 'Fix'
Students: Identifying and Penalizing Over-Correction in the Evaluation of Multi-line Handwritten Math OCR."
arXiv:2604.22774v2 [cs.CY], 26 May 2026. Venue: not stated in the paper.

The paper shows that vision-language models transcribing a student's multi-line handwritten solution often write the
mathematically correct token where the student wrote a wrong one ("over-correction"), which erases the error a marker
needs to see. It proposes PINK, a metric that grades the model transcript and the ground-truth transcript with the same
LLM rubric and penalises every rubric component on which the model transcript scores higher. Across 15 untrained VLMs
on FERMAT, 42.1–66.2% of transcriptions show at least one such event, larger models show more, and an explicit
"transcribe exactly, do not fix" prompt gives no net gain.

## 2. Which of our bottlenecks it addresses

**Addressed directly (as a diagnosis):** the transcription bottleneck in section 6.6 of our review. Our transcriber
turned a wrong final answer right on 8 of 173 rows, and the annotated error text survives in only 55% of machine
transcripts. This paper names that failure, gives a way to count it, and gives a mechanism (the text already generated
outweighs the image at the error token).

**Addressed indirectly:** Claude's −0.048 when it reads our transcript instead of the photo (section 6.4). Our reading
measures (number recall 0.881, character error 0.095) are lexical, like BLEU, and the paper shows lexical measures miss
a one-token change that removes the error. That fits our finding that the loss is the same on well-read and badly-read
rows (0.08 and 0.06). This link is our interpretation.

**Where it does not transfer:**

- It is an evaluation paper. The only fix it tests (a prompt) nets to zero; it proposes no training or decoding method.
- All 15 models are untrained general VLMs. Our transcriber is a LoRA trained on teacher transcripts that keep the
  errors, the kind of training-level intervention the paper calls for and does not test.
- FERMAT errors are perturbations injected into correct solutions and copied out by hand by 43 annotators, not real
  student errors. Every evaluated sample is a perturbed solution (500 clean samples are used only to check the
  oracle score, p5), so nothing transfers to our no-error rows (34%). The
  level is "middle or high school", with long solutions mixing prose and LaTeX.
- It says nothing about the larger part of our hard-row gap (about 0.19 of 0.27 is reasoning on identical input),
  about memorised (question, code) pairs, or about label conventions.
- Its headline rate counts any rubric component with $\Delta S_r>0$, including differences of 1 to 10 points that the
  authors attribute to grader noise or trivial variation. It is not comparable with our hard count of 8 of 173.

## 3. Exact method

### 3.1 Pipeline (Fig. 2, p4)

1. A VLM transcribes the photo with the fixed OCR prompt (Fig. 18).
2. An LLM grader scores the model transcript against a five-component rubric, giving $S_{\text{model},r}$.
3. The same grader scores the ground-truth transcript, which contains the student's error, giving
   $S_{\text{oracle},r}$.
4. Component differences are penalised and the result is normalised over the dataset.

Rubric, $R=5$ components, 20 points each, total 100 (pp3, 13, 20):

| $r$ | Component | Criterion (verbatim) |
|---|---|---|
| 1 | Core Formula/Principle Identification | "Correctly identified and stated the necessary formulas for solving the problem" |
| 2 | Condition/Boundary Application | "Accurately applied given conditions (boundaries, constraints, units, etc.)" |
| 3 | Calculation Accuracy (Early-Mid) | "Intermediate calculations were performed without omissions or contradictions" |
| 4 | Calculation Accuracy (Late) | "Final transformations and simplifications were logical and error-free" |
| 5 | Final Value Verification | "Final answer correctly incorporates all conditions and calculations" |

Fig. 10 (p13) shows the same rubric with 2 points per component and a total of 10. The text and the grading prompt say
20 and 100. The human study used a 0–10 scale.

### 3.2 Equations (transcribed from the page images, pp4–5)

Over-correction difference for rubric component $r \in \{1,\dots,R\}$:

$$\Delta S_r = S_{\text{model},r} - S_{\text{oracle},r} \tag{1}$$

An over-correction event occurs whenever $\Delta S_r > 0$.

Penalised component score:

$$S'_r = \begin{cases} S_{\text{model},r}, & \Delta S_r \le 0,\\ S_{\text{oracle},r}, & 1 \le \Delta S_r \le 10 \quad \text{(Minor)},\\ 0, & \Delta S_r > 10 \quad \text{(Major)}. \end{cases} \tag{2}$$

The cases cover every value only for integer scores (the prompt requires an integer total). Fig. 2 writes the minor
case as $0 < \Delta S \le 10$. The
threshold $T=10$ is half the component maximum.

Penalised score for one item:

$$S_{\text{penalized}} = \sum_{r=1}^{R} S'_r \tag{3}$$

Final score over $N$ items:

$$S_{\text{PINK}} = \frac{\sum_{n=1}^{N} S_{\text{penalized},n}}{\sum_{n=1}^{N} S_{\text{oracle},n}} \tag{4}$$

A model whose penalised scores equal the oracle scores gets 1.0. Lost content ($\Delta S_r<0$) also lowers the score,
so PINK mixes ordinary reading loss with the over-correction penalty.

Worked example (Fig. 2): the student writes $(1-1)-0=0$ and the model writes $(1+1)-0=2$. Oracle 70 (components 4 and
5 score 5 each), model 100, $\Delta S_4=\Delta S_5=+15$, both nullified, penalised score 60. The figure's dataset
example is $60.4/71.3 = 84.7$ (shown ×100).

### 3.3 Over-correction taxonomy (Appendix C, pp13–15)

- **Type 1, Simple-Level (Local Fix).** The model acts as a "mathematical spell-checker": a local token substitution
  that fixes the error and keeps the structure. Example: the student wrote $\sin^{-1}y=\sin^{-1}x+C$ after
  $\int \frac{dx}{1+x^2}$; InternVL3-8B wrote $\tan^{-1}$. Oracle 34, model 99, PINK-penalised 27.
- **Type 2, Critical-Level (Global Rewrite).** The model drops the student's erroneous steps and writes a correct
  summary. Example: GPT-4o omits the derivation and outputs "On solving (i) and (ii), we get x = 4". Oracle 14,
  model 100.
- By model (Appendix A.2): "robust" (Gemini 2.5 Flash), "minor over-correctors" (GPT-4o, Qwen2.5-VL family, mostly
  $\Delta S<10$), "critical over-correctors" (Ovis2 8B, InternVL3 family, frequent $\Delta S>10$).

The paper gives no breakdown by FERMAT's own error categories (computational, conceptual, notational, presentation,
superficial). Its only breakdown is by rubric component.

### 3.4 Mechanism evidence (Appendix C.3, p14; Fig. 13, p16)

Attention maps were taken at the step that generates the target token, given everything generated so far. For
InternVL3-8B writing "tan", the map shows "negligible activation" on the handwritten "sin". For "Qwen2.5-VL 8B", which
wrote "sin", attention sits on those strokes. This is one case; the authors mention "similar patterns in additional
cases" without a count. "Qwen2.5-VL 8B" is not among the 15 evaluated models (the list has 7B). The layer, head and
aggregation used for the maps are not stated in the paper.

### 3.5 Prompts (Appendix G)

- **OCR prompt (Fig. 18, p22).** System: a maths assistant converts a handwritten question-and-answer photo to
  LaTeX, "based solely on the information visible in the image". User: return the content "exactly as it appears in
  the image", add nothing that is not in the image, in the fields `**Question:**` and `**Answer:**`.
- **Mitigated OCR prompt (Fig. 19, p22).** The same text with two additions. Appended to the system prompt:
  "IMPORTANT: You must transcribe EXACTLY what is written, including any errors or mistakes made by the student. Do
  NOT correct or fix any mathematical errors." Inserted in the user prompt, a block headed "CRITICAL INSTRUCTIONS FOR
  FAITHFUL TRANSCRIPTION" with five bullets: transcribe exactly including errors; do not correct, fix or improve;
  the example `If the student wrote "2 + 2 = 5", you must transcribe it as "2 + 2 = 5", NOT "2 + 2 = 4"`; the goal
  is faithful transcription, not correction; preserve all calculation errors and wrong answers.
- **Auto-grading prompt (Fig. 16, p20).** Role line ("You are a math exam grader..."), the rubric as JSON, four
  guidelines (analyse "character by character" with a justification; cite line and position; integer total 0–100;
  JSON only), and an output schema with `item`, `score` (0–20) and `justification` per component plus `total`.
  Whether the grader also sees the problem statement or a reference solution is not stated in the paper.
- **Auto-labelling prompt (Fig. 17, p21).** Classifies a GT and prediction pair as "LaTeX format difference only",
  "Recognition error only", "Both LaTeX and recognition errors" or "No errors". Seven format categories (math mode,
  spacing, list style, fraction style, relation symbols, arrows, vector notation), six recognition error types (wrong
  symbol, wrong number, wrong word, missing content, extra content, broken LaTeX) and tolerance rules (spaces, case,
  punctuation spacing). The output JSON includes `gt_has_math_error`.

## 4. Experimental setup

| Item | Value |
|---|---|
| Dataset | FERMAT (Nath et al., 2025): 609 curated problems, perturbed solutions hand-copied by 43 annotators, 2,244 handwritten samples (p5) |
| Oracle score | Perturbed solutions average 71.3; 500 clean samples score 95.1 (p5) |
| Models (15) | Qwen2.5-VL 7B/32B/72B; InternVL3 8B/38B/78B; Ovis2 8B/34B; Llama-3.2 11B/90B; Pixtral 12B/124B; GPT-4o; GPT-4o Mini; Gemini 2.5 Flash |
| Primary grader | GPT-5 (`gpt-5-2025-08-07`) |
| Second grader | Qwen3-Next-80B-A3B-Thinking ("Qwen3-80B") |
| Threshold | $T=10$; $R=5$; 20 points per component |
| Baselines | BLEU, Edit Distance, MathBERTa-F1 (rescaled with baseline F1 = 0.8786 from 1,000 random pairs) |
| Human study | 3 annotators, 200 Ovis2-8B transcripts, 0–10 grade plus a blind choice between two metric scores |
| Mitigation test | 6 open models, Qwen3-80B grader, $N=2{,}207$ (37 of 2,244 dropped for grader failures) |

Temperature, sampling, maximum tokens, image resolution, quantisation and the graders' reasoning settings are not
stated in the paper. No model is fine-tuned.

## 5. Results

**Table 1 (p7): PINK, BLEU, edit distance.**

| Model | PINK | BLEU | Edit dist. | Over-corrected transcripts (Fig. 14, p17) |
|---|---|---|---|---|
| Gemini 2.5 Flash | 0.935 | 0.614 | 0.2007 | 59.3% |
| Qwen2.5 72B VL | 0.920 | 0.733 | 0.1978 | 65.7% |
| Qwen2.5 32B VL | 0.919 | 0.734 | 0.1975 | 66.2% |
| Ovis2 34B | 0.916 | 0.723 | 0.1997 | 62.5% |
| InternVL3 78B | 0.911 | 0.749 | 0.1969 | 58.5% |
| GPT-4o | 0.907 | 0.735 | 0.1676 | 62.3% |
| InternVL3 38B | 0.905 | 0.771 | 0.1925 | 58.8% |
| Ovis2 8B | 0.887 | 0.652 | 0.2321 | 53.6% |
| InternVL3 8B | 0.873 | 0.642 | 0.2364 | 50.8% |
| Pixtral 12B (as printed; the 124B model, see below) | 0.850 | 0.606 | 0.2836 | 56.1% (Fig. 14 "Pixtral 124B") |
| Qwen2.5 7B VL | 0.832 | 0.565 | 0.3332 | 54.6% |
| Llama-3.2 90B | 0.814 | 0.512 | 0.3339 | 50.1% |
| GPT-4o Mini | 0.787 | 0.618 | 0.2698 | 49.6% |
| Pixtral 124B (as printed; the 12B model, see below) | 0.722 | 0.606 | 0.2495 | 43.9% (Fig. 14 "Pixtral 12B") |
| Llama-3.2 11B | 0.676 | 0.407 | 0.4649 | 42.1% |

Table 4 (p16) gives "Pixtral Large" 0.85 (rank 10) and Pixtral 12B 0.72 (rank 14); Figs. 3, 5 and 15 also put
"Pixtral 124B" at PINK rank 10 and "Pixtral 12B" at rank 14. So Table 1's two Pixtral labels are swapped. Our check:
equation (2) gives $S'_r \le S_{\text{model},r}$, so PINK cannot exceed (mean model score)/71.3; Fig. 4 puts Pixtral
12B's unpenalised score at about 57 (read from the plot), a ceiling of about 0.80, below 0.850. The over-correction
column above is therefore joined by model, not by the Table 1 label: 56.1% belongs to the 0.850 row and 43.9% to the
0.722 row. Which model the BLEU and edit-distance cells of those two rows belong to follows the same swap (both BLEU
cells are printed as 0.606).

- **Rank reversal (pp2, 7).** GPT-4o is 3rd on BLEU and 6th on PINK; Gemini 2.5 Flash is 10th and 1st. On raw rubric
  scores they are 4th and 5th, so the change comes from the penalty.
- **Scale (Fig. 14, p17).** Within every family the smallest model over-corrects least: Qwen2.5 7B 54.6 → 32B 66.2 /
  72B 65.7; InternVL3 8B 50.8 → 38B 58.8 / 78B 58.5; Ovis2 53.6 → 62.5; Llama 42.1 → 50.1; Pixtral 43.9 → 56.1;
  GPT-4o Mini 49.6 → GPT-4o 62.3. The trend is not monotone at the top: in both three-member families the largest
  model is slightly below the middle one (65.7 against 66.2; 58.5 against 58.8). The paper's wording (p2) is that
  larger models within a family "consistently exhibit higher over-correction rates".
- **Where (Fig. 3, p6).** Event counts by component, summed over the 15 models (our addition of the printed cells):
  5,317 / 7,238 / 7,270 / 9,588 / 7,787, total 37,200, or 14.3% / 19.5% / 19.5% / 25.8% / 20.9%. The last two
  components hold 46.7% of events against 40% if uniform. The authors call this "heavily concentrated"; it is a
  moderate skew.
- **Size of events (Table 2, p12).** 0–5 points: 22,483 (60.4%); 5–10: 7,038 (18.9%); 10–15: 3,548 (9.5%); 15–20:
  4,131 (11.1%). The total matches Fig. 3. Major events are 20.6% of all events. A per-model rate of major events is
  not stated in the paper.
- **Format noise (p2).** Of the BLEU discrepancies on Ovis2-8B transcripts (2,244 samples), 60.9% involved
  formatting: 8.2% were formatting only and 52.7% included formatting alongside other differences.
- **Human study (p8; Table 8, p18).** Votes: PINK 330 (55.0%), BLEU 237 (39.5%), neither 33 (5.5%). Item majority:
  PINK 119, BLEU 69, tied 12, $p=3.27\times10^{-4}$. Against MathBERTa 124 / 52 / 24; against edit distance
  122 / 57 / 21. Annotator agreement: weighted kappa 0.69. Grader against mean expert score: Pearson 0.78.
- **Metric agreement (Table 9).** Kendall's $\tau$ among BLEU, edit distance and MathBERTa: 0.75–0.88. Each against
  PINK: 0.47–0.52.

## 6. Ablations, failure modes, limitations

- **Prompt mitigation (Table 10, p18).** PINK original → mitigated, then share of transcripts with an event:
  Qwen2.5-32B .88 → .85, 62% → 53%; Qwen2.5-72B .87 → .84, 61% → 50%; Ovis2-34B .85 → .85, 59% → 54%; Qwen2.5-7B
  .73 → .74, 46% → 46%; Pixtral-12B .59 → .61, 41% → 38%; Llama-3.2-11B .57 → .64, 39% → 42%; average .75 → .75
  (+.01), 51% → 47%. The authors' reading: the instruction "also suppresses correct mathematical content" in strong
  models, so "prompt-level intervention alone is insufficient".
- **Threshold (Appendix A, p12).** Kendall's $\tau$ of the ranking against $T=10$ is 0.905 at $T=5$ and 0.981 at
  $T=15$. At $T=0$ it is about 0.66 (read from the plot).
- **Grader robustness (Appendix E, pp15–17).** GPT-5 against Qwen3-80B: per-sample grading $r=0.78$ (QWK 0.78);
  per-sample penalty $r=0.74$ (QWK 0.73); per-model PINK $r=0.99$; rank $\tau=0.90$ ($N$ = 33k samples, 15 VLMs).
  Five repeat gradings of GPT-4o output by Qwen3-80B ($N=1{,}803$): PINK mean 0.872, CV 0.41%, penalised-score QWK
  0.926. Grading-prompt paraphrases (three variants, four VLMs, Qwen3-80B grader): std ≤ 0.006. OCR-prompt paraphrases
  (three variants, four VLMs, 1,500 samples, Qwen3-80B grader): std ≤ 0.015, with gaps of at least 0.07 between
  quartiles.
- **Limitations stated (p9).** PINK is one axis, not overall quality; prompting does not solve it; the primary grader
  is proprietary; one dataset; a small human study.
- **Limitations we add.** Any $\Delta S_r>0$ counts as an event and per-sample scores are noisy, so the 42–66%
  headline includes grader noise. No noise floor is reported (the event rate when the oracle transcript is graded
  twice). A model that drops content scores lower and has fewer chances to exceed the oracle, which may explain part
  of the "stronger models over-correct more" trend. The mechanism rests on one attention example.

## 7. Recipe for our pipeline

### (a) Transcription prompt and decoding

1. **Keep "do not fix" wording, expect little from it.** Supported by the paper: explicit instructions cut event
   frequency by 4 points on average (10–11 on the 32B and 72B models) with no net quality gain, and for 7B–12B
   models the mitigated prompt did not hurt (PINK +.01 to +.07). Our transcriber is 8B, so the Fig. 19 additions are
   safe to try. Our extrapolation: put them in the training prompt of the next LoRA, since the paper tested untrained
   models only.
2. **Do not move to a larger untrained transcriber for faithfulness.** Supported by the paper: within every family
   the smallest model over-corrects least (the two largest members of Qwen2.5 and InternVL3 are within 0.5 points of
   each other). Caveat: the confound in section 6.
3. **Decoding.** The paper states no decoding settings, so it supports no choice. Our extrapolation: keep greedy as
   the reference and use sampling only as a detector (see (c)).
4. **Cut the context that creates the pressure.** Our extrapolation from Fig. 3 and Fig. 13: late errors are fixed
   more often because more correct context precedes them. Test (i) a second read of the final-answer line from a
   crop with no preceding lines, (ii) a "final answer first" output order, and (iii) removing the question text from
   the transcriber prompt if it is there, since the question implies the right answer.
5. **Training-level fix.** The paper names this direction and does not test it. Our extrapolation: raise the loss
   weight on tokens inside annotated error spans and oversample rows whose error is in the last two lines.

### (b) Measuring over-correction on our 1,500 rows

Use out-of-fold machine transcripts only. The transcriber was trained on teacher transcripts, so in-fold rows would
understate the rate (our hygiene rule, not from the paper).

1. **Final-answer flip rate** (our extrapolation; deterministic analogue of $\Delta S_5>0$). With teacher final answer
   $t$, machine final answer $m$ and correct answer $a^\*$: flip-to-right $= \#\{t\ne a^\*, m=a^\*\} / \#\{t\ne a^\*\}$
   and flip-to-wrong $= \#\{t=a^\*, m\ne a^\*\} / \#\{t=a^\*\}$. Random misreading rarely lands exactly on $a^\*$, so
   a flip-to-right rate well above chance is the signature of a prior. We have 8 of 173 so far.
2. **Fate of the error span** (our extrapolation of the Type 1 / Type 2 split). For the 45% of rows where the
   annotated error text is absent from the machine transcript, label each as (i) replaced by the value that makes the
   line valid (Type 1), (ii) line missing (Type 2; compare our 40 rows with a wrong line count), or (iii) misread to
   another wrong value (ordinary reading error). Where our step checker applies (40% of lines), test
   $\text{valid}(\text{machine line})=1$ and $\text{valid}(\text{teacher line})=0$. Normalise formatting first with
   the Fig. 17 lists (supported as the paper's way to separate format from recognition).
3. **PINK itself** (supported by the paper as defined; the transfer to short K-12 workings is our extrapolation).
   Grade the teacher and machine transcripts with the Fig. 16 prompt and apply equations (1)–(4) with $T=10$. Report
   minor and major events separately and by component, and grade the teacher transcript twice for the noise floor.
   Trust only aggregates: per-sample grader agreement was $r=0.78$. This needs a judge model, so it needs the user's
   permission under the current hold.
4. **Split every measure by line position.** Supported by the paper: 46.7% of events fall in the last two components.

### (c) Verifying the error line on the image

- The paper supports the diagnosis only: when the model writes the corrected token it is not attending to the strokes
  (one example, Fig. 13). It tests no verification method.
- Our extrapolation, in order of cost:
  1. Re-read a crop of the single suspect line with no other lines in context, and compare with the full-page read. A
     difference on a number or operator marks a likely fix.
  2. Ask a two-choice question on the crop ("is this 5 or 6?"): the full-page reading against the value the candidate
     misconception would produce. This removes the generation prior.
  3. Sample several transcripts. A token that alternates between a value that makes the line valid and one that does
     not is a suspect location.
  4. Flag numeric tokens generated with low attention mass on image tokens. This follows Fig. 13 and has no
     quantitative support in the paper.
- A model that diagnoses from the photo (Claude, Qwen3.8-27B with thinking) may suffer the same pressure on its
  internal reading. The paper measured transcription only; this is our extrapolation.

## 8. Pages inspected as images

1, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22 (flattened onto white first; the supplied PNGs are
transparent). Pages 2, 3, 9, 10, 11 and 19 are prose or references, read from the text layer. The Fig. 3 cell values
come from the text layer; their total (37,200) matches Table 2.

Verified against page images: 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22 (Fig. 3 cells and Fig. 4 re-read from a 6× render of page 6)
