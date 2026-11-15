# Paper 8: a 55-misconception algebra benchmark, GPT-4-turbo with one example per misconception

Extraction date: 14 November 2026. Source: 37-page PDF, read in full as text, with every table, figure and appendix
page checked against the page image. "Our reading" marks numbers we derived from a figure; everything else is stated
in the paper.

## 1. Citation and summary

Otero, N., Druga, S., Lan, A. "A Benchmark for Math Misconceptions: Bridging Gaps in Middle School Algebra with
AI-Supported Instruction." arXiv:2412.03765v1 [cs.HC], 4 December 2024. Data and code: GitHub
`creature-ai/math-misconceptions` (ref. 77), plus a Hugging Face copy (no URL printed).

The authors collect 55 algebra misconceptions and common errors ("MaEs") from 145 published sources, each with four
typed question / wrong-answer / correct-answer examples (220 in all). GPT-4-turbo, shown every description plus one
labelled example per misconception, names the right misconception for 53% of held-out examples; when the test is
restricted to one topic at a time it reaches 74%, and 84% after two educators accept some of its "wrong" answers as
valid. Ratio and proportion misconceptions stay near 20–33% in both settings.

## 2. Which of our bottlenecks it addresses, and how directly

- **Hard rows (code known only from other questions).** The whole protocol is this regime: the one labelled example
  per code is a different item from the test item. Direct in structure, far in data.
- **Shortlisting candidates before asking the model.** Experiment 2 against Experiment 1. Direct, but the topic
  restriction is an oracle.
- **Whether one example per candidate helps.** Not tested: every run has descriptions and one example per code.
- **Confusions between similar codes.** A 55 × 55 confusion matrix and worked wrong predictions. Direct.
- **Reading handwriting.** Not addressed.

Critical points on distance from our task:

- **Data type.** Typed, cleaned text copied from research papers; figures replaced by a bracketed description. No
  handwriting, no photos, no reading errors.
- **Size.** 55 labels, exactly 4 examples each, balanced, no "no misconception" class. We have 75 codes plus NONE, a
  long tail, and 1,500 rows.
- **Model and metric.** One model (GPT-4-turbo), one answer per item, top-1 accuracy. We score MAP@3.
- **No baselines.** No zero-shot, description-only or example-only run, no second model, no confidence intervals. The
  paper cannot say what the example or the description contributes.
- **Label quality.** Each label is a source paper's diagnosis of one item; the authors' review found a quarter to
  two-fifths of "errors" were acceptable alternative labels.

## 3. Benchmark and method

### 3.1 Construction (pp. 5–7)

- Snowball sampling: a Google Scholar search for peer-reviewed work on algebra misconceptions, then following references
  and citations. 145 sources (journal papers, proceedings, books, dissertations); 56% of the research was done in the US.
- For each distinct MaE the authors looked for four problems with incorrect answers that researchers had used to
  diagnose it. With more than four available they preferred studies of diverse and underserved populations.
- Stopping rule: collection ended when no new MaE had four examples, or a new MaE was too general to diagnose from one
  question and answer.
- Grades 4–8. Two kinds of entry: errors (a rule not mastered) and misconceptions (a wrong concept that affects every
  use of it).
- Formatting: the correct answer is included with every item; some items were re-typed for uniformity; researchers'
  comments on the student answer were removed because they made the model's job easier (p. 10). Removal was
  incomplete: bracketed researcher notes survive in Figs. 13 and 14.

### 3.2 Structure of one item (Appendix A.1, pp. 25–26)

Per misconception: ID, description, topic, four examples. Per example: question; incorrect answer showing the
misconception; correct answer; if a figure is involved, a second copy with a bracketed description of the figure and
links to the original and a redrawn image; and a link to the source paper.

Example, MaE28 (Fig. 9): lemonade is mixed at 3 cups of water to 2 of juice; one more cup of each is added; does it
taste the same? Wrong answer: yes, the same amount of both was added. Correct answer: no, the proportions changed.
Example, MaE55 (Fig. 12): simplify 3(6x−4)+2(3x−3); wrong working 18x−12+6x−6 = 18x−18+6x; correct 24x−18. Many wrong
answers include the working or the student's spoken reasoning, not only a final value.

### 3.3 Topics and counts

Topic names and id ranges are from p. 6–7; scores are transcribed from Fig. 4 (p. 11).

| Topic | MaE ids | Count | Exp. 1 precision | Exp. 1 recall | Exp. 2 precision | Exp. 2 recall |
|---|---|---|---|---|---|---|
| Number sense | 01–05 | 5 | 0.593 | 0.654 | 0.776 | 0.76 |
| Number operations | 06–22 | 17 | 0.685 | 0.77 | 0.905 | 0.891 |
| Ratios and proportional thinking | 23–28 | 6 | 0.215 | 0.191 | 0.286 | 0.333 |
| Properties of number and operations | 31–34 in the text; 29–34 by our reading | 4 or 6 | 0.462 | 0.471 | 0.776 | 0.783 |
| Patterns, relationships and functions | 35–42 | 8 | 0.489 | 0.346 | 0.6 | 0.581 |
| Algebraic representations | 43–44 | 2 | 0.4151 | 0.37 | 1 | 1 |
| Variables, expressions and operations | 45–48 | 4 | 0.657 | 0.4975 | 0.688 | 0.67 |
| Equations and inequalities | 49–55 | 7 | 0.493 | 0.468 | 0.731 | 0.7 |

The text assigns MaE29 and MaE30 to no topic. Our reading: the diagonal of Fig. 18 reproduces the "Properties" recall
of 0.471 only if MaE29–34 are counted (284/600 = 0.473; MaE31–34 alone give 0.41), so the topic has six entries and
the counts sum to 55. The same check reproduces every other Exp. 1 recall to within 0.001; the table appears to
truncate, not round (Ratios 115/600 = 0.1917 printed 0.191; Equations 328/700 = 0.4686 printed 0.468; 283/600 = 0.4717
would print as the Properties 0.471, which fits the one-count gap noted in section 5). A second check on re-reading:
the mean per-code precision from the columns of Fig. 18 reproduces all eight Exp. 1 precisions in the table after truncation
(0.5939, 0.6851, 0.2155, 0.4629, 0.4892, 0.4151, 0.6574, 0.4932 in the table's row order), the Properties value only
with MaE29–34 (MaE31–34 alone give 0.355).

Short labels of all 55 (our paraphrase of Figs. 10–12, pp. 27–29):

- Number sense: 01 fraction for a shaded figure; 02 unequal parts; 03 exponents as repeated addition; 04 powers of
  algebraic expressions; 05 longer decimal is larger.
- Number operations: 06 simplifying by guessing; 07 simplifying only top or bottom; 08 adding tops and bottoms;
  09 common denominator, numerators unchanged; 10 mixed-number subtraction without regrouping; 11 mixed-number
  subtraction part by part; 12 fraction product run as a carry algorithm; 13 fraction × whole scales top and bottom;
  14 fraction division top-by-top; 15 inverting the dividend; 16 decimal point in addition; 17 uneven decimals in
  subtraction; 18 sign of positive plus negative; 19 two negatives sum to a positive; 20 decimal point in
  multiplication; 21 subtraction without regrouping; 22 remainder appended as decimals.
- Ratios: 23 absolute against relative comparison; 24 same or different units; 25 operations on ratios as fractions;
  26 equivalent forms; 27 no unitization; 28 ratio not seen as a relationship.
- Properties: 29 one proportion formula for all percentage problems; 30 negatives added to a positive in expressions;
  31 commutative subtraction and division; 32 minuend and subtrahend swapped; 33 three types of percent problem;
  34 left-to-right order of operations.
- Patterns and functions: 35 representation missing components; 36 purpose of a graphical representation; 37 graph
  scales; 38 constant rate of change; 39 linear against exponential; 40 slope sign; 41 linking graph, equation and
  table; 42 linear confused with direct proportion.
- Algebraic representations: 43 x and y reversed when plotting; 44 independent against dependent variable.
- Variables: 45 variables switched when transposing; 46 letters as labels; 47 two letters must differ in value;
  48 variables as varying quantities.
- Equations: 49 meanings of variables; 50 forming equations; 51 equals sign as "the answer is"; 52 not checking;
  53 operation symbols misread (7h as 7+h); 54 reversal error (6S=P); 55 like terms.

### 3.4 Evaluation protocol (pp. 7–9)

- Repeated random subsampling, 100 iterations. In each, one of the four examples of every MaE is drawn as the
  "training" example and another as the test example.
- "Training" is in-context only. One prompt holds (i) all misconception ids with descriptions, (ii) one labelled
  example per misconception (question, wrong answer, correct answer; Fig. 1), (iii) the unlabelled test items
  (Fig. 2). The model returns one id per test item.
- So every candidate code carries exactly one in-context example: same misconception, different item from the test
  item, drawn at random and not chosen for similarity.
- Experiment 1: test items drawn from the whole set; 5,500 predictions (55 × 100).
- Experiment 2: test items drawn only from the same topic as the training MaEs; 1,100 predictions. Not stated in the
  paper: why the count is five times smaller, how many items each topic contributed, and whether the description list
  was also cut to the topic. A perfect score on the two-code topic suggests the choice was effectively within-topic.
- Scoring: exact match with the source label. Precision and recall "per MaE" are defined only in words (p. 5); no
  formula is printed. Accuracy (Fig. 6) is correct predictions over all predictions. Our reading: the headline recall
  is the mean per-MaE recall (2,913/5,500 = 0.5296, printed as 0.529).
- Fig. 1 numbers its last training row 57 although the benchmark has 55 entries; unexplained.

### 3.5 Prompt (p. 7 and Appendix A.2, p. 30)

One prompt, about 140 words plus the inserted data. Structure: a role sentence (expert middle-school maths tutor who
has built a set of misconceptions and a labelled set of items); the task (diagnose each unlabelled item); slot 1, the
misconception list with brief descriptions; slot 2, the labelled items; slot 3, the unlabelled items; then an output
rule in capitals, one line per item in the form "UNLABELED ITEM $N DIAGNOSIS: MaE_ID", and an instruction to output
nothing else. No reasoning is requested or allowed. There is no separate system prompt described.

## 4. Experimental setup

| Item | Value |
|---|---|
| Model | "gpt4-turbo", accessed 1 March to 25 April 2024; exact snapshot not stated in the paper |
| Temperature, top-p, seed, max tokens | not stated in the paper |
| Runs | 100 random draws per experiment |
| Input modality | text only; figures given as bracketed descriptions |
| With / without topic constraint | Exp. 1 without, Exp. 2 with |
| With / without examples | always with one example per misconception; no ablation |
| Chain of thought; other models | none (both proposed as future work, p. 16) |

Educator protocols, two separate groups:

- **Survey and interview (pp. 9–11).** Five teachers, 4 to 10+ years' experience, grades 6–10, public schools; four
  serve underserved populations. They rated the clarity and classroom occurrence of eight misconceptions, one per
  topic, each with one example, and answered questions on AI use. Four of the five took a structured interview.
- **Error review (pp. 13–14).** Two experienced algebra educators who were not in the survey sorted each wrong label
  into: agree with the model (more than one MaE fits, or one MaE is a subset of the other); a picture is needed; the
  model is wrong. They agreed on 90.91% of answers, then re-rated disagreements together until one pair remained.

## 5. Results

Headline (Fig. 3, p. 11): Exp. 1 precision 0.526, recall 0.529; Exp. 2 precision 0.753, recall 0.748.

Fig. 6 (p. 14), transcribed from the image:

| | Exp. 1 count | Exp. 1 % | Exp. 2 count | Exp. 2 % |
|---|---|---|---|---|
| Total predictions | 5,500 | 100.00 | 1,100 | 100.00 |
| Correct | 2,913 | 52.96 | 812 | 73.82 |
| Incorrect | 2,587 | 47.04 | 288 | 26.18 |
| Experts agree with the model | 687 | 12.49 | 111 | 10.09 |
| Picture needed | 284 | 5.16 | 56 | 5.09 |
| Correct after review (correct + agree) | 3,600 | 65.45 | 923 | 83.91 |
| Model truly wrong | 1,487 | 27.04 | 121 | 11.00 |

The printed Exp. 1 rows do not add up: 687 + 284 + 1,487 = 2,458, not 2,587 (129 unaccounted). Exp. 2 adds up.

Per topic (table in 3.3): Number operations is best and Ratios worst in both experiments. In Exp. 2 every topic
improves; Algebraic representations goes to 1.0; Ratios stays at 0.286 / 0.333, about twice chance if the choice was
among its six codes. The authors' explanation (p. 12) is that proportional reasoning is inherently complex and
slow to develop in students; no mechanism is tested.

Per-misconception recall in Exp. 1, percent of 100 test draws, our reading of the diagonal of Fig. 18 (p. 36; the
readings sum to 2,914 against the reported 2,913):

- 01–05: 63, 34, 93, 37, 100
- 06–22: 44, 63, 61, 78, 36, 96, 90, 78, 59, 76, 96, 86, 85, 98, 100, 82, 82
- 23–28: 10, 25, 14, 21, 20, 25
- 29–34: 68, 52, 0, 22, 65, 77
- 35–42: 38, 49, 58, 16, 64, 23, 25, 4
- 43–44: 66, 8
- 45–48: 40, 63, 83, 13
- 49–55: 11, 39, 52, 62, 37, 55, 72

Survey (pp. 10–11, 15): 80% or more of educators said they meet these misconceptions. Three of five were unfamiliar
with educational AI tools; four of five wanted to use the dataset with AI for diagnosis or teacher training. Three
teachers rated the same two of eight examples unclear; both had lost their researcher comment. One teacher valued
seeing the student's process, since her platform shows only wrong answers.

## 6. Ablations, failure modes, limitations

- **Topic constraint.** The only ablation. Accuracy 52.96% to 73.82% (+20.9 points). Our reading of Fig. 18 explains
  part of where the gain lands: 1,256 of the 5,500 Exp. 1 predictions (22.8%, about half of the errors) name a code
  outside the true topic. A topic whose errors all leave the topic gains most (Algebraic representations, where every
  wrong answer for MaE43 and MaE44 is a code from another topic, 0.37 to 1.0), and Ratios, whose errors mostly stay
  inside the topic, removes the smallest share of its errors (431 of 600 predictions already fall inside the topic and
  only 115 are correct; recall 0.191 to 0.333, about 18% of its errors removed). This is not a rule (corrected on
  re-check): Number operations has only 49 of 1,700 predictions outside its topic yet rises 0.77 to 0.891, and in
  absolute recall points Number sense (+0.106) and Number operations (+0.121) gain less than Ratios (+0.142). So Exp. 2
  must differ from Exp. 1 in more than the removal of out-of-topic codes; how is not stated in the paper.
- **In-context examples.** No ablation of zero, one or more examples. The one related observation is anecdotal: items
  that kept the researcher's comment on the student answer were easier for the model to label (p. 10). Whether the
  comment sat in the labelled example or the test item is not stated in the paper.
- **Confusions.** The text (p. 13) names MaE07/MaE11 and MaE44/MaE49 as frequent confusions. Our reading of Fig. 18
  does not show those (07→11 is 0; 44→49 is 6). The largest cells we read are 42→23 (52), 35→36 (50), 38→23 (45),
  36→35 (38), 41→36 (38), 07→06 (35), 14→15 (35), 30→33 (34), 37→43 (33), 38→37 (32), 02→01 (31), 02→06 (31), 27→24 (29),
  10→11 (27), 23→28 (26), 28→24 (26), 40→04 (26). MaE31 is never predicted for its own items. Several pairs are
  reciprocal (35/36, 06/07, 14/15, 08/09, 01/02); we take that to mean the two descriptions are not separable.
  Function codes (38, 42) go to ratio code 23 but not the reverse; the authors read this as a prerequisite hierarchy.
- **Surface pull of the example (our reading of Figs. 13–17).** In at least three of the nine worked errors the test item
  shares its question context with the in-context example of the predicted code: an eggs-and-flour recipe item goes to
  the code whose example is an eggs-and-cakes recipe; an hourly-pay item to the code whose example is an overtime-pay
  item; a shaded-fraction item to the neighbouring code whose example is the same kind of worksheet. The authors say
  in general terms that the model matches patterns more than it understands the classification (p. 16).
- **Ambiguous labels and missing pictures.** 10–12% of all predictions were "wrong" yet accepted by experts because
  two codes fit or one contains the other; about 5% could not be judged from a text description (Fig. 6).
- **Limitations stated by the authors:** one model only; text only, with a call for multimodal work; small educator
  sample; source studies lack demographic data; MaEs sit at different levels of abstraction; chain-of-thought and
  reconstructing the student's thinking before labelling are proposed but untested.

## 7. Recipe for our case

**(a) One example per candidate code against none.**
- Supported by the paper: a strong model given a description and one cross-question example for every candidate, with
  no reasoning, gets 53% top-1 over 55 codes on clean typed text. That is a level, not a gain; the paper has no
  no-example arm.
- Supported by the paper (anecdotal): items that carry an explanation of the error are easier to label.
- Our extrapolation: the paper's regime is our hard-row regime, and description plus one cross-question example does
  not by itself make it easy. Expect the card index to help only if the model reconstructs the student's steps first
  (the authors' untested proposal) and the cards carry the mechanism, not the question.

**(b) Constraining the candidate set first.**
- Supported by the paper: restricting to the right topic adds 21 points; every topic improves; the topic with
  within-topic confusions (ratios) stays lowest (0.286 / 0.333). It does not improve least in absolute points: Number
  sense and Number operations gain less recall (+0.106 and +0.121 against +0.142) and Variables gains less precision
  (+0.031 against +0.071); Ratios removes the smallest share of its errors (our reading).
- Our extrapolation: the gain roughly tracks the share of predictions that fall outside the restricted set (our
  reading of Fig. 18): 22.8% of Exp. 1 predictions leave the true topic and the gain is 20.9 points. It is not an
  identity. Per topic the two are equal only for Algebraic representations (0.63 outside, +0.63); Ratios gains half
  its outside share (0.28 outside, +0.14) and Number operations gains more than its outside share (0.03 outside,
  +0.12). Our wrong top answers on hard rows are outside the truth's code family 87% of the time, the case where a
  constraint pays most. But the paper's topic is an oracle taken from the true label. Our equivalent is the top-10
  shortlist, which holds the truth on 90 of 106 hard rows; that recall is the ceiling. Ten cards per prompt is also
  far less clutter than the paper's 55. Do not batch many test items into one prompt as the paper does.

**(c) Card format.**
- Supported by the paper: code id and description; question; the student's wrong working, not only the final answer;
  the correct answer; a bracketed description of any figure; source.
- Our extrapolation: add one line saying why the working shows this code (the removed "researcher comment"), and one
  line naming the nearest confusable code and the deciding difference, since the paper's reciprocal confusions sit
  between near-identical descriptions. Pick a canonical example with a plain, context-free question, or give two
  examples from unlike questions, so the model cannot match on the story.

**(d) Our earlier negative results.**
- −0.052 for cross-question examples appended to candidate cards in a text matcher: not contradicted. The paper has no
  with/without comparison, and its worked errors show the same failure, an example pulling in items with a similar
  question (our reading). A similarity matcher is the extreme case of that pull.
- −0.008 [−0.034, +0.017] for four few-shot examples in Claude's prompt: not contradicted. The paper never tests a
  handful of arbitrary shots; its design gives every candidate exactly one example, which is the design we propose
  and have not yet run.
- The 0.50 to 0.82 lift from one example of the exact (question, code) pair: not tested by the paper, whose example
  and test items are different questions.
- What could make our outcome differ (our extrapolation): balanced coverage of the whole shortlist instead of a few
  shots; mechanism-level cards with a why line; reasoning before the label; and MAP@3, which credits the alternative
  labels experts accepted in a quarter of the paper's errors.

## 8. Pages inspected as images

8, 9, 11, 12, 14 and 25–36 (page 36 also as two enlarged crops of the confusion matrix). The remaining pages are prose
or references and were read as text only.

Verified against page images: 1, 5–16, 25–37 (Figs. 10–18 on pages 27–29 and 31–36 also read from the figure images embedded in the PDF at native resolution; every cell of Fig. 18 was re-read by digit template matching, all 55 rows sum to 100 and the diagonal sums to 2,914)
