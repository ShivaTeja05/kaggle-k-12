# Paper 3: STaR (Self-Taught Reasoner), extracted for the K-12 misconception task

Extracted 14 November 2026 from the 30-page arXiv PDF (text read in full; equations, Algorithm 1 and tables
transcribed from the page images). Section 7 marks each item **[paper]** (supported by the paper) or **[ours]** (our
extrapolation).

## 1. Citation and summary

Eric Zelikman\*, Yuhuai Wu\*, Jesse Mu, Noah D. Goodman. "STaR: Self-Taught Reasoner: Bootstrapping Reasoning With
Reasoning." Stanford / Google Research. arXiv:2203.14465v2, 20 May 2022.

STaR turns a dataset of (question, answer) pairs with no rationales into rationale training data: the model writes a
rationale and an answer for every training question, only rationales whose answer equals the gold answer are kept,
and the original pre-trained model is fine-tuned on them; the fine-tuned model then regenerates the data and the loop
repeats. For questions it still gets wrong, "rationalisation" gives the gold answer as a hint, lets the model write a
justification, and stores it as if no hint had been given. On GPT-J (6B), CommonsenseQA dev accuracy goes from 36.6%
(few-shot chain of thought) and 60.0% (direct fine-tuning) to 72.5%.

## 2. Which of our bottlenecks it addresses, and how well it transfers

**Addressed directly.**

- *Whose conventions the reasoning follows.* The filter is the human gold label, so every kept chain ends at the human
  annotators' code. This is the opposite of Stage B, where the student imitated Claude's chains and Claude's
  conventions (a Claude example of a pair is worth 0.656, a human one 0.820).
- *Whose reasoning style is imitated.* The chains are the model's own, so the training targets are text the model can
  already produce. Stage B asked an 8B model to copy another model's explanations and got the form without the
  substance. The paper never compares self-generated rationales with rationales distilled from a stronger model, so
  it gives no evidence on this point either way.
- *Seen-pair rows* (Qwen3.8-27B 0.709 against 0.868 for our classifier). Fine-tuning on gold-filtered chains teaches
  the label mapping for pairs present in the fit set. In the paper STaR beats direct fine-tuning of the same model
  (72.5 against 60.0 on CQA; 10.7 against 5.8 on GSM8K).

**Addressed only through rationalisation.** Hard rows the model never gets right contribute nothing without the hint
step. The paper says improvement "ends when the model fails to solve new problems in the training set".

**Not addressed.** Misreading the handwriting; ranking three codes (their metric is top-1 accuracy); rows whose
(question, code) pair is absent from the fit set, where any gain would have to come from generalisation. The paper's
CQA dev and GSM8K test scores are on held-out questions from the same distribution as training; generalisation outside
the training distribution is shown only for arithmetic (unseen 9- and 10-digit sums).

**Why transfer is doubtful.**

| | Paper | Us |
|---|---|---|
| Base model | GPT-J, 6B, text only, no built-in reasoning; few-shot CoT 36.6% on CQA, 3.1% on GSM8K, <1% on 2-digit sums | Qwen3.8-27B, vision, native thinking; top-1 0.704, MAP@3 0.759 with no training |
| Rationale | 2–3 sentences in a fixed template | A long free-form thinking chain |
| Training rows | 9,741 (CQA), 7,473 (GSM8K), 10,000 per iteration (arithmetic) | 1,199 |
| Label space | 5 options listed per question; free numeric answer | 75 codes + NONE; 34% of rows are NONE |
| Outer iterations | 16 to 46, each a full retrain from the base | One or two are affordable |
| Fine-tuning | GPT-J repository fine-tuning script, lr $10^{-6}$ (full-weight updates are our reading; the paper does not say) | LoRA (not in the paper) |

Most of the paper's gain is a weak model learning a task it could barely do. Ours already reasons and scores 0.70
top-1, so the headroom is of a different kind (label conventions, not the ability to reason). The paper reports no
CQA or GSM8K result after a single outer iteration, which is the regime we would be in. With 8 times fewer rows and
much longer chains, memorising chains is a larger risk.

## 3. Exact method

**Setup (p. 3).** Pre-trained LLM $M$; dataset $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{D}$; a small prompt set with
rationales $\mathcal{P}=\{(x^p_i,r^p_i,y^p_i)\}_{i=1}^{P}$, $P\ll D$ (e.g. $P=10$). Each input is the few-shot prompt
followed by the question: $x_i=(x^p_1,r^p_1,y^p_1,\dots,x^p_P,r^p_P,y^p_P,x_i)$. The model produces a rationale
$\hat r_i$ and then an answer $\hat y_i$. Stated assumption: rationales that lead to correct answers are of better
quality than those that lead to incorrect ones.

**Algorithm 1, STaR (p. 4; lines 4, 6 and the $\cup\,\mathcal{D}^{\text{rat}}_n$ in line 7 are the rationalisation
parts, printed in blue; Algorithm 2 on p. 26 is the same without them).**

```
Input  M: a pretrained LLM; dataset D = {(x_i, y_i)}_{i=1}^{D}  (w/ few-shot prompts)
1: M_0 ← M                                                        # Copy the original model
2: for n in 1...N do                                              # Outer loop
3:     (r̂_i, ŷ_i) ← M_{n-1}(x_i)                   ∀ i ∈ [1, D]   # Perform rationale generation
4:     (r̂_i^rat, ŷ_i^rat) ← M_{n-1}(add_hint(x_i, y_i))  ∀ i ∈ [1, D]   # Perform rationalization
5:     D_n ← {(x_i, r̂_i, y_i) | i ∈ [1, D] ∧ ŷ_i = y_i}           # Filter rationales using ground truth answers
6:     D_n^rat ← {(x_i, r̂_i^rat, y_i) | i ∈ [1, D] ∧ ŷ_i ≠ y_i ∧ ŷ_i^rat = y_i}   # Filter rationalized rationales
7:     M_n ← train(M, D_n ∪ D_n^rat)                              # Finetune the original model on correct solutions - inner loop
8: end for
```

**Objective and policy-gradient reading (p. 4).** $M$ is viewed as a discrete latent-variable model
$p_M(y\mid x)=\sum_r p(r\mid x)\,p(y\mid x,r)$. With the indicator reward $\mathbb{1}(\hat y=y)$:

$$J(M,X,Y)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\,\mathbb{1}(\hat y_i=y_i) \tag{1}$$

$$\nabla J(M,X,Y)=\sum_i \mathbb{E}_{\hat r_i,\hat y_i\sim p_M(\cdot\mid x_i)}\big[\mathbb{1}(\hat y_i=y_i)\cdot\nabla\log p_M(\hat y_i,\hat r_i\mid x_i)\big] \tag{2}$$

The indicator discards the gradient of every sampled rationale that does not reach $y_i$; that is the filter in
line 5. STaR approximates $J$ by "(1) greedily decoding samples of $(\hat r_i,\hat y_i)$ to reduce variance of this
estimate (at the cost of potentially biased exploration of rationales), and (2) taking multiple gradient steps on the
same batch of data". Rationalisation is framed (p. 8) as sampling rationales from $p(r\mid x,y)$ and not
$p(r\mid x)$: "an off-policy estimate of the objective in Equation 1, sampling from the hint-augmented model as a
proposal distribution". No importance weights are applied.

**Rationalisation format (pp. 4, 5, 24).** Applied only to problems the model failed (the text says so; as written,
Algorithm 1 line 4 runs the hinted pass for all $i$ and line 6 keeps only rows with $\hat y_i\neq y_i$). On CQA the hint is the string
`(CORRECT)` appended to the gold option inside the answer list, e.g. `(b) grocery cart (CORRECT)`; the few-shot
examples for this step carry the same marker; the model then writes rationale and answer in the usual style. On
arithmetic the gold sum is written after `Target:` and the model produces the scratchpad and reproduces the answer. A
rationalised rationale is kept only if $\hat y^{\text{rat}}_i=y_i$. The stored training example uses the original
prompt: "we do not include the hint in its corresponding prompt, as if the model had come up with the rationale
without the hint."

**Restart from the base model (pp. 3, 9).** "Once we collect a new dataset, we train from the original pre-trained
model $M$ instead of continually training one model to avoid overfitting." $M_{n-1}$ is used only to generate data.

**Training steps per iteration (pp. 5, 6, 8).** 40 steps in the first outer loop, then 20% more each loop:
$S_n = 40\cdot 1.2^{\,n-1}$. "Training more slowly at the beginning ultimately benefits model performance."
Arithmetic with rationalisation starts at 300 steps and adds 20 steps per iteration (starting at 300 without
rationalisation overfits 1-digit sums). GSM8K: steps capped from the 30th iteration, "after 7912 steps"
($40\cdot1.2^{29}=7912.5$, so this is the per-iteration count).

**Greedy or sampled.** One rationale per question per iteration. Section 3.1 says greedy decoding; Section 5 says "low
sampling temperature". The value is not stated in the paper.

## 4. Exact hyperparameters

| Item | Value |
|---|---|
| Base model | GPT-J, 6B; 28 layers, embedding 1024, 16 heads of dimension 256, FFN 16384, vocabulary 50.4K, pre-trained on the Pile (p. 27) |
| Few-shot prompts | CQA: text says the "same 10 questions" as Wei et al., but Appendix B prints 7. Arithmetic: 10 random examples per digit length, 1–5 digits. GSM8K: Appendix I prints 6; the count is not stated |
| Learning rate | $10^{-6}$, best of a search over $10^{-7}$ to $10^{-4}$; 100-step warm-up, then constant |
| Batch | 8 sequences of length 1024, short examples packed to fill 1024 |
| Optimiser | Adam, no weight decay; fine-tuning with the GPT-J repository script (the paper does not say whether all weights are updated; full fine-tuning is our reading); training and sampling on one TPU-v3 node |
| Steps | 40 in iteration 1, ×1.2 per iteration (arithmetic with rationalisation: 300, +20 per iteration) |
| Outer iterations | Arithmetic: 16 with rationalisation, 28 plotted without, 36 in the extra-digits run. GSM8K: 36 without rationalisation, then 10 more with. CQA: not stated ("until we see performance saturate, and we report the best results") |
| Temperature | "Low"; value not stated. 0.5 and 0.7 were tried as alternatives and were worse |
| Maximum rationale length | Not stated |
| Data | Arithmetic 50,000 generated, 10,000 sampled per iteration. CQA 9,741 train / 1,221 dev / 1,285 test. GSM8K 7,473 train / 1,319 test |

Note [ours]: with a 100-step warm-up and 40 steps, iteration 1 never reaches the full learning rate; the per-iteration
count first exceeds 100 at iteration 7 ($40\cdot1.2^{6}=119$).

## 5. Results

**Table 1, CommonsenseQA dev accuracy (p. 7).**

| Model | Accuracy (%) | Train data used (%) |
|---|---|---|
| GPT-3 direct fine-tuned (about 30× larger) | 73.0 | 100 |
| Few-shot direct GPT-J | 20.9 | ~0 |
| Few-shot CoT GPT-J | 36.6 | ~0 |
| Few-shot CoT LaMDA 137B | 55.6 | ~0 |
| GPT-J direct fine-tuned | 60.0 | 100 |
| STaR without rationalisation | 68.8 | 69.7 |
| STaR with rationalisation | **72.5** | 86.7 (78.2 from generation + 8.5 from rationalisation) |

**Table 2, GSM8K test accuracy (p. 8).**

| Model | Accuracy (%) | Train data used (%) |
|---|---|---|
| Few-shot direct GPT-J | 3.0 | ~0 |
| Few-shot CoT GPT-J | 3.1 | ~0 |
| GPT-J direct fine-tuned | 5.8 | 100 |
| STaR without rationalisation | 10.1 | 25.0 |
| STaR with rationalisation | **10.7** | 28.7 (0.5 from rationalisation) |

**Arithmetic (p. 6).** 89.5% overall after 16 iterations (the text does not name the variant; 16 iterations and the
Figure 4b end points, whose mean is about 89.4, indicate the run with rationalisation); a no-rationale baseline
(10,000 examples, 5,000 steps) gets 76.3%. Few-shot 2-digit accuracy is below 1%; one iteration with rationalisation lifts it to 32%. Without
rationalisation learning is stage-wise: $n$-digit sums stay poor until $(n-1)$-digit sums are good. Read from Figure
4: without rationalisation 5-digit sums take off only after about iteration 23; with it, at iteration 16 the five
lengths sit at about 100 / 96 / 91 / 85 / 75%. Figure 5: 9- and 10-digit sums, never trained, reach roughly 60% and
10% (noisy).

**Kept fraction per iteration.** Only final fractions are given (the "train data used" columns). The per-iteration
curve is not stated in the paper.

**Other numbers.** Keeping the few-shot prompts in the fine-tuning examples: 60.9 → 68.8 without rationalisation,
69.9 → 72.5 with (p. 8). Human ranking (20 crowdworkers): STaR rationales 30% more likely to be ranked above few-shot
ones ($p=.039$), 74% more likely above human-written ones ($p<.001$) (p. 7). GSM8K: the number of calculation steps
matches the human solution on 53–57% of problems (p. 8).

## 6. Ablations, failure modes, limitations

- **Rationalisation.** +3.7 points on CQA (68.8 → 72.5), where rationalised examples are 8.5% of the training set
  and total data used rises from 69.7% to 86.7% (the generation share itself rises from 69.7% to 78.2%); +0.6 on
  GSM8K (10.1 → 10.7), where rationalised examples are 0.5% of the training set and total data used rises from 25.0%
  to 28.7% ("does not substantially improve performance"); on arithmetic it removes the stage-wise delay. Its hinted examples give a
  stronger gradient than the unhinted ones, which are the cases the model was already most confident about (p. 9).
- **Rationalised chains can be template matching.** Appendix D: "the answer must have a set of properties. correct
  answer has those properties" and, since the model first got these wrong, this "resembles template-matching more
  than reasoning". Some state a fact about a wrong option and then assert the hinted answer.
- **Hint short-cutting (A.6).** When hints were shown during training, the model learned that the answer always
  equals the hint: it argued for "airport" and then answered "train station (e)".
- **Temperature and more samples (p. 9).** Higher-temperature sampling to enlarge the kept set "is
  counterproductive": it "substantially increases the likelihood of a correct answer despite incorrect reasoning, and
  training on bad or irrelevant reasoning prevents generalization". Temperatures 0.5 and 0.7 "consistently led to
  models worse than models with reasoning alone". Ten samples cost about ten times one. Their suggested use for many
  samples is a majority vote standing in for a missing gold label (untested).
- **Few-shot prompts while sampling** reduce drift from the initial rationale style; without them the model gets
  gradually worse at rationalisation.
- **Right answer, wrong reason.** CQA is 5-way, so about 20% correct by chance and about 30% with simple heuristics
  "without any reasoning"; kept rationales in Appendix A are all correct answers with bad reasoning (question implies
  answer, begging the question, no justification, world-state assertions, red herrings). The conclusion says high-chance settings "yield many
  poor rationales" and how to filter them is "an open problem".
- **Bias amplification (Appendix G).** STaR amplifies whatever leads to correct answers on the dataset: "if biases
  are 'useful' in solving the dataset then they will be amplified", and "this is made worse by rationalization". A
  model may pick the answer first and justify it afterwards.
- **Starting accuracy.** "Few-shot performance must be above chance"; GPT-2 could not bootstrap even on arithmetic.
- **Not in the paper:** LoRA, vision inputs, models with native thinking, datasets under 7,000 rows, label spaces
  with a dominant class, any comparison with rationales from another model, variance across seeds.

## 7. Recipe for our case

1. **Filter.** Keep rollout $j$ of fit row $i$ iff $\mathbb{1}[\hat y_{ij}=y_i]=1$, where $\hat y_{ij}$ is the first
   code of the final answer and $y_i$ the human code. **[paper]** for the indicator; **[ours]** for applying it to
   $N=8$ sampled rollouts; the paper reports that more, higher-temperature sampling was worse than one
   low-temperature sample.
2. **Yield.** If a row's per-rollout top-1 probability is $p_i$, then
   $P(\text{row kept})=1-(1-p_i)^N$ and $\mathbb{E}[\text{kept chains}]=Np_i$. Over a slice with mean $\bar p$,
   $\mathbb{E}_i[1-(1-p_i)^N]\le 1-(1-\bar p)^N$ (the function is concave), so the slice formula is an upper bound;
   rows the model never gets right pull the real yield below it. **[ours]**

   | $p$ | 0.90 | 0.59 | 0.50 | 0.30 | 0.10 | 0.05 | 0.02 |
   |---|---|---|---|---|---|---|---|
   | $1-(1-p)^8$ | 1.000 | 0.999 | 0.996 | 0.942 | 0.570 | 0.337 | 0.149 |
   | $8p$ | 7.2 | 4.7 | 4.0 | 2.4 | 0.8 | 0.4 | 0.16 |

   Per-slice top-1 is not recorded; only overall top-1 (0.704 on 270 rows) is. If NONE rows are near 0.90, error rows
   are near 0.59 (101 of the 270 answered dev rows are NONE: $(0.704\cdot270-0.90\cdot101)/169=0.59$). MAP@3 by slice is an upper bound on top-1. Measure $p_i$ on the fit rows before trusting the table.
3. **Lucky passes.** A rollout choosing blindly among $K$ plausible codes passes with probability $1/K$; over $N$
   rollouts at least one passes with probability $1-(1-1/K)^N$. For $K=76,N=8$ that is 0.10 (0.013 for one rollout);
   for $K=5$ it is 0.83. Sampling eight times multiplies the chance route about eightfold (our arithmetic); the
   paper's stated reason that more and higher-temperature sampling failed is the same route, a correct answer despite
   incorrect reasoning. Guard: keep a row's chains only if at least $m$ of 8 rollouts
   reach $y_i$ (start with $m=2$; rows below go to step 5), and keep the temperature low. **[ours]**, motivated by
   **[paper]** p. 9.
4. **NONE dominance.** A rollout that defaults to "no error" without checking passes on every NONE row, so NONE
   chains are the least trustworthy and the most numerous. With about 403 NONE and 796 error fit rows, uncapped
   counts are about $403\cdot8\cdot0.90=2{,}900$ against $796\cdot8\cdot0.59=3{,}760$: 44% NONE against 34% of rows.
   Rule: at most $c$ chains per row ($c=1$ or 2), then subsample NONE rows so that
   $n_{\text{NONE}}\le \tfrac{0.336}{0.664}\,n_{\text{error}}\approx 0.51\,n_{\text{error}}$. Check that the code
   distribution of the kept set is close to the gold distribution; codes with no kept row must come from step 5.
   **[ours]**; the risk itself is **[paper]** (high-chance settings, bias amplification).
5. **Rationalisation for rows with no pass.** Mark the gold code inside the candidate list in the paper's way
   (`<code> (CORRECT)`), ask for the same thinking-then-answer format, keep only if $\hat y^{\text{rat}}_i=y_i$, and
   store the chain under the original, unhinted prompt. **[paper]**. Additions **[ours]**: a native-thinking model is
   likely to mention the marker in its chain, so drop any chain that refers to the hint ("CORRECT", "marked",
   "given answer"); drop chains that never cite a specific line of the working (the template-matching failure); do
   not put hinted prompts into training (hint short-cutting, A.6). Expect the gain to depend on how many rows this
   adds: rationalised examples worth 8.5% of the training set came with +3.7 points on CQA, 0.5% came with +0.6 on
   GSM8K (in both cases the generation share of the data also grew, so the gain is not from the rationalised rows
   alone).
6. **Restart from base.** Every round trains a fresh LoRA from the untouched Qwen3.8-27B weights on
   $\mathcal{D}_n\cup\mathcal{D}^{\text{rat}}_n$; the previous round's adapter is used only to sample. **[paper]**
   (LoRA itself **[ours]**).
7. **Steps.** Begin small and grow: $S_n=S_1\cdot1.2^{\,n-1}$. **[paper]** for the shape; $S_1$ for 1,199 rows and
   long chains is not derivable from the paper and must be chosen on dev. **[ours]**
8. **Ranking for MAP@3.** The paper's reward is top-1 only. Equation (2) admits any reward, e.g.
   $R=\sum_{k=1}^{3}\mathbb{1}[\hat y^{(k)}=y]/k$ used as a per-chain weight. **[ours]**, untested.
9. **Gate before any further spend.** On the 301 dev rows, against Qwen3.8-27B untrained on the same rows: hard rows
   above 0.608, NONE not below 0.924, seen-pair rows above 0.709, plus the Stage B checks (share of rows at ≥0.95
   confidence and their accuracy, agreement across candidate orderings). **[ours]**
10. **Cost.** $8\times1{,}199=9{,}592$ rollouts; at the test-scoring quote ($7.80 for 1,593 photos) that is about
    $47 before any training, which needs a new and larger approval. The paper notes the same linear cost and prefers
    rationalisation to more samples. A cheaper first step closer to the paper: one low-temperature rollout per fit
    row (about $6), filter, rationalise the failures. **[ours]**

## 8. Pages inspected as images

2 (Figure 1), 3 (notation), 4 (Equations 1–2, Figure 2, Algorithm 1), 5 (protocol, Figure 3), 6 (Figures 4–5),
7 (Table 1), 8 (Table 2, Figure 6), 9 (discussion, limitations), 13–15 (Appendix A failure cases), 16–17 (Appendix B
prompts), 24 (Appendix D rationalisations), 26 (Algorithm 2, case study), 27 (Appendices G and H). Pages 1, 10–12,
18–23, 25 and 28–30 were read as text only (abstract, references, human-evaluation examples, GSM8K prompts).

Verified against page images: 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 30
