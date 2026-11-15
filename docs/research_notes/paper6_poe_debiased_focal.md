# Paper 6: Product of Experts and Debiased Focal Loss

## 1. Citation and summary

Rabeeh Karimi Mahabadi, Yonatan Belinkov, James Henderson. "End-to-End Bias Mitigation by Modelling Biases in Corpora." arXiv:1909.06321v3, 23 April 2020 (ACL 2020; the venue is not printed on the PDF). Code: `github.com/rabeehk/robust-nli`.

A small bias-only model $f_B$ sees only the shortcut features (for example the NLI hypothesis alone) and is trained beside the base model $f_M$. Its predictions change the base model's loss, by ensembling the two in log space (Product of Experts, PoE) or by weighting each row's loss by $(1-\text{bias-only probability of the truth})^{\gamma}$ (Debiased Focal Loss, DFL). The bias-only model is dropped at test time; the base model alone gains 0.8 to 10.1 points on bias-free test sets, with in-domain accuracy within about one point for BERT (except DFL on FEVER, −2.9, and the HANS-tuned DFL with $\gamma=3$ on MNLI, −1.75) and 4 to 11 points lower for the weaker InferSent.

## 2. Which bottleneck it addresses, and how directly

**Bottleneck:** our classifiers learn "the usual codes for this question" (seen-pair rows 0.867, hard rows 0.500). That is a known shortcut with a natural bias-only model, the question-only prior (0.485–0.497 alone). The paper is a training-time recipe for this situation, so the match in mechanism is direct.

**The match in evaluation is poor, and this matters.**

- Every headline gain is on a set built so that the bias is absent or wrong: FEVER Symmetric (claim cues balanced, so they are a random guess), SNLI/MNLI hard sets (the rows a hypothesis-only model gets wrong), HANS (rows where syntactic heuristics fail).
- Our test set is the opposite: 95.7% of test rows are on training questions, and the prior helps on about two thirds of rows.
- What the paper reports for in-domain accuracy (base model alone, bias-only model removed):

| Setting | CE | PoE | DFL ($\gamma=2$) |
|---|---|---|---|
| FEVER dev, BERT | 85.99 | 86.46 (+0.47) | 83.07 (−2.92) |
| SNLI test, BERT | 90.53 | 90.11 (−0.42) | 89.57 (−0.96) |
| SNLI test, InferSent | 84.24 | 80.35 (−3.89) | 73.54 (−10.70) |
| MNLI mismatched dev, BERT | 84.53 | 84.85 (+0.32) | 84.85 (+0.32) |
| MNLI mismatched dev, InferSent | 69.99 | 65.85 (−4.14) | 61.12 (−8.87) |
| MNLI matched dev, BERT | 84.41 | 84.58 (+0.17) | 83.72 (−0.69) |
| MNLI matched dev, InferSent | 69.97 | 66.02 (−3.95) | 60.78 (−9.19) |
| MNLI matched dev, BERT, syntactic bias | 84.51 | 84.19 (−0.32) | 83.95 (−0.56); 82.76 (−1.75) at $\gamma=3$ |

- The paper's own words (Section 6): improving out-of-domain performance "could come at the expense of decreased in-domain performance since the removed biases are useful for performing the in-domain task", especially for DFL and for a weak base model.
- The NLI hard-set gains for BERT are small: +0.8 to +2.5 points. Hard rows are 35% of our dev set, so a gain of that size on hard rows is worth about +0.003 to +0.009 overall; a FEVER-sized gain (+10 points) would be worth about +0.035, and only if the other two thirds are not damaged.
- SNLI and MNLI have hundreds of thousands of rows; we have 1,500, about 3.2 per question. The paper never tests a bias-only model that can memorise the training set.

## 3. Exact method

**Setup (Section 3).** Dataset $\mathcal{D}=\{x_i,y_i\}_{i=1}^{N}$, $x_i\in\mathcal{X}$, $y_i\in\mathcal{Y}$. Base model $f_M:\mathcal{X}\to\mathbb{R}^{|\mathcal{Y}|}$ with parameters $\theta_M$. $x_i^b\in\mathcal{X}^b$ are the biased features of $x_i$. Bias-only model $f_B:\mathcal{X}^b\to\mathbb{R}^{|\mathcal{Y}|}$ with parameters $\theta_B$. $f^j(\cdot)$ is the $j$th element and $\sigma$ is the softmax:

$$\sigma(u^j)=\frac{e^{u^j}}{\sum_{k=1}^{|\mathcal{Y}|}e^{u^k}}$$

**Bias-only loss (Eq. 1).**

$$\mathcal{L}_B(\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\log\big(\sigma(f_B^{y_i}(x_i^b;\theta_B))\big)$$

Bias-only inputs: the hypothesis only (SNLI, MNLI), the claim only (FEVER), hand-built syntactic and similarity features (HANS).

**Product of Experts (Section 3.2.1).** The element-wise product $\sigma(f_B(x_i^b))\odot\sigma(f_M(x_i))$, computed in log space:

$$f_C(x_i,x_i^b)=\log\big(\sigma(f_B(x_i^b))\big)+\log\big(\sigma(f_M(x_i))\big)$$

$$\mathcal{L}_C(\theta_M;\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\log\big(\sigma(f_C^{y_i}(x_i,x_i^b))\big)\qquad\text{(Eq. 2)}$$

$$\sigma(f_C^{y_i}(x_i,x_i^b))=\frac{\sigma(f_B^{y_i}(x_i^b))\,\sigma(f_M^{y_i}(x_i))}{\sum_{k=1}^{|\mathcal{Y}|}\sigma(f_B^{k}(x_i^b))\,\sigma(f_M^{k}(x_i))}$$

$$\nabla_{\theta_M}\mathcal{L}_C(\theta_M;\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{|\mathcal{Y}|}\Big[\big(\delta_{y_ik}-\sigma(f_C^{k}(x_i,x_i^b))\big)\,\nabla_{\theta_M}\log\big(\sigma(f_M^{k}(x_i))\big)\Big]$$

with $\delta_{y_ik}=1$ when $k=y_i$, else 0. Two limits stated by the paper: if the bias-only model classifies the row correctly, $\sigma(f_C^{y_i})=1$ and the gradient is 0 (the row is ignored); if it is uniform, $\sigma(f_B^k(x_i^b))=\frac{1}{|\mathcal{Y}|}$, then $\sigma(f_C^{y_i})=\sigma(f_M^{y_i}(x_i))$ and the gradient equals plain cross-entropy.

**Debiased Focal Loss (Eq. 3).**

$$\mathcal{L}_C(\theta_M;\theta_B)=-\frac{1}{N}\sum_{i=1}^{N}\Big(1-\sigma(f_B^{y_i}(x_i^b))\Big)^{\gamma}\log\big(\sigma(f_M^{y_i}(x_i))\big)$$

$\gamma$ is the focusing parameter; $\gamma=0$ gives cross-entropy. The weight uses the bias-only model's probability of the truth, not the base model's.

**RUBi baseline (Section 3.3).** $\phi$ is the sigmoid; the mask multiplies the base model's outputs, and $\theta_M$ is trained with the cross-entropy of $f_C$:

$$f_C(x_i,x_i^b)=f_M(x_i)\odot\phi(f_B(x_i^b))$$

**Several bias models (Section 3.4).** Biased feature sets $\{x_i^{b_j}\}_{j=1}^{K}$ with models $f_{B_j}$; a classifier on their concatenation is deliberately avoided.

$$\text{Joint PoE:}\quad f_C(x_i,\{x_i^{b_j}\}_{j=1}^{K})=\sum_{j=1}^{K}\log\big(\sigma(f_{B_j}(x_i^{b_j}))\big)+\log\big(\sigma(f_M(x_i))\big)$$

$$\text{Joint DFL:}\quad f_B(\{x_i^{b_j}\}_{j=1}^{K})=\frac{1}{K}\sum_{j=1}^{K}f_{B_j}(x_i^{b_j})\ \text{ then Eq. 3}$$

**Joint training and gradient flow.**

- Trained end-to-end in one stage: $f_B$ and $f_M$ learn together (the stated difference from Clark et al. and He et al., who train the bias-only model first).
- Stated in the text: "the bias-only loss $\mathcal{L}_B$ is not back-propagated to any shared parameters of the base model, such as a shared sentence encoder." So $\mathcal{L}_B$ updates $\theta_B$ only.
- $\theta_M$ is updated from $\mathcal{L}_C$ only.
- Whether $\mathcal{L}_C$ sends gradient into $\theta_B$: not stated in a sentence. Figure 1 draws backward arrows from $\mathcal{L}_C$ through the combination model to the NLI model and its inputs, and from $\mathcal{L}_B$ to the bias-only model only; none runs from the combination model to the bias-only model. The notation $\mathcal{L}_C(\theta_M;\theta_B)$ agrees.
- The total objective as one formula: not stated in the paper. The description implies $\mathcal{L}_C(\theta_M;\theta_B)$ and $\mathcal{L}_B(\theta_B)$ are minimised together, each on its own parameters. Appendix D (InferSent transfer only) adds a weight $\beta$ "to modulate the loss of the bias-only model in ensembling", which implies $\mathcal{L}_C+\beta\,\mathcal{L}_B$ for that experiment (our reading; the formula is not printed).
- A strength parameter $\alpha$ for PoE is introduced for transfer only ("to modulate the strength of the bias-only model in ensembling"); its formula is not stated in the paper.

**Inference.** "After training, we remove the bias-only model and use the predictions of the base model": $\hat y=\arg\max_k f_M^k(x)$.

## 4. Hyperparameters

| Item | Value |
|---|---|
| Base models | BERT (HuggingFace implementation of Wolf et al.; base or large not stated; hidden sizes 768 suggest base) and InferSent |
| BERT training | 3 epochs, "default parameters", learning rate $2\mathrm{e}{-5}$ (stated for FEVER; NLI and HANS state 3 epochs only) |
| InferSent training | 20 epochs, no early stopping, default hyperparameters, BiLSTM dimension 512, classifier with 512 and 512 hidden units, Tanh |
| Batch size, optimiser, warm-up, seeds | not stated in the paper ("default hyperparameters of the baselines") |
| Bias-only classifier, BERT (FEVER, SNLI, MNLI) | "shallow nonlinear classifier with 768, 384, 192 hidden units with Tanh nonlinearity" |
| Bias-only classifier, InferSent | "shallow linear classifier with 512 and 512 hidden units" (wording as printed) |
| Bias-only classifier, HANS | nonlinear classifier with 6 and 6 hidden units, Tanh, on five feature groups: all hypothesis words in the premise; hypothesis is a contiguous subsequence; hypothesis is a subtree of the premise parse; shared tokens divided by premise length; cosine similarity of BERT pooled token representations with min, mean and max pooling. Contradiction and neutral share one weight in the bias-only loss |
| Which representation the bias-only classifier reads | not stated in the paper beyond "only uses hypothesis sentences" / "only claims as input"; the stop-gradient sentence implies it can sit on the shared encoder |
| $\gamma$ in Section 4 | $\gamma=2$ "through all experiments", untuned |
| $\gamma$ tuned on HANS | sweep $\{0.5,1,2,3,4\}$, selected $\gamma=3$ (selected on the test set, as Clark et al. did, because HANS has no dev set) |
| Transfer, BERT | $\gamma,\alpha\in\{0.4,0.6,0.8,2,3,4,5\}$, chosen on each target's dev set; selected values not stated |
| Transfer, InferSent | $\gamma,\alpha\in\{0.02,0.05,0.1,0.6,2.0,4.0,5.0\}$, $\beta\in\{0.05,0.2,0.4,0.8,1.0\}$; selected values not stated |
| Runs | HANS (Table 4): mean and standard deviation over 4 runs. ± values are also printed for the per-heuristic HANS numbers (Table 8) and for the MNLI mismatched hard set of the syntactic-bias and joint models (Table 5); the run count for those is not stated. Elsewhere single numbers |
| MNLI hard sets | built by the authors: a fastText hypothesis-only classifier, rows it gets wrong |

## 5. Results

**Table 1 (p. 5), FEVER, BERT.** Dev / symmetric test: CE 85.99/56.49; RUBi 86.23/57.60 (+1.1); Schuster et al. 84.6/61.6 (+5.1); DFL 83.07/64.02 (+7.5); PoE 86.46/66.25 (+9.8).

**Table 2 (p. 6), SNLI test / hard.**

| Loss | BERT | InferSent |
|---|---|---|
| CE | 90.53/80.53 | 84.24/68.91 |
| RUBi | 90.69/80.62 (+0.1) | 83.93/69.64 (+0.7) |
| AdvCls (Belinkov et al.) | — | 83.56/66.27 (−2.6) |
| AdvDat (Belinkov et al.) | — | 78.30/55.60 (−13.3) |
| DFL | 89.57/83.01 (+2.5) | 73.54/73.05 (+4.1) |
| PoE | 90.11/82.15 (+1.6) | 80.35/73.69 (+4.8) |

**Table 3 (p. 6), MNLI mismatched / hard.** BERT dev: CE 84.53/77.55; RUBi 85.17/78.63 (+1.1); DFL 84.85/78.92 (+1.4); PoE 84.85/79.23 (+1.7). InferSent dev: CE 69.99/56.53; RUBi 70.53/58.08 (+1.5); DFL 61.12/59.05 (+2.5); PoE 65.85/59.14 (+2.6). BERT test: CE 83.51/75.75; PoE 83.47/76.83 (+1.1).

**Table 7 (p. 12), MNLI matched / hard.** BERT dev: CE 84.41/76.56; RUBi 84.48/77.13 (+0.6); DFL 83.72/77.37 (+0.8); PoE 84.58/78.02 (+1.5). InferSent dev: CE 69.97/57.03; RUBi 70.51/57.97 (+0.9); DFL 60.78/57.88 (+0.9); PoE 66.02/59.37 (+2.3). BERT test: baseline 84.11/75.88; PoE 84.11/76.81 (+0.9).

**Table 4 (p. 6), syntactic bias: MNLI matched dev / HANS, BERT.**

| Loss | MNLI | HANS | $\Delta$ |
|---|---|---|---|
| CE | 84.51 | 61.88 ± 1.9 | |
| RUBi | 84.53 | 61.76 ± 2.7 | −0.1 |
| Reweight (Clark et al.) | 83.54 | 69.19 | +7.3 |
| Learned-Mixin (Clark et al.) | 84.29 | 64.00 | +2.1 |
| Learned-Mixin+H (Clark et al., tuned) | 83.97 | 66.15 | +4.3 |
| PoE | 84.19 | 66.31 ± 0.6 | +4.4 |
| DFL ($\gamma=2$) | 83.95 | 69.26 ± 0.2 | +7.4 |
| DFL ($\gamma=3$, tuned) | 82.76 | 71.95 ± 1.4 | +10.1 |

**Table 8 (p. 12), HANS by heuristic (constituent / lexical overlap / subsequence).**

| Loss | Gold entailment (heuristic right) | Gold non-entailment (heuristic wrong) |
|---|---|---|
| CE | 98.98/96.41/99.72 | 20.12/48.86/7.18 |
| RUBi | 99.22/95.59/99.50 | 21.89/46.82/7.58 |
| DFL | 90.90/84.78/94.33 | 50.20/71.06/24.28 |
| PoE | 97.24/92.16/98.58 | 36.08/59.18/14.63 |

**Table 5 (p. 7), joint debiasing: MNLI mismatched dev / hard / HANS.** CE 84.53 / 77.55 / 61.88 ± 1.9. Hypothesis bias only: PoE 84.85 / 79.23 (+1.7) / 60.43 (−1.5); DFL 84.85 / 78.92 (+1.4) / 60.63 (−1.2). Syntactic bias only: PoE 84.55 / 77.90 ± 0.3 (+0.4) / 66.31 ± 0.6 (+4.4); DFL 84.30 / 77.66 ± 0.6 (+0.1) / 69.26 ± 0.2 (+7.4). Joint: PoE 84.39 / 78.61 ± 0.1 (+1.1) / 68.04 ± 1.2 (+6.2); DFL 84.49 / 78.36 ± 0.4 (+0.8) / 69.10 ± 0.7 (+7.2).

**Table 6 (p. 8), transfer from SNLI, BERT (CE / DFL / PoE).** SICK 57.05/57.91/57.28; ADD1 87.34/88.89/87.86; DPR 49.50/50.68/50.14; SPR 59.85/61.41/62.45; FN+ 53.16/54.77/53.51; JOCI 50.06/51.13/50.85; MPE 69.50/70.2/70.1; SciTail 67.64/69.33/71.40; GLUE diagnostic 54.08/54.80/54.71; QQP 67.78/69.28/68.61; MNLI 74.40/73.58 (−0.8) / 73.61 (−0.8); MNLI-M 73.98/74.0 (0.0) / 73.49 (−0.5). Gains on the ten other sets are +0.2 to +3.8.

**Table 9 (p. 13), transfer from SNLI, InferSent (CE / DFL / PoE; relative %).** SICK 54.09/55.00/55.79; ADD1 75.19/78.29/77.00; DPR 49.95/50.59/49.95; SPR 41.31/47.95/50.50; FN+ 48.65/49.58/49.35; JOCI 46.47/46.48/47.53; MPE 60.60/60.70/61.80; SciTail 64.25/65.19/63.17; GLUE 48.73/46.83/49.09; QQP 61.80/66.24/66.36; MNLI 56.99/56.70/56.59; MNLI-M 57.01/57.75/57.84. Average relative change: DFL +2.57%, PoE +3.39%; the re-implemented Belinkov et al. methods M1 and M2 −2.36% each.

## 6. Ablations, failure modes, limitations

- **Effect of $\gamma$ (Figures 2 and 4, read off the plots, approximate).** InferSent: SNLI test falls from about 84 at $\gamma=0$ to 73 at $\gamma=2$ and 50 at $\gamma=6$; the hard set rises from about 68 to 72–73 for $\gamma\approx1$–$3.5$ (with a dip to about 70.5 at $\gamma=3$), then falls to about 65.5 at $\gamma=6$. BERT: SNLI test about 90.8 at small $\gamma$, 89.6 at 2, 87.5 at 3, 86 at 4, 82.5 at 7; hard set about 81, peaking near 83 at $\gamma=2$–3, back to 81 at 7. The hard-set gain saturates while in-domain accuracy keeps falling.
- **Where in-domain accuracy drops.** Section 2 table: up to −10.70 (DFL, InferSent, SNLI). On HANS rows where the heuristic is right, DFL loses 8.1/11.6/5.4 points and PoE 1.7/4.3/1.1. PoE is the gentler method throughout.
- **Several bias models (Table 5).** Debiasing against one bias does not transfer to the other: hypothesis-debiased models lose 1.2–1.5 on HANS, and syntactic-debiased models gain only +0.1 to +0.4 on the MNLI hard set (the paper calls this "a slight improvement"). Joint training recovers most of each single gain (hard set +1.1/+0.8 against +1.7/+1.4; HANS +6.2/+7.2 against +4.4/+7.4, PoE/DFL).
- **Clark et al.** Their two-stage PoE "gets similar results"; their Reweight is DFL with $\gamma=1$ (69.19 against 69.26 for $\gamma=2$); tuned DFL beats their best by 2.8 points on HANS, with model selection on the test set for both.
- **RUBi** gives −0.1 to +1.5 everywhere: the sigmoid mask barely debiases.
- **What the bias-only model captures (Figure 3, approximate).** Pearson correlation between each model's per-example cross-entropy and the bias-only model's: SNLI baseline 0.37, RUBi 0.37, DFL 0.29, PoE 0.28; MNLI 0.275/0.28/0.245/0.23; HANS 0.725/0.72/0.52/0.645. The paper states for MNLI (PoE) and HANS (DFL) that the method with the lower correlation also scores higher on the unbiased set; it makes no such statement for SNLI, where the two bars are within about 0.005 and the BERT hard-set result goes the other way (DFL 83.01, PoE 82.15). Which base model Figure 3 uses is not stated. The bias-only model's own accuracy is not reported.
- **Not reported:** the accuracy of the ensemble $f_C$ in-domain; any cross-fitting; small-data behaviour; variance outside the HANS experiments (the only other ± values are the MNLI hard-set column of Table 5 for the syntactic-bias and joint models).
- **Stated limitation:** the bias must be known in advance.

## 7. Recipe for our case

Notation: $q_i$ is the question (QuestionId or question text), $x_i$ the full input (photo, question, transcript), 76 classes (75 codes and NONE), $p_i=\mathrm{softmax}(f_M(x_i))$, $b_i=\mathrm{softmax}(f_B(q_i))$.

1. **PoE training logits** (supported by the paper; Eq. 2 with our $f_B$):
$$z_i^{\text{train}}=\log\mathrm{softmax}(f_M(x_i))+\log\mathrm{softmax}(f_B(q_i)),\qquad \mathcal{L}_{\text{PoE}}=-\frac{1}{N}\sum_{i}\log\mathrm{softmax}(z_i^{\text{train}})^{y_i}$$
Per-row gradient on the base logits: $\partial\mathcal{L}_i/\partial f_M^k=\mathrm{softmax}(z_i^{\text{train}})^k-\delta_{y_ik}$.

2. **Inference is logit subtraction** (use of $f_M$ alone is supported by the paper; the identity is our derivation). Softmax ignores per-row constants, so $z_i=f_M(x_i)+\log b_i+c_i$. Therefore
$$\mathrm{softmax}(z_i-\log b_i)=\mathrm{softmax}(f_M(x_i)),\qquad p_i^k\propto\frac{\mathrm{softmax}(z_i)^k}{b_i^k}$$
Training the sum and testing with $f_M$ is the ensemble with $\log b$ subtracted. If the ensemble fits $P(y\mid x,q)$ and $b\approx P(y\mid q)$, then $f_M^k\approx\log\frac{P(k\mid x,q)}{P(k\mid q)}+c_i$ (up to a per-row constant $c_i$ that softmax removes): the evidence in the working beyond the question. Unlike our rejected CPU proxy, the subtraction is in the training loss, so $f_M$ is trained to carry the residual. The damage to expect on no-error and seen-pair rows is the same in kind (Table 8, entailment rows).

3. **DFL weights for 76 classes** (formula supported by the paper; the slice numbers are our extrapolation):
$$w_i=(1-b_i^{y_i})^{\gamma},\qquad \mathcal{L}_{\text{DFL}}=-\frac{1}{N}\sum_i w_i\log\mathrm{softmax}(f_M(x_i))^{y_i}$$
   - No-error row, prior gives NONE $b=0.7$, $\gamma=2$: $w=0.3^2=0.09$, an 11-fold cut. ($\gamma=1$: 0.30; $\gamma=3$: 0.027.)
   - Seen-pair row with $b=0.3$: $w=0.49$.
   - Hard row, truth has only smoothing mass, $b=0.01$: $w=0.98$.
   - With the dev mix (33.6% no-error, 31.2% seen, 35.2% hard) and these illustrative $b$ values, the weight mass (the loss mass if per-row losses were equal) becomes about 6% / 29% / 65% and the mean weight falls to 0.53, so renormalise $w$ to mean 1 (the paper does not). The real $b$ values must be measured.
   - PoE on the same no-error row is gentler: with the base model at 0.5 on NONE and 0.5 on one competitor, and the prior's other 0.3 on that competitor, the ensemble gives NONE $\frac{0.7\times0.5}{0.7\times0.5+0.3\times0.5}=0.7$, so the gradient magnitude on the true logit is 0.3 against 0.5 for plain cross-entropy (DFL at $\gamma=2$: $0.09\times0.5=0.045$). If the prior's other 0.3 is spread over classes the base model gives no mass, the ensemble gives NONE 1 and the PoE gradient is 0.
   - Start at $\gamma\in\{1,2\}$; the paper's SNLI curves (Figures 2 and 4) show values above 2–3 buy nothing on the hard set. On HANS the tuned $\gamma=3$ did beat $\gamma=2$ (71.95 against 69.26), but it was selected on the test set and cost −1.75 in-domain.

4. **Cross-fit $b$; do not train $f_B$ jointly** (our extrapolation; the paper trains $f_B$ in-sample on hundreds of thousands of rows). With about 3.2 rows per question, any in-sample question model sees the row's own label. Use the leave-one-row-out count prior, with $n_{q,k}$ the training count of code $k$ on question $q$, $\pi_k$ a global or topic prior computed without row $i$, and $\lambda$ a smoothing strength:
$$b_i^k=\frac{n_{q_i,k}-\mathbb{1}[k=y_i]+\lambda\,\pi_k}{n_{q_i}-1+\lambda}$$
   - In-sample, a singleton pair on a 3-row question gets $b^{y}\approx\frac13$ ($w=0.44$ at $\gamma=2$); leave-one-out gives roughly $\lambda\pi_y/(2+\lambda)\approx0$ ($w\approx1$), which is what a test row of that type will face.
   - A jointly trained $f_B$ on question text would memorise the 1,500 rows. Rows of one question share the same input, so it cannot reach $b^{y_i}\to1$ everywhere: it converges to the in-sample frequency $b^{y_i}\to n_{q_i,y_i}/n_{q_i}$, which counts the row's own label. That is 1 on single-row questions and on questions whose rows all share one code, where it switches off the base model's gradient (the paper's "extreme case"), and it is inflated on every other row.
   - Fold over rows, not questions: the prior must see the question's other rows, as it does at test.
   - Freezing a precomputed $b$ is the Clark et al. two-stage form, which the paper reports as giving similar results (indirectly supported).

5. **Deploy only as the hard-row specialist** (our extrapolation, backed by the paper's in-domain numbers).
   - The model is trained to lack the prior that is right on two thirds of our test rows; the paper's weak-model drops are 4 to 11 points.
   - DFL also starves NONE rows, so the specialist's NONE decision should not be trusted; use it to rank error codes.
   - Use: as a fusion member inside the existing cross-fitted fusion, or with the prior added back at strength $\alpha$, $\mathrm{softmax}(f_M+\alpha\log b)$, where $\alpha=1$ is the trained ensemble and $\alpha=0$ the debiased model, chosen on dev by slice.
   - Pass mark to set before any run: alone on hard rows above 0.500, with a target near 0.61 (Qwen3.8-27B alone); fused, no loss on no-error and seen-pair rows.
   - Training this needs a GPU run, so it needs the user's approval first.

6. **Why this could still disappoint** (our extrapolation). Masking the question text gave +0.005 [−0.021, +0.027], and the review's reading is that hard rows need new evidence, not re-weighting. PoE changes what the gradient asks of the photo; it does not make the photo easier to read.

## 8. Pages inspected as images

Pages 1, 2 (Figure 1), 3, 4, 5, 6, 7, 8, 12 and 13 were read as images; all equations and tables above were transcribed from them. The supplied PNGs of pages 1–7 have a transparent background and were flattened onto white copies first. Pages 9–11 (conclusion and references) were read as text only.

Verified against page images: 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13
