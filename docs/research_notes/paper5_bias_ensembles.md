# Paper 5: ensemble-based debiasing with a bias-only model (Clark, Yatskar, Zettlemoyer)

## 1. Citation and summary

Christopher Clark, Mark Yatskar, Luke Zettlemoyer. "Don't Take the Easy Way Out: Ensemble Based Methods for Avoiding
Known Dataset Biases." arXiv:1909.03683v1, 9 September 2019. 14 pages. Code: github.com/chrisc36/debias.

The paper first trains a "bias-only" model on known shortcut features, then trains the main model inside an
ensemble with that frozen bias model, so the main model gets little gradient for what the shortcut already
explains. At test time the main model is used alone. On five tasks whose test sets remove or reverse the shortcut it
gains up to 12.9 points out of domain (VQA-CP 39.18 to 52.05), usually at a cost in domain.

## 2. Which bottleneck it addresses, and how directly

**Target.** Our classifiers answer "the usual code for this question": 0.867 on seen-pair rows, 0.500 on hard rows,
and the same first code for two students on one question 42% of the time against 27% in the labels. Our question-only
prior (0.485–0.497 MAP@3) is the paper's bias-only model, and the paper's own VQA-CP bias model (question type only) is
the closest published analogue. Figure 2's low-gate cases (a common question type whose answer is unusual) are our
group B rows: known question, new code.

**How directly: the mechanism fits, the evaluation setting does not.**

- Every out-of-domain test set in the paper has the bias randomised, reversed or made misleading (Table 1). Ours keeps
  it: 95.7% of test rows are on training questions and the prior is right on about two thirds of rows. Our test set
  is the paper's *in-domain* column for those rows and its out-of-domain column only for the hard third.
- The paper reports the in-domain cost and it is large when the bias is strong. Synthetic MNLI, where the bias token
  matches the label 80% of the time in domain on Indicator (Excluder: 3%, so it rules a class out; Dependent: 90% on
  the 80% of rows flagged reliable, random on the rest): with the ensemble methods the main model alone loses 2.9 to
  11.3 points in domain (Table 2),
  and with Learned-Mixin +H it lands just below a model that never saw the bias (76.65–78.57 against 78.94, or 78.86
  on Dependent). SQuAD
  dev loses 2.0 to 6.5 F1 for the methods that help. Section 5.6 says the trade-off is the usual outcome, with
  TriviaQA-CP the exception.
- On the rows where the shortcut gives the right answer, the cost is far larger than the averages show (Table 8, HANS
  entailment rows): 97.83 to 67.18, 99.67 to 61.05, 97.28 to 47.13 for the recurrent model with Learned-Mixin +H.
  These rows are the analogue of our seen-pair and no-error rows.
- The paper never reports the ensemble $\hat p_i$ at test time, in or out of domain. Not stated in the paper.
- The entropy weight was selected on the test set (section 4.2). The headline +H numbers are optimistic.
- The only training-set sizes the paper states for its debiasing experiments are 60,133 and 52,953 (Table 9; the
  1,100 hand-labelled questions in appendix E train the question-type classifier, not a debiased model). We have
  1,500. Section 5.6 warns
  that the method effectively shrinks the training data.
- Debiasing removes a shortcut; it does not supply the diagnostic skill. The paper's main models can solve the task
  from the remaining input. Whether ours can read a new error with no example of the pair is not something this paper
  answers.

Verdict: relevant only as a way to build a hard-row specialist. As a replacement for the current system it would lose.

## 3. Exact method

**Notation (section 3.2.1).** Training examples $\langle x_1, x_2, \dots, x_n\rangle$, labels
$y_i \in \{1, 2, \dots, C\}$. A pre-trained bias-only predictor $h$ with
$h(x_i) = b_i = \langle b_{i1}, b_{i2}, \dots, b_{iC}\rangle$, where $b_{ij}$ is its probability of class $j$ for
example $i$. The main model $f$ with parameters $\theta$: $f(x_i, \theta) = p_i$, a distribution over the $C$ classes.

**General approach (3.2.2).** For each example a combined distribution $\hat p_i$ is computed from $p_i$ and $b_i$.
The loss is computed on $\hat p_i$ and gradients flow through $f$ only (Figure 1 shows no gradient arrow into the
bias-only model). "During evaluation $f$ is used alone." The loss function itself is not printed; cross-entropy on
$\hat p_i$ is implied, i.e. $\mathcal{L}_i = -\log \hat p_{i y_i}$ (our reading).

**Reweight baseline (4.1).** No ensemble. Train $f$ normally with example weight

$$\omega_i = 1 - b_{i y_i}.$$

**Bias Product (3.2.3).**

$$\hat p_i = \mathrm{softmax}\big(\log(p_i) + \log(b_i)\big), \qquad \text{equivalently } \hat p_i \propto p_i \circ b_i$$

with $\circ$ element-wise multiplication. Justification: let $x^b$ be the bias features and $x^{-b}$ a view holding
everything else, assumed conditionally independent given the label $c$:

$$
\begin{aligned}
p(c \mid x) &= p(c \mid x^{b}, x^{-b}) && (1)\\
&\propto p(c \mid x^{-b})\, p(x^{b} \mid c, x^{-b}) && (2)\\
&= p(c \mid x^{-b})\, p(x^{b} \mid c) && (3)\\
&= p(c \mid x^{-b})\, \frac{p(c \mid x^{b})\, p(x^{b})}{p(c)} && (4)\\
&\propto p(c \mid x^{-b})\, \frac{p(c \mid x^{b})}{p(c)} && (5)
\end{aligned}
$$

The $p(c)$ factor is dropped in practice (uniform classes on MNLI; pointer outputs on QA; class priors unwanted on
VQA). The paper notes a view $x^{-b}$ usually cannot be built directly, which is why it trains the product instead.

**Learned-Mixin (3.2.4).**

$$\hat p_i = \mathrm{softmax}\big(\log(p_i) + g(x_i)\,\log(b_i)\big), \qquad g(x_i) = \mathrm{softplus}(w \cdot h_i), \qquad \mathrm{softplus}(x) = \log(1 + e^{x})$$

$w$ is a learned vector trained with $\theta$; $h_i$ is the main model's last hidden layer for $x_i$. Softplus keeps
$g \ge 0$ so the model cannot reverse the bias. $g(x_i) = 1$ gives Bias Product. Known failure: the model absorbs the
bias into $p_i$ and sets $g(x_i) = 0$.

**Learned-Mixin +H (3.2.5).** Add to the loss

$$R = w\,H\big(\mathrm{softmax}(g(x_i)\,\log(b_i))\big), \qquad H(z) = -\sum_j z_j \log(z_j)$$

Here $w$ is a scalar hyperparameter (the paper reuses the letter $w$; it is not the vector inside $g$). Low entropy
forces the scaled bias term to be non-uniform, so it cannot be switched off.

**Bias-only model training (3.1, 4).** Trained first, separately, on the same training set, on the bias features
only, then held fixed. Its predictions on the training set are **in-sample** (our reading: the paper says both models
are trained on the train set and never mentions cross-fitting or held-out bias predictions). Footnote 2
says care is needed to limit overfitting but that the bias model is usually simple enough for this not to matter.

**Two-class reduction (5.2).** VQA uses a multi-label objective, so each answer is treated as a two-class problem. No
formula is printed. Under that statement the Learned-Mixin form is
$\hat p_{ij} = \sigma\big(\mathrm{logit}(p_{ij}) + g(x_i)\,\mathrm{logit}(b_{ij})\big)$ (our derivation). Footnote 4:
because the bias sometimes gives zero probability, $\sigma(\alpha)$ with a learned parameter $\alpha$ is added to the
bias probabilities (the paper does not say whether $\alpha$ is a single scalar or per answer, and prints no formula).

**Extractive QA (4).** Independent bias-only models for start and end tokens, ensembled separately with the start
and end distributions. The hidden state for $g$: ReLU layer on question and passage embeddings, then max-pooling.

## 4. Hyperparameters per task

| Task | Bias-only model | Main model | Entropy weight $w$ (Table 7) |
|---|---|---|---|
| Synthetic MNLI: Indicator / Excluder / Dependent | Predicts the label from the first hypothesis token | Recurrent co-attention (ESIM-like) | 0.01 / 0.005 / 0.005 |
| VQA-CP v2 | One of 65 question types as a categorical input, same multi-label objective as the main model | BottomUpTopDown (hengyuan-hu implementation) | 0.36 |
| HANS | Shallow linear classifier on five features: hypothesis is a subsequence of the premise; all hypothesis words in premise; share of hypothesis words in premise; mean and max of the minimum fasttext cosine distances between premise and hypothesis words. Equal mass forced on neutral and contradiction; data reweighted so entailment and non-entailment have equal total weight | Co-attention; BERT base uncased | 0.03; 0.03 |
| Adversarial SQuAD | TF-IDF score of each sentence against the question (all tokens in a sentence share one score); "Filtered" drops pronouns and numbers | Modified BiDAF | 2.0 (both) |
| TriviaQA-CP Location / Person | Stanford NER tags of passage tokens; untagged tokens get the average score of tagged ones | Larger modified BiDAF | 0.4 / 0.2 |

- Co-attention (appendix B): character CNN plus fasttext crawl embeddings, shared BiLSTM, co-attention, second
  shared BiLSTM, max-pool, fully connected, 3-way softmax. Variational dropout 0.2 between layers and on recurrent
  states; 30 epochs; Adam; batch 32; learning rate decayed by 0.999 every 100 steps; 200-dimensional LSTMs;
  50-dimensional fully connected layer.
- Modified BiDAF (appendix D): variational dropout 0.2; Adam; batch 45; decay 0.999 every 100 steps. SQuAD: 200
  fully connected, 100 LSTM. TriviaQA: 256 fully connected, 128 LSTM, highway connections, recurrent dropout 0.2.
- BERT: default hyperparameters. VQA model hyperparameters: not stated in the paper.
- Initial learning rates, QA epochs, the bias-only models' own optimisers and the grid searched for $w$: not stated
  in the paper. No early stopping and no other tuning of the main models.

## 5. Results (main model alone; in-domain beside out-of-domain)

**Table 2 (p. 7), synthetic MNLI accuracy. "Acc." = bias randomised; "w/Bias" = in domain.**

| Method | Indicator Acc. | w/Bias | Excluder Acc. | w/Bias | Dependent Acc. | w/Bias |
|---|---|---|---|---|---|---|
| None | 69.36 | 86.49 | 68.06 | 83.56 | 63.23 | 87.90 |
| Reweight | 75.44 | 82.74 | 70.36 | 83.29 | 69.81 | 85.50 |
| Bias Product | 76.27 | 81.32 | 77.33 | 80.41 | 71.85 | 84.98 |
| Learned-Mixin | 76.29 | 81.35 | 77.80 | 78.86 | 75.75 | 77.70 |
| Learned-Mixin +H | 76.77 | 77.65 | 77.90 | 78.57 | 75.79 | 76.65 |
| Unbiased training (upper bound) | 78.94 | 78.94 | 78.94 | 78.94 | 78.86 | 78.86 |

**Table 3 (p. 7), VQA-CP v2 test accuracy.** None 39.18; Reweight 40.06; Bias Product 39.93; Learned-Mixin 48.69;
Learned-Mixin +H 52.05; Ramakrishnan et al. 41.17; Grand and Belinkov 42.33. In-domain VQA: not stated in the paper.

**Table 4 (p. 7), HANS / MNLI matched dev accuracy.**

| Method | Co-attention HANS | MNLI | BERT HANS | MNLI |
|---|---|---|---|---|
| None | 50.58 | 78.73 | 62.40 | 84.24 |
| Reweight | 52.85 | 77.03 | 69.19 | 83.54 |
| Bias Product | 53.69 | 76.63 | 67.92 | 82.97 |
| Learned-Mixin | 51.65 | 78.05 | 64.00 | 84.29 |
| Learned-Mixin +H | 53.35 | 74.50 | 66.15 | 83.97 |

**Table 5 (p. 9), SQuAD F1.**

| Method | Filtered: AddSent | AddSentOne | Dev | TF-IDF: AddSent | AddSentOne | Dev |
|---|---|---|---|---|---|---|
| None | 42.54 | 53.91 | 80.61 | 42.54 | 53.91 | 80.61 |
| Reweight | 41.55 | 53.06 | 80.59 | 42.74 | 53.83 | 80.51 |
| Bias Product | 47.17 | 57.74 | 78.63 | 44.41 | 55.73 | 78.22 |
| Learned-Mixin | 42.25 | 53.51 | 80.39 | 42.00 | 53.46 | 80.46 |
| Learned-Mixin +H | 51.84 | 60.66 | 75.94 | 48.30 | 58.26 | 74.14 |

**Table 6 (p. 9), TriviaQA-CP exact match.**

| Method | Location CP | Dev | Person CP | Dev |
|---|---|---|---|---|
| None | 41.23 | 59.27 | 39.69 | 55.26 |
| Reweight | 40.14 | 59.18 | 39.96 | 55.38 |
| Bias Product | 44.42 | 60.02 | 40.58 | 55.20 |
| Learned-Mixin | 41.15 | 61.64 | 41.31 | 56.08 |
| Learned-Mixin +H | 47.77 | 57.74 | 44.37 | 54.83 |

**Table 8 (p. 13), HANS by heuristic, mean of 8 runs.** Columns: rows where the shortcut is right (entailment:
lexical / subsequence / constituent) and wrong (non-entailment, same order).

| Model, method | Entailment | Non-entailment |
|---|---|---|
| Co-attention, None | 97.83 / 99.67 / 97.28 | 1.37 / 3.68 / 3.68 |
| Reweight | 80.10 / 77.84 / 73.76 | 15.68 / 34.27 / 35.44 |
| Bias Product | 77.89 / 76.61 / 70.95 | 17.89 / 35.11 / 43.71 |
| Learned-Mixin | 94.84 / 97.25 / 91.19 | 3.69 / 9.57 / 13.37 |
| Learned-Mixin +H | 67.18 / 61.05 / 47.13 | 27.41 / 56.82 / 60.53 |
| BERT, None | 96.30 / 99.58 / 99.30 | 49.03 / 7.88 / 22.30 |
| Reweight | 67.93 / 84.34 / 80.97 | 77.44 / 44.87 / 59.57 |
| Bias Product | 53.67 / 69.47 / 70.88 | 81.34 / 62.93 / 69.23 |
| Learned-Mixin | 95.64 / 99.52 / 99.14 | 55.41 / 8.34 / 25.96 |
| Learned-Mixin +H | 91.98 / 98.20 / 97.98 | 64.99 / 13.25 / 30.48 |

Tables 9 and 10 (pp. 13–14) cover TriviaQA-CP construction only (1,992 / 2,865 test questions; question-type
classifier 94.46% accurate).

## 6. Ablations, failure modes, limitations

- **Bias Product over-compensates when independence fails (5.1, Dependent).** When a second feature tells the main
  model the bias is unreliable, it learns to subtract the bias model's output. On rows where the appended indicator
  is 1 (the bias token is random) it then gives the bias-indicated class
  22.5% on average where an unbiased model gives 33%. Learned-Mixin gives 34.5%, with
  $g(x_i) = 0.00 \pm 0.0001$ "when the indicator is turned off" (the paper's wording; read as the rows where the bias
  is unreliable) and $1.91 \pm 0.285$ otherwise. The paper adds that the entropy penalty appears unnecessary on this
  dataset because $g$ does not go to zero.
- **Learned-Mixin ignores the bias without the penalty.** Mean $g$: 0.13 against 5.16 with +H (AddSent, filtered
  bias); 0.25 against 5.01 (TriviaQA Location dev). Without +H it gains nothing on SQuAD or TriviaQA Location.
- **The gate tracks bias reliability.** On VQA-CP (test data, "the learned-mixin ensemble"; the paper does not
  say whether with or without +H), Spearman correlation 0.77
  between $g(x_i)$ and the bias model's
  expected accuracy $\sum_j s_{ij} b_{ij} / \sum_j b_{ij}$ ($s_{ij}$ the score of class $j$ on example $i$). Figure 2
  (training data): 4.65 / 5.96
  (without / with +H) for a typical-answer photo, 0.00 / 0.48 for an atypical one.
- **In-domain drop.** See section 2. The paper ties it to how useful the bias is in domain (5.6). That it scales with
  $w$ is our inference, not a statement in the paper (not found on re-check; no sweep is reported, and the largest
  drop, 11.25 points on Dependent, comes with the smallest weight, 0.005).
- **Sensitivity to $w$.** No sweep is reported. Chosen values span 0.005 to 2.0 and were picked on the test set.
- **Weak against strong bias model.** The better-targeted filtered TF-IDF bias beats plain TF-IDF (51.84 against 48.30
  AddSent). A weak bias model leads the gate to scale it up (5.16). A bias model that is too strong is not tested;
  section 5.6 gives the argument: where the bias already answers correctly, the ensemble is already right, the main
  model has nothing to learn, and the training set is in effect smaller. The paper says a real fix would need
  per-example regularisation of how the model solves each example.
- **Reweight** needs some bias-free examples: on Excluder the bias gives about 50% to the truth almost everywhere, the
  weights are nearly uniform and the gain is 2.3 points. It is the best method for BERT on HANS (69.19).
- **Variance.** BERT on HANS varied between runs; the authors suspect the ensembles add instability.

## 7. Recipe for our case

Symbols: row $i$, question $q_i$, topic $t_i$, gold code $y_i$ among $C = 76$ classes, main-model logits $z_i$ with
$p_i = \mathrm{softmax}(z_i)$.

**7.1 Bias-only model, cross-fitted and smoothed.** *Our extrapolation* (the paper uses in-sample $b_i$; footnote 2
only flags the overfitting risk).

$$b_i(c) = \frac{n^{(-i)}_{q_i,c} + \alpha\,\pi^{(-i)}_{t_i}(c)}{n^{(-i)}_{q_i} + \alpha}, \qquad b_i(c) \leftarrow \max\big(b_i(c), \epsilon\big)\ \text{then renormalise}$$

$n^{(-i)}_{q,c}$ counts training rows of question $q$ with code $c$ excluding row $i$; $\pi^{(-i)}_{t}$ is the
topic-level code distribution, also excluding row $i$. Choose $\alpha$ and $\epsilon$ by the leave-one-out log-loss
of $b$ alone; it should reproduce the measured 0.485–0.497. For test rows use all training rows.

Why in-sample $b_i$ is degenerate here. For the product loss $\mathcal{L}_i = -\log \hat p_{i y_i}$,

$$\frac{\partial \mathcal{L}_i}{\partial z_{ij}} = \hat p_{ij} - \mathbb{1}[j = y_i], \qquad \hat p_{ij} = \frac{p_{ij}\, b_{ij}}{\sum_k p_{ik}\, b_{ik}}.$$

With about 3.2 rows per question, an in-sample count gives $b_{i y_i} \ge 1/n_{q_i}$ on every row and
$b_{i y_i} = 1$ on single-row or unanimous questions, so $\hat p_{i y_i} \to 1$ and the gradient is zero whatever the
photo shows. The bias model has memorised the label, not learned a prior. Worse, the case we care about never occurs
in training: at test a hard row has $b_{i y_i} \approx \epsilon$, but in-sample it has at least $1/n_q$. Leave-one-out
makes every row whose (question, code) pair is unique look exactly like a test hard row. The paper's VQA bias model
has only 65 input values, which is why it could skip this. Softening zero bias probabilities is *supported by the
paper* (footnote 4 adds a learned $\sigma(\alpha)$ to the bias probabilities); the fixed floor $\epsilon$ and the
topic-prior smoothing are *our extrapolation*.

**7.2 Training loss.** Forms *supported by the paper*; applying them to a 1,500-row fine-tune is *our extrapolation*.
In order of risk:

1. Reweight: weight $\omega_i = 1 - b_{i y_i}$ on the usual loss. Small in-domain cost in the paper (from a 0.1-point
   gain to a 3.8-point loss across Tables 2, 4, 5 and 6), no added parameters, and we do have bias-free rows (the
   hard third).
2. Bias Product: $\mathcal{L}_i = -\log \mathrm{softmax}(z_i + \log b_i)_{y_i}$. This is the paper's form with the
   $p(c)$ divisor of equation 5 dropped; the paper's reasons for dropping it (uniform classes, pointer outputs, or
   wanting class priors removed) only partly apply to 76 unbalanced codes, so the main model alone will lack the
   class prior (*our extrapolation*).
3. Learned-Mixin +H, with $u$ the learned vector and $h_i$ the pooled last hidden state:

$$\mathcal{L}_i = -\log \mathrm{softmax}\big(z_i + g(x_i)\log b_i\big)_{y_i} + w\,H\big(\mathrm{softmax}(g(x_i)\log b_i)\big), \qquad g(x_i) = \mathrm{softplus}(u \cdot h_i)$$

   For the pointwise yes/no ranker use the two-class form in section 3. Unlike the authors we can choose $w$
   honestly, on cross-fitted dev hard rows; search across the paper's range (0.005 to 2.0). Check that mean $g$ is
   high on seen-pair rows and near zero on hard rows, as in the Dependent experiment.

Our conditional-independence assumption is doubtful (the photo shows which question it is), which is the case where
the paper found Learned-Mixin better than Bias Product. Masking the question text failed for the same reason the
paper gives for not building $x^{-b}$: the bias cannot be cut out of the input. *Supported by the paper.*

**7.3 Deploy only as the hard-row specialist.** *Supported by the paper* for the direction, *our extrapolation* for
the size. A main model used alone gives up the bias: with Learned-Mixin +H it falls to 76.65–78.57 in domain on
synthetic MNLI against 83.56–87.90 for the biased model, and to 47–67% from 97–100% on shortcut-correct HANS rows
(recurrent model). Our prior is right on
about two thirds of rows, so the debiased model alone would lose most of what the seen-pair (0.867) and no-error
(0.911) rows earn. Keep the current system as the memoriser and add the debiased model as a fusion member whose
weight rises where the memoriser is unsure, with the gate cross-fitted. Options to gate on dev: fuse $f$ alone; or
re-attach the prior, $\mathrm{softmax}(\log p_i + \lambda \log b_i)$ with $\lambda$ cross-fitted (the paper's
equation 5 says that, under conditional independence, $p(c \mid x) \propto p(c \mid x^{-b})\,p(c \mid x^{b})/p(c)$,
so the product, with the $p(c)$ divisor the paper drops, is the in-distribution posterior; the paper never evaluates
it; *our extrapolation*).

Pass marks to predeclare (*our extrapolation*): hard rows above 0.500 alone and above the submitted system fused, no
loss on seen-pair and no-error rows after fusion, and the same-first-code rate for two students moving from 42%
towards 27%. Any GPU run needs the user's approval; a CPU proxy on frozen features comes first.

**7.4 Why our inference-time proxy collapsed.** Expected, on three counts, all *supported by the paper*.

1. The method is a training-time method. The main model is trained against the product and then used alone; nothing
   in the paper divides a prior out of an already-trained biased model. Our classifiers were trained without the
   ensemble, so the question prior is fused into their logits and cannot be separated by subtracting $\log b_i$.
2. Subtracting the bias is the failure the paper documents in the Dependent experiment: a model that cancels the
   bias model's output becomes anti-correlated with it (22.5% against 33%). An inference-time division imposes that
   anti-correlation by hand, and on seen-pair and no-error rows the bias-indicated code is the truth.
3. Even the properly trained method collapses on bias-aligned rows (Table 8 entailment columns; Table 2 "w/Bias").
   A proxy without retraining cannot do better than the real method. The same holds for dividing the prior out of the
   blend, where the best strength was zero.

**7.5 Limits on the expected gain.** *Our extrapolation.* Section 5.6 implies the main model learns mostly from rows
the bias gets wrong, roughly a third of 1,500. The paper's gains come where the main model can solve the task once
the shortcut is removed; our 0.19 reasoning gap to Claude on identical transcripts suggests part of the hard-row loss
is skill, which this method does not add.

## 8. Pages inspected as images

Pages 2 (Figure 1), 3 (notation, footnote 2), 4 (all equations), 5 (Table 1, sections 4.1–4.2), 6 (footnote 4), 7
(Tables 2–4, footnote 5), 8 (Figure 2), 9 (Tables 5–6, section 5.6), 12 (Table 7, appendices B–D), 13 (Tables 8–9).
Pages 1, 10, 11 and 14 as text only.

Verified against page images: 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14 (pages 10–11 are references and the appendix A heading; checked in the text extraction only)
