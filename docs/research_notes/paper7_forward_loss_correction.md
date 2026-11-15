# Paper 7: forward and backward loss correction with a noise transition matrix

## 1. Citation and summary

Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, Lizhen Qu. "Making Deep Neural Networks Robust
to Label Noise: a Loss Correction Approach." arXiv:1609.03683v2 [stat.ML], 22 March 2017, 9 pages. Venue: not stated
in the paper (the PDF carries only the arXiv stamp).

The paper trains networks on labels corrupted by class-conditional noise, described by a row-stochastic matrix $T$,
using two loss corrections: "backward" (multiply the loss vector by $T^{-1}$, an unbiased loss estimate) and "forward"
(multiply the softmax output by $T^{\top}$ before the loss, which keeps the clean minimiser). When $T$ is unknown it is
estimated from a network trained on the noisy labels, by reading the softmax at the most confident example of each
class. Forward correction with the true $T$ is usually the best method; with an estimated $\hat T$ the gain shrinks and
can vanish (CIFAR-100).

## 2. Which bottleneck it addresses, and how directly

Target: section 5.2 of `docs/experiment_review_2026-11-14.md`, "Claude's labels follow Claude's conventions, not the
teachers'". The 26,202 Claude-labelled rows are the noisy set $\tilde y$; the 1,500 teacher rows are clean $y$.

Direct fit:

- The forward loss is built for systematic, asymmetric flips. Claude putting NONE first on 70.9% of pool photos, and
  one "sloppy notation" code second on 17,021 of 26,202 answers, are flips of that kind.
- We are better placed than the paper's main setting. They estimate $T$ from noisy data alone (anchor points) and name
  that estimator as the bottleneck. We can count $T$ from paired (teacher, Claude) rows, which is what the paper does
  on Clothing1M.

Critical caveats:

- **Class-conditional assumption.** The paper assumes $p(\tilde y\mid y, x)=p(\tilde y\mid y)$ and calls
  instance-dependent noise "an open question". Our noise probably depends on the question and error family, so a single
  $T$ can only remove the average convention gap.
- **Sample size.** 33.7% NONE means about 505 of 1,500 pairs sit in the NONE row; the other 995 or so spread over 75
  codes, a mean of about 13 per row (about 11 when only the 1,199 fit rows may be used), and the sloppy-notation code
  has one. Each row has 75 free entries. Unsmoothed counting is unusable. The paper's own many-class case (CIFAR-100,
  500 images per class) is where its estimate failed.
- **It does not add coverage.** Only 30% of hard rows have their pair anywhere in Claude's pool labels, and 28% of
  test rows are on questions with no pool photos. Loss correction changes how existing rows are read, not which rows
  exist.
- **Prior evidence is mixed.** Our simple prior correction gave +0.027 on one Claude-trained model and −0.010 to
  +0.001 on the blend. $T$ is richer than a prior, but the same wash-out in the blend is possible.
- **Size of gain in the paper.** With an estimated matrix and low noise the change over plain cross-entropy is small:
  forward with $\hat T$ is +0.2 to +2.0 points at low asymmetric noise and −1.1 to +1.3 at symmetric noise (Table 2,
  our subtraction). Large gains need heavy asymmetric noise: there forward with $\hat T$ gains +11.9 to +24.0 on five
  of six settings and loses 1.2 on CIFAR-100, while the true $T$ gains +13.3 to +51.3.

## 3. Exact method

### 3.1 Notation (page 2–3)

$[c]\doteq\{1,\dots,c\}$; $\mathbf 1$ is the all-ones vector; $\Delta^{c-1}\subset[0,1]^c$ is the simplex. Features
$\mathcal X\subseteq\mathbb R^d$, labels $\mathcal Y=\{e^i: i\in[c]\}$ with $e^i$ the $i$th canonical vector. Data
$(x,y)\sim p(x,y)=p(y\mid x)\,p(x)$. The network is $h:\mathcal X\to\mathbb R^c$, $h=h^{(n)}\circ\dots\circ h^{(1)}$,

$$(\forall i\in[n-1])\quad h^{(i)}(z)=\sigma(W^{(i)}z+b^{(i)}),\qquad h^{(n)}(z)=W^{(n)}z+b^{(n)}$$

(the page prints the last layer with superscript $(i)$, a typo), with $x^{(i)}\doteq h^{(i)}(x^{(i-1)})$, $x^{(0)}\doteq x$.
The softmax output $e^{h_i(x)}/\sum_{k=1}^{c}e^{h_k(x)}$ is written $\hat p(y\mid x)\in\Delta^{c-1}$. Loss
$\ell:\mathcal Y\times\Delta^{c-1}\to\mathbb R$; cross-entropy is

$$\ell(e^i,\hat p(y\mid x))=-(e^i)^{\top}\log\hat p(y\mid x)=-\log\hat p(y=e^i\mid x)\tag{1}$$

and the vector loss $\boldsymbol\ell:\Delta^{c-1}\to\mathbb R^c$ is

$$\boldsymbol\ell(\hat p(y\mid x))=\big(\ell(e^1,\hat p(y\mid x)),\dots,\ell(e^c,\hat p(y\mid x))\big)^{\top}\in\mathbb R^c.\tag{2}$$

### 3.2 Noise model (page 3)

Each label $y$ is flipped to $\tilde y\in\mathcal Y$ with probability $p(\tilde y\mid y)$; features are untouched:

$$p(x,\tilde y)=\sum_{y}p(\tilde y\mid y)\,p(y\mid x)\,p(x),\qquad T\in[0,1]^{c\times c},\quad \forall i,j\;\; T_{ij}=p(\tilde y=e^j\mid y=e^i).$$

Rows are the true class, columns the noisy class. "The matrix is row-stochastic and not necessarily symmetric across
the classes."

### 3.3 Backward correction (Theorem 1, page 3)

Suppose $T$ is non-singular. Given a loss $\boldsymbol\ell$, the backward corrected loss is

$$\boldsymbol\ell^{\leftarrow}(\hat p(y\mid x))=T^{-1}\,\boldsymbol\ell(\hat p(y\mid x)).$$

Then the correction is unbiased:

$$\forall x,\quad \mathbb E_{\tilde y\mid x}\,\ell^{\leftarrow}(y,\hat p(y\mid x))=\mathbb E_{y\mid x}\,\ell(y,\hat p(y\mid x)),$$

$$\operatorname*{argmin}_{\hat p(y\mid x)}\mathbb E_{x,\tilde y}\,\ell^{\leftarrow}(y,\hat p(y\mid x))=\operatorname*{argmin}_{\hat p(y\mid x)}\mathbb E_{x,y}\,\ell(y,\hat p(y\mid x)).$$

(As printed, the left sides carry $y$ inside $\ell^{\leftarrow}$; the observed label there is $\tilde y$.) Proof as
printed:
$\mathbb E_{\tilde y\mid x}\boldsymbol\ell^{\leftarrow}(\hat p(y\mid x))=\mathbb E_{y\mid x}T\boldsymbol\ell^{\leftarrow}(\hat p(y\mid x))=\mathbb E_{y\mid x}TT^{-1}\boldsymbol\ell(\hat p(y\mid x))=\mathbb E_{y\mid x}\boldsymbol\ell(\hat p(y\mid x))$.
The loss is "differentiable — although not always non-negative". If $T$ is badly conditioned: "mix $T$ with the
identity matrix before inversion", "a more conservative noise-free prior" (page 4).

### 3.4 Forward correction (Theorem 2, page 4)

A network trained without correction predicts noisy labels, so for cross-entropy

$$\ell(e^i,\hat p(y\mid x))=-\log\hat p(\tilde y=e^i\mid x)\tag{3}$$
$$=-\log\sum_{j=1}^{c}p(\tilde y=e^i\mid y=e^j)\,\hat p(y=e^j\mid x)\tag{4}$$
$$=-\log\sum_{j=1}^{c}T_{ji}\,\hat p(y=e^j\mid x),\tag{5}$$

in matrix form $\boldsymbol\ell(\hat p(y\mid x))=-\log T^{\top}\hat p(y\mid x)$. **Verified on the page image: it is
$T^{\top}$ applied to the prediction, inside the loss.** (The introduction and Table 1 say loosely "multiplies the
network predictions by $T$".)

Proper composite losses. A link function $\psi:\Delta^{c-1}\to\mathbb R^c$ is invertible; a composite loss
$\ell_\psi:\mathcal Y\times\mathbb R^c\to\mathbb R$ satisfies

$$\ell_\psi(y,h(x))=\ell(y,\psi^{-1}(h(x))).\tag{6}$$

For cross-entropy the softmax is the inverse link. If the composite loss is also proper,

$$\operatorname*{argmin}_{h}\mathbb E_{x,y}\,\ell_\psi(y,h(x))=\psi(p(y\mid x)).\tag{7}$$

Cross-entropy and square loss are proper composite.

**Theorem 2.** Suppose $T$ is non-singular. Given a proper composite loss $\ell_\psi$, define

$$\boldsymbol\ell^{\rightarrow}_{\psi}(h(x))=\boldsymbol\ell\big(T^{\top}\psi^{-1}(h(x))\big).$$

Then

$$\operatorname*{argmin}_{h}\mathbb E_{x,\tilde y}\,\ell^{\rightarrow}_{\psi}(y,h(x))=\operatorname*{argmin}_{h}\mathbb E_{x,y}\,\ell_{\psi}(y,h(x)).$$

Proof:

$$\ell^{\rightarrow}_{\psi}(y,h(x))=\ell(y,T^{\top}\psi^{-1}(h(x)))=\ell_{\phi}(y,h(x)),\tag{8}$$

with $\phi^{-1}=\psi^{-1}\circ T^{\top}$, that is $\phi=(T^{-1})^{\top}\circ\psi$, invertible, so $\ell_\phi$ is proper
composite with link $\phi$, and by (7)

$$\operatorname*{argmin}_{h}\mathbb E_{x,\tilde y}\,\ell_{\phi}(y,h(x))=\phi(p(\tilde y\mid x))\tag{9}$$
$$=\psi\big((T^{-1})^{\top}p(\tilde y\mid x)\big)=\psi(p(y\mid x)).\tag{10}$$

Weaker than Theorem 1: "Robustness applies to the minimizer only." There is no explicit inversion; "the 'de-noising'
link function $\phi$ does it behind the scene".

### 3.5 Estimating $T$ without clean labels (Theorem 3, pages 4–6)

**Theorem 3.** Assume (1) perfect examples exist for each class $j\in[c]$:
$(\exists\bar x^j\in\mathcal X):\;p(\bar x^j)>0\wedge p(y=e^j\mid\bar x^j)=1$; (2) given sufficiently many corrupted
samples, $h$ is rich enough to model $p(\tilde y\mid x)$ accurately. Then

$$\forall i,j\in[c],\quad T_{ij}=p(\tilde y=e^j\mid\bar x^i).$$

Proof: for any $j$ and $x$,

$$p(\tilde y=e^j\mid x)=\sum_{k=1}^{c}p(\tilde y=e^j\mid y=e^k)\,p(y=e^k\mid x)=\sum_{k=1}^{c}T_{kj}\,p(y=e^k\mid x),\tag{11}$$

and at $x=\bar x^i$, $p(y=e^k\mid\bar x^i)=0$ for $k\neq i$. The estimator, on any feature set $X'$ (no labels needed):

$$\bar x^i=\operatorname*{argmax}_{x\in X'}\hat p(\tilde y=e^i\mid x)\tag{12}$$
$$\hat T_{ij}=\hat p(\tilde y=e^j\mid\bar x^i).\tag{13}$$

Cost $O(c^2\cdot|X'|)$. Percentile variant (page 6): "we take a $\alpha$-percentile in place of the argmax of
Equation 12, and we found $\alpha=97\%$ to work well for most experiments"; on CIFAR-100 the estimator "performs very
poorly" and "it is better off computing the argmax instead". $X'$ is training plus validation; "after estimation, we
row-normalize the matrix".

### 3.6 Algorithm 1, "Robust two-stage training" (page 5), in full

- **Input:** the noisy training set $\mathcal S$, any loss $\ell$
- If $T$ is unknown:
  - Train a network $h(x)$ on $\mathcal S$ with loss $\ell$
  - Obtain an unlabeled sample $X'$
  - Estimate $\hat T$ by Equations (12)–(13) on $X'$
- Train the network $h(x)$ on $\mathcal S$ with loss $\ell^{\leftarrow}$ or $\ell^{\rightarrow}$
- **Output:** $h(\cdot)$

Text: if $T$ is known, "for example by cleaning manually a subset of training data", skip estimation; retraining may
use the first network to initialise the second.

### 3.7 Hessian result for ReLU networks (Theorem 4, page 5)

**Theorem 4.** Assume all activation functions are ReLUs (footnote: $\ell$ must be a linear-odd loss; cross-entropy and
square loss qualify; it extends to any piece-wise linear network, including max-pooling). Then the Hessian of $\ell$
does not change under noise, and the Hessians of $\ell^{\leftarrow}$ and $\ell$ are the same for any $T$.

Proof sketch as printed: for $y=e^i$ the loss is
$-W^{(n)}_{i\cdot}x^{(n-1)}+b^{(n)}_i+\log\sum_{k=1}^{c}e^{W^{(n)}_{k\cdot}x^{(n-1)}+b^{(n)}_k}$ (the sign of $b^{(n)}_i$ is
printed as $+$), and

$$\ell^{\leftarrow}(e^j,\hat p(y\mid x))=\big(T^{-1}\boldsymbol\ell(\hat p(y\mid x))\big)_j\tag{14}$$
$$=-\big(T^{-1}W^{(n)}\big)_{j\cdot}x^{(n-1)}-\big(T^{-1}b^{(n)}\big)_j\tag{15}$$
$$+\log\sum_{k=1}^{c}e^{W^{(n)}_{k\cdot}x^{(n-1)}+b^{(n)}_k}.\tag{16}$$

With $A(x)=\log\sum_k e^{W^{(n)}_{k\cdot}x^{(n-1)}+b^{(n)}_k}$: $T^{-1}A(x)\mathbf 1=A(x)\mathbf 1$ because
$T\mathbf 1=\mathbf 1$. The first part is piece-wise linear in the parameters, so its Hessian vanishes. The theorem
"does not provide any assurance on minima"; it concerns curvature and convergence rate only.

Table 1 (page 5):

| loss | correction | $\mathbb E_{x,\tilde y}$ | Hessian of $\mathbb E_{x,\tilde y}$ |
|---|---|---|---|
| $\ell$ | – | no guarantee | unchanged |
| $\ell^{\leftarrow}$ | $T^{-1}\cdot$ | unbiased estimator of $\ell$ | unchanged |
| $\ell^{\rightarrow}$ | $T\cdot$ | same minimizer of $\ell$ | no guarantee |

## 4. Experimental setup (pages 6–7)

Common: $\ell$ is cross-entropy; 10% of training data held out for validation, with the (corrected) loss validated on
noisy data; standard test sets; ReLU everywhere; He initialisation before ReLUs, otherwise uniform in
$[-0.05,0.05]$; mini-batch 128. Noise is injected by a parametric $T$ with $N\in[0,1]$: the true class keeps $1-N$,
the wrong class gets $N$. Symmetric noise: $N$ is spread uniformly over all other classes. The network used to
estimate $T$ is trained with plain $\ell$ on the noisy set (Algorithm 1); separate settings for it are not stated in
the paper.

| Dataset | Model | Asymmetric noise | Optimiser |
|---|---|---|---|
| MNIST | 2 dense layers of 128, dropout 0.5; pixels in $[0,1]$ | 2→7, 3→8, 5↔6, 7→1 | AdaGrad, 40 epochs, lr 0.01, $\delta=10^{-6}$; 5 runs |
| IMDB (a) | 50-d word embedding then ReLUs, dropout 0.8 on embeddings; top 5,000 words, 400 words per review | 0→1 fixed at 5%, 1→0 at $N$ | AdaGrad, 50 epochs, same setup; 5 runs |
| IMDB (b) | 256-d embedding, LSTM 512, 512-d hidden layer with dropout 0.5 | as above | as above |
| CIFAR-10 | Pre-activation ResNet, depth 14 and 32; mean subtraction, flips, 32×32 crops after 4-pixel padding | TRUCK→AUTOMOBILE, BIRD→AIRPLANE, DEER→HORSE, CAT↔DOG | SGD, momentum 0.9, lr 0.01, ÷10 after epochs 40 and 80 (120 total), weight decay $10^{-4}$; one run |
| CIFAR-100 | ResNet depth 44 | 20 super-classes of 5; each class flips to the next, circularly | same, ÷10 after 80 and 120 (150 total); one run |
| Clothing1M | ResNet-50 (bottleneck), ImageNet pre-trained; 14 classes; 1M noisy, 50k/14k/10k clean train/val/test | real noise | SGD, lr $10^{-3}$ then $10^{-4}$, 5 epochs each, momentum 0.9, batch 32; weight decay $5\cdot10^{-2}$ plus augmentation on 50k, $10^{-3}$ only on 1M; the fine-tune on 50k (#8) uses "the same learning parameters as in #7" |

Example $T$ for MNIST at $N=0.7$ (Eq. 17, left): identity except rows for digits 2, 3, 5, 6, 7, which hold 0.3 on the
diagonal and 0.7 at columns 7, 8, 6, 5, 1. The estimated $\hat T$ (right) has diagonal/off-diagonal pairs .33/.67,
.35/.65, .29/.71, .26/.73, .25/.75, other entries $\epsilon<10^{-6}$. For Clothing1M, "instead of estimating the matrix
$T$ by (12)-(13), we exploit the curated labels of 50k and their noisy versions in 1M"; the counting formula is not
stated in the paper. Baselines: unhinged and sigmoid (each needing a batch-norm layer before the loss), Savage, soft
and hard bootstrapping.

## 5. Results

Table 2 (page 8), accuracy in %. Columns: no noise / symmetric / asymmetric low / asymmetric high. ⋆ marks a best
result obtained with the true $T$; † marks estimated-$\hat T$ runs that beat $T=I$ without injected noise. Bold face
is omitted here.

| Method | MNIST, dense ($N$ = 0.2, 0.2, 0.6) | IMDB, embedding ($N$ = 0.1, 0.1, 0.4) | IMDB, LSTM ($N$ = 0.1, 0.1, 0.4) |
|---|---|---|---|
| cross-entropy | 97.9±0.0 / 96.9±0.1 / 97.5±0.0 / 53.0±0.6 | 86.7±0.0 / 84.6±0.1 / 85.0±0.2 / 58.1±0.5 | 87.8±0.4 / 85.2±0.5 / 86.8±0.4 / 71.4±1.3 |
| unhinged (BN) | 97.6±0.0 / 96.9±0.1 / 97.0±0.1 / 71.2±1.0 | 83.3±0.0 / 76.9±0.5 / 80.6±0.3 / 72.9±0.4 | 84.3±4.4 / 69.7±15.9 / 85.2±1.2 / 59.4±12.9 |
| sigmoid (BN) | 97.2±0.1 / 93.1±0.2 / 96.7±0.1 / 71.4±1.3 | 84.3±0.0 / 80.2±0.3 / 81.7±0.5 / 72.8±0.6 | 87.7±0.5 / 77.6±13.6 / 86.3±3.1 / 70.0±14.6 |
| Savage | 97.3±0.0 / 96.9±0.0 / 97.0±0.1 / 51.3±0.4 | 86.5±0.0 / 84.3±0.4 / 85.2±0.3 / 58.3±1.0 | 87.4±0.3 / 85.1±0.6 / 87.2±0.3 / 70.4±3.8 |
| bootstrap soft | 97.9±0.0 / 96.9±0.0 / 97.5±0.0 / 53.0±0.4 | 86.7±0.0 / 84.5±0.1 / 85.1±0.1 / 57.8±0.7 | 87.1±0.6 / 83.5±2.5 / 86.1±1.2 / 69.0±5.3 |
| bootstrap hard | 97.9±0.0 / 96.8±0.0 / 97.4±0.0 / 55.0±1.3 | 86.7±0.0 / 84.6±0.3 / 85.1±0.3 / 59.0±0.6 | 86.5±0.5 / 84.3±1.0 / 86.7±0.4 / 71.8±3.3 |
| backward | 97.9±0.0 / 96.3±0.1 / 96.6±1.1 / 93.0±0.9⋆ | 86.7±0.0 / 85.3±0.3⋆ / 85.7±0.1 / 82.1±0.1⋆ | 87.6±0.2 / 84.3±0.9 / 86.7±0.5 / 83.6±1.4 |
| backward $\hat T$ | 97.9±0.0 / 96.9±0.0 / 96.7±0.1 / 67.4±1.5 | 87.0±0.0† / 85.1±0.4 / 85.8±0.2 / 77.0±1.4 | 87.2±0.7 / 82.8±2.7 / 87.3±0.1 / 82.3±1.7 |
| forward | 97.9±0.0 / 97.3±0.0⋆ / 97.7±0.0 / 97.3±0.0⋆ | 86.7±0.0 / 85.3±0.2⋆ / 85.9±0.1 / 80.9±1.3⋆ | 87.5±0.2 / 85.0±0.2 / 87.0±0.4 / 84.7±0.6⋆ |
| forward $\hat T$ | 97.9±0.0 / 96.9±0.0 / 97.7±0.0 / 64.9±4.4 | 87.0±0.0† / 85.2±0.3 / 85.9±0.2 / 73.0±1.2 | 87.8±1.5† / 84.1±1.0 / 87.5±0.2 / 84.2±0.9 |

| Method | CIFAR-10, ResNet-14 ($N$ = 0.2, 0.2, 0.6) | CIFAR-10, ResNet-32 | CIFAR-100, ResNet-44 |
|---|---|---|---|
| cross-entropy | 87.8 / 83.7 / 85.0 / 57.6 | 90.1 / 86.6 / 89.0 / 53.6 | 68.5 / 57.9 / 63.5 / 17.1 |
| unhinged (BN) | 86.9 / 84.1 / 83.8 / 52.1 | 90.2 / 86.5 / 87.1 / 60.0 | 50.9 / 47.5 / 48.0 / 14.5 |
| sigmoid (BN) | 76.0 / 66.6 / 71.8 / 57.0 | 81.6 / 69.6 / 79.1 / 61.8 | 58.2 / 47.6 / 55.6 / 16.4 |
| Savage | 80.1 / 77.4 / 76.0 / 50.5 | 88.3 / 86.2 / 86.3 / 53.5 | 1.4 / 2.0 / 1.8 / 1.6 |
| bootstrap soft | 87.7 / 84.3 / 84.6 / 57.8 | 90.9 / 86.9 / 88.6 / 53.1 | 67.9 / 57.8 / 63.8 / 16.3 |
| bootstrap hard | 87.3 / 83.6 / 84.7 / 58.3 | 90.4 / 86.4 / 88.6 / 54.7 | 68.5 / 57.3 / 63.9 / 17.0 |
| backward | 87.6 / 81.5 / 83.8 / 75.2⋆ | 90.1 / 83.0 / 84.4 / 74.3 | 68.5 / 55.1 / 53.8 / 36.8⋆ |
| backward $\hat T$ | 87.7 / 80.4 / 83.8 / 66.7 | 90.8 / 86.9 / 86.4 / 66.7 | 68.6 / 51.7 / 63.8 / 18.5 |
| forward | 87.8 / 85.6⋆ / 86.3 / 84.5⋆ | 91.2 / 87.7 / 89.9 / 87.6⋆ | 68.8 / 64.0⋆ / 68.1⋆ / 68.4⋆ |
| forward $\hat T$ | 87.4 / 83.4 / 87.0 / 74.8 | 90.5 / 87.9 / 90.1 / 77.6 | 68.1 / 58.6 / 64.2 / 15.9 |

Table 3 (page 8), Clothing1M accuracy:

| # | Model | Loss | Init | Training data | Accuracy |
|---|---|---|---|---|---|
| 1 | AlexNet | cross-entropy | ImageNet | 50k | 72.63 |
| 2 | AlexNet [39] | cross-entropy | #1 | 1M, 50k | 76.22 |
| 3 | AlexNet [42] | cross-entropy | #1 | 1M, 50k | 78.24 |
| 4 | ResNet-50 | cross-entropy | ImageNet | 1M | 68.94 |
| 5 | ResNet-50 | backward | ImageNet | 1M | 69.13 |
| 6 | ResNet-50 | forward | ImageNet | 1M | 69.84 |
| 7 | ResNet-50 | cross-entropy | ImageNet | 50k | 75.19 |
| 8 | ResNet-50 | cross-entropy | #6 | 50k | **80.38** |

Authors' reading: estimating $T$ costs "between 0 (IMBD with LSTM model) and 27 points (MNIST)" on average excluding
CIFAR-100, with "a median drop of only 10 points" (pages 7–8).

## 6. Ablations, failure modes, limitations

- **Backward is unstable.** It needs $T^{-1}$; the loss can be negative; coefficients "can be far by orders of
  magnitude" (page 8). "Except with the shallowest ResNet, backward correction does not seem to work well in the low
  noise regime": ResNet-32 at asymmetric 0.2 gives 84.4 against 89.0 for plain cross-entropy; CIFAR-100 gives 53.8
  against 63.5. In Figure 1f the backward curve with the true $T$ dips to roughly 0.37 near $N$ = 0.4–0.6 (read off
  the plot) while forward stays near 0.68.
- **Sensitivity to $\hat T$.** At high asymmetric noise forward falls from 97.3 to 64.9±4.4 (MNIST), 84.5 to 74.8 and
  87.6 to 77.6 (CIFAR-10), and 68.4 to 15.9 (CIFAR-100, below uncorrected 17.1) when $T$ is estimated. The example
  $\hat T$ printed in Eq. 17 ($N=0.7$) is within 0.01–0.05 of the truth, yet on MNIST at $N=0.6$ the estimated-matrix
  runs lose 25.6 (backward) and 32.4 (forward) points against the true $T$; the paper does not reconcile these beyond
  naming estimation as the bottleneck (page 5 says only that "small estimation errors have a mild, not catastrophic
  effect").
- **Many classes.** On CIFAR-100 the estimator "performs very poorly", attributed to "high noise and limited number
  of images per class (500)", and it is sensitive to $\alpha$ (numbers not stated). Future work: "priors of the noise
  structure, for example assuming low rank $T$".
- **Clothing1M.** With a $T$ counted from paired labels, correction alone adds little: 68.94 → 69.13 (backward) →
  69.84 (forward). The large step, 80.38 against 75.19 for clean-only, comes from fine-tuning the forward-trained
  network on clean data. Single run; only #6 was fine-tuned, so no cross-entropy-pretrained control exists.
- **When correction does nothing.** Without noise, forward with $\hat T$ is within −0.4 to +0.4 points of
  cross-entropy. The theorems hold "in the limit of infinite data", and need a non-singular $T$, a proper composite
  loss and an expressive network. Forward has no Hessian guarantee.
- **Assumption.** Class-conditional noise only. Deep-ResNet experiments have no standard deviations.

## 7. Recipe for our case

No GPU run, pod or submission is implied here; any training run needs the user's permission under the current hold.

**(a) Count $\hat T$ from paired rows.** Using paired clean/noisy labels instead of anchor points is *supported by the
paper* (Clothing1M; "cleaning manually a subset", section 4.3). The formula, soft counts, smoothing and cross-fitting
are *our extrapolation*.

$$\hat T_{ij}=\frac{\sum_{n\in\mathcal P}\mathbb 1[y_n=i]\,c_n(j)+\lambda\,\pi_{ij}}{\sum_{n\in\mathcal P}\mathbb 1[y_n=i]+\lambda}$$

- $y_n$: teacher label; $c_n\in\Delta^{75}$: Claude's answer on row $n$, encoded exactly as the training target
  (0.7/0.2/0.1 over its ranked codes); $\pi_{i\cdot}\in\Delta^{75}$ with full support, so every row sums to 1 and no
  column is all zero.
- Cross-fitting: for any dev number, $\mathcal P$ is the 1,199 fit rows only; all 1,500 only for a final test model.
  Check first that Claude's fit-row answers were blind (no fit labels in its prompt or examples), otherwise the
  diagonal is inflated.
- Prior $\pi$: either identity-heavy, $\pi_{ij}=(1-\varepsilon)\mathbb 1[i=j]+\varepsilon/76$ (the paper's "mix $T$ with
  the identity", proposed there for the backward inversion), or a tied family-block prior for $i\neq\text{NONE}$:
  $\pi_{ij}=a\,\mathbb 1[j=i]+b\,\mathbb 1[j=\text{NONE}]+s\,\mathbb 1[j=\text{SLOPPY}]+f\,\frac{\mathbb 1[j\in F(i)\setminus\{i\}]}{|F(i)|-1}+\frac{u}{76}$,
  with $a+b+s+f+u=1$ fitted on all non-NONE pairs pooled (five numbers, four of them free, from about 800–995 rows;
  $F(i)$ is the error family of code $i$; if $|F(i)|=1$ the family term is undefined, so fold $f$ into $u$ for that
  row). This is a structured prior of the kind the paper proposes as future work.
- Choose $\lambda\in\{1,3,10,30,100\}$ by leave-one-out log-likelihood
  $\sum_{n}\sum_j c_n(j)\log\hat T^{(-n)}_{y_n j}$.

**(b) Forward loss on a Claude row.** The hard-label form is *supported by the paper* (Eq. 5, Theorem 2); the
soft-target form is *our extrapolation*.

$$\mathcal L=-\sum_{j}\tilde q_j\log\Big(\sum_{i}\hat p_i\,\hat T_{ij}\Big)=-\tilde q^{\top}\log\big(\hat T^{\top}\hat p\big)$$

- Consistency: if $c_n$ and $\tilde q$ share one encoding, then $\mathbb E[\tilde q\mid x]=T^{\top}p(y\mid x)$ under the
  class-conditional model, and the minimiser is $\hat p=p(y\mid x)$ for non-singular $T$.
- Gradient with respect to logits $z$ (our derivation): $\partial\mathcal L/\partial z_k=\hat p_k-r_k$, with
  $r_k=\sum_j\tilde q_j\,\dfrac{\hat p_k\hat T_{kj}}{\sum_i\hat p_i\hat T_{ij}}$. So $r$ is the model's own posterior over
  the true class given Claude's answer. A column that is the same in every row (the sloppy-notation code, if Claude
  adds it regardless of the true class) gives $r_k=\hat p_k$ for that share of the target: zero gradient, so the mass
  is ignored.
- Teacher rows use $T=I$. Compute $\log(\hat T^{\top}\hat p)_j$ as
  $\operatorname{logsumexp}_i(\log\hat p_i+\log\hat T_{ij})$. Predict with $\hat p$, never $\hat T^{\top}\hat p$. Treat
  the 0.8 row weight as a setting to re-tune (0.8 or 1.0). Needs a 76-way softmax head; it does not apply as written
  to the pointwise yes/no ranker.
- Sequence *supported by the paper* (Table 3, #6 → #8): forward-corrected training on the Claude rows, then
  fine-tuning on teacher rows with plain cross-entropy.

**(c) Label shift against noise** (*our extrapolation* throughout).

- $T$ conditions on the true class, so it does not depend on the class prior. Rows counted on teacher data (33.7%
  NONE) remain valid for a pool with a different NONE rate, provided the class-conditional model holds. The reverse
  matrix $p(y\mid\tilde y)$ does depend on the prior and must not be used for relabelling the pool.
- Estimate the pool prior from Claude's pool marginal $\tilde\mu=\frac{1}{26{,}202}\sum_n\tilde q_n$:
  $\hat\pi^{\text{pool}}=\operatorname*{argmin}_{\pi\in\Delta^{75}}\lVert\hat T^{\top}\pi-\tilde\mu\rVert_2^2$. Two-class
  check with first choices only: with $\alpha=P(\text{Claude NONE first}\mid y=\text{NONE})$ and
  $\beta=P(\text{Claude NONE first}\mid y\neq\text{NONE})$,
  $\hat\pi^{\text{pool}}_{\text{NONE}}=(0.709-\beta)/(\alpha-\beta)$. A value outside $[0,1]$ means the
  class-conditional model is violated.
- The corrected model learns the pool's $p(y\mid x)$. If $\hat\pi^{\text{pool}}$ differs from the teacher prior,
  rescale at inference, $\hat p^{\text{test}}_k\propto\hat p_k\,\pi^{\text{teacher}}_k/\hat\pi^{\text{pool}}_k$, tuned on
  cross-fitted dev. Do not stack the earlier Claude-skew prior correction on top; $\hat T$ already absorbs it.
- Instance dependence is not handled. Diagnostic: compare leave-one-out log-likelihood of one global $\hat T$ with a
  per-family $\hat T$; a large gap means the single-matrix model is too coarse.
- Judge on the cross-fitted blend, not a single model, since the prior correction's gain did not survive blending.

**(d) Unit tests** (*our extrapolation*, except where noted).

1. $\hat T\mathbf 1=\mathbf 1$ and $\hat T_{ij}\ge 0$ to $10^{-6}$ (row-stochastic: *supported by the paper*).
2. $T=I$ gives exactly $-\sum_j\tilde q_j\log\hat p_j$ and gradient $\hat p-\tilde q$.
3. Finite-difference gradient check against $\hat p_k-r_k$.
4. Direction: with synthetic $T_{11}=0.3$, $T_{12}=0.7$, counting recovers $T$ and not $T^{\top}$; a small model
   recovers clean accuracy with $\hat T^{\top}\hat p$ and fails with $\hat T\hat p$ (mirrors MNIST at asymmetric
   $N=0.6$ with the true $T$: 53.0 → 97.3).
5. A constant column yields zero gradient.
6. Report the condition number of $\hat T$ (non-singular $T$ is required: *supported by the paper*).
7. Leakage: the $\hat T$ used for dev contains no dev rows.

## 8. Pages inspected as images

All nine: p01–p09. The supplied PNGs have a transparent background and display black, so each was flattened onto
white first. Table 2 and Equation 17 were also read at 3× magnification and checked against the extracted text.

Verified against page images: p01–p08 (Table 2, Equation 17 and Figure 1f re-read at 3× magnification; p09 holds references only and was checked in the extracted text)
