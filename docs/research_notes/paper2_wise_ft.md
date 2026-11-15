# Paper 2: WiSE-FT, "Robust fine-tuning of zero-shot models"

Extracted 14 November 2026 from the 51-page PDF. Numbers come from the paper's tables unless marked "read off the
figure" (approximate).

## 1. Citation and summary

Wortsman, M.\*, Ilharco, G.\*, Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Gontijo-Lopes, R., Hajishirzi, H.,
Farhadi, A., Namkoong, H., Schmidt, L. "Robust fine-tuning of zero-shot models." arXiv:2109.01903v3, 21 June 2022.
Code: https://github.com/mlfoundations/wise-ft. Venue: not stated in the PDF.

Fine-tuning a zero-shot model (CLIP) raises accuracy on the fine-tuning distribution but lowers accuracy under
distribution shift, often below the zero-shot model. WiSE-FT fine-tunes normally and then linearly interpolates the
weights of the zero-shot and fine-tuned models with one coefficient $\alpha$. On ImageNet and five shifted test sets
this improves shift accuracy by 4 to 6 pp over prior work while raising ImageNet accuracy by 1.6 pp, at no extra
training or inference cost.

## 2. Which bottleneck it addresses, and how directly

**Bottleneck.** Research question 8 in the experiment review (sections 5.3 and 6.1): Qwen3.8-27B zero-shot is strong
on hard rows (0.608) and weak on seen-pair rows (0.709); our fine-tuned system is the reverse (0.500, 0.867); training
Qwen3-32B lowered hard rows (0.392 to 0.357). The paper's mapping is: reference distribution = seen-pair rows, shifted
distribution = hard rows, and its headline pattern is the same shape as ours (CLIP ViT-L/14@336px, end-to-end, Table 4:
fine-tuning moves ImageNet 76.6 to 86.2 and the shift average 73.3 to 68.6; Table 1 prints the zero-shot shift average
as 73.4).

**Directness: moderate.** The algebra transfers exactly (section 7). The evidence does not: every experiment is image
classification with a single forward pass and a $k$-way logit vector.

(a) *End-to-end against linear classifier.* Shown: both work. End-to-end plus WiSE-FT is best (ViT-L/14, $\alpha=0.5$:
86.8 ImageNet, 76.9 shifts) against linear classifier plus WiSE-FT (83.7, 75.9). Plain end-to-end fine-tuning is less
robust than a plain linear classifier (68.6 against 72.6 on shifts), and interpolation recovers more there (+8.3 pp
against +3.3 pp at $\alpha=0.5$). For a linear classifier the weight ensemble equals the output ensemble exactly; for
end-to-end the benefit is an empirical observation with no guarantee. Not shown: LoRA or any adapter, partial
fine-tuning, generative or sequence models, multi-step reasoning.

(b) *Small fine-tuning sets.* Shown: sweeps at $k\in\{1,5,10,25,50\}$ examples per class on seven datasets, but
**linear classifier only**. The only end-to-end few-shot result is 16 examples per class of ImageNet (16,000 images)
for one model (ViT-B/16): Table 7 gives numbers at $\alpha=0.5$ only, and the CoOp-comparison figure on p. 31 draws
the full $\alpha$ curve on four shifts without numbers. Finding: the less data, the smaller the best $\alpha$. Not shown: end-to-end $\alpha$
curves at about 1,000 examples in total, long-tailed labels, or unseen label-context pairs.

(c) *When interpolation fails.* The paper states (pp. 6, 10–11, citing Frankle et al.) that interpolating the weights
of two networks trained from a random initialisation gives accuracy no better than a random classifier, and that
success depends on a shared starting point or shared part of the optimisation trajectory. On its own path it shows no
barrier. It does show that $\alpha=0.5$ is not free: ALIGN loses 1.9 pp on ImageNet, JFT ViT-H 2.5 pp, CLIP linear
classifier 1.5 pp; on YTBB-Robust $\alpha=0.5$ is 6.1 pp below zero-shot; on BASIC-L it is 0.7 pp below zero-shot on
the shifts. For end-to-end fine-tuning it never tests learning rates above $3\cdot10^{-5}$ (the linear-classifier
search does go higher, up to $10^{0}$; section 4). "Optimal $\alpha$" is chosen on the evaluation set
itself, per column: it is an oracle, not a validation protocol.

**Further limits on transfer (our reading).**
- A thinking model's answer depends on a sampled token chain; Proposition 1 (section 3) does not cover that.
- We have labels but no gold thinking chains, so the fine-tuning target is a design choice outside the paper.
- For CLIP each $\alpha$ is one cheap pass (21 values used); for us each is a full generative pass over dev.
- Table 13 shows that output-ensembling zero-shot CLIP with an *independently trained* model recovers most of the
  gain. Our product-of-probabilities fusion (0.805 against 0.772) is already that technique.

## 3. Exact method

**Zero-shot head (p. 5).** Image encoder $g$, text encoder $h$, class names $C=\{c_1,\dots,c_k\}$, caption
$s_i=$ "a photo of a $\{c_i\}$". Prediction: $\arg\max_j \langle g(x), h(s_j)\rangle$. Equivalently build
$\mathbf{W}_{\text{zero-shot}}\in\mathbb{R}^{d\times k}$ with columns $h(s_j)$ and compute

$$f(x)=g(x)^\top \mathbf{W}_{\text{zero-shot}} .$$

Prompt ensembling averages the embeddings of several captions per class (80 CLIP prompts, appendix D.1). CLIP's
learned temperature is factored into $\mathbf{W}_{\text{zero-shot}}$.

**Step 1, standard fine-tuning (pp. 5–6).** $f(x,\theta)=g(x,\mathbf{V}_{\text{enc}})^\top\mathbf{W}_{\text{classifier}}$,
$\mathbf{W}_{\text{classifier}}\in\mathbb{R}^{d\times k}$, $\theta=[\mathbf{V}_{\text{enc}},\mathbf{W}_{\text{classifier}}]$:

$$\arg\min_\theta\Big\{\sum_{(x_i,y_i)\in\mathcal{S}^{\text{tr}}_{\text{ref}}}\ell\big(f(x_i,\theta),y_i\big)+\lambda R(\theta)\Big\}$$

with $\ell$ the cross-entropy loss and $R$ a regulariser (for example weight decay). End-to-end changes all of
$\theta$; linear-classifier fine-tuning fixes $\mathbf{V}_{\text{enc}}$.

**Step 2, weight-space ensemble (p. 6, Eq. 1).** For $\alpha\in[0,1]$, zero-shot parameters $\theta_0$, fine-tuned
$\theta_1$:

$$\mathrm{wse}(x,\alpha)=f\big(x,(1-\alpha)\cdot\theta_0+\alpha\cdot\theta_1\big).$$

**Output-space ensemble it is compared with (pp. 6, 27, 29, 30).** On unnormalised scores:
$(1-\alpha)\cdot f(x,\theta_0)+\alpha\cdot f(x,\theta_1)$. Linear-classifier case, exact identity:
$(1-\alpha)\cdot g(x,\mathbf{V}_{\text{enc}})^\top\mathbf{W}_{\text{zero-shot}}+\alpha\cdot g(x,\mathbf{V}_{\text{enc}})^\top\mathbf{W}_{\text{classifier}}$.
Softmax variant: $(1-\alpha)\cdot\mathrm{softmax}(f(x,\theta_0))+\alpha\cdot\mathrm{softmax}(f(x,\theta_1))$.
Random interpolation: $(1-\xi)\cdot f(x,\theta_0)+\xi\cdot f(x,\theta_1)$, $\xi\sim\mathrm{Bernoulli}(\alpha)$.

**Effective robustness (p. 5).** With $\beta:[0,1]\to[0,1]$ the fitted trend such that
$\mathrm{Acc}_{\text{shift}}(f)\approx\beta(\mathrm{Acc}_{\text{ref}}(f))$ for models trained on
$\mathcal{S}^{\text{tr}}_{\text{ref}}$ (linear on logit-scaled axes):

$$\rho(f)=\mathrm{Acc}_{\text{shift}}(f)-\beta\big(\mathrm{Acc}_{\text{ref}}(f)\big).$$

**Error-landscape observations (pp. 10–11, empirical, not theorems).**
Observation 1 (Eq. 2), on ImageNet and the five shifts, for all $\alpha\in[0,1]$:

$$\mathrm{Acc}_{\mathcal{D},f}\big((1-\alpha)\cdot\theta_0+\alpha\cdot\theta_1\big)\;\ge\;(1-\alpha)\cdot\mathrm{Acc}_{\mathcal{D},f}(\theta_0)+\alpha\cdot\mathrm{Acc}_{\mathcal{D},f}(\theta_1).$$

Observation 2: there exists an $\alpha$ with
$\mathrm{Acc}_{\mathcal{D},f}((1-\alpha)\cdot\theta_0+\alpha\cdot\theta_1)\ge\max\{\mathrm{Acc}_{\mathcal{D},f}(\theta_0),\mathrm{Acc}_{\mathcal{D},f}(\theta_1)\}$.

**Linear-regime analysis (p. 51, appendix F).** Let $\Theta=\{(1-\alpha)\theta_0+\alpha\theta_1:\alpha\in[0,1]\}$.
*Proposition 1.* When $f(\theta)=f(\theta_0)+\nabla f(\theta_0)^\top(\theta-\theta_0)$ for all $\theta\in\Theta$, the
weight- and output-space ensemble of $\theta_0$ and $\theta_1$ are equivalent. Proof (Eqs. 12–17):

$$\begin{aligned}
f((1-\alpha)\theta_0+\alpha\theta_1)
&=f(\theta_0)+\nabla f(\theta_0)^\top\big((1-\alpha)\theta_0+\alpha\theta_1-\theta_0\big)\\
&=f(\theta_0)+\alpha\nabla f(\theta_0)^\top(\theta_1-\theta_0)\\
&=f(\theta_0)+\alpha\nabla f(\theta_0)^\top(\theta_1-\theta_0)+\alpha f(\theta_0)-\alpha f(\theta_0)\\
&=(1-\alpha)f(\theta_0)+\alpha\big(f(\theta_0)+\nabla f(\theta_0)^\top(\theta_1-\theta_0)\big)\\
&=(1-\alpha)f(\theta_0)+\alpha f(\theta_1).
\end{aligned}$$

The paper adds: "In practice we observe a difference" between the two.

**Alternatives compared (p. 30).** Distillation (Eq. 3):
$\sum_{(x_i,y_i)\in\mathcal{S}^{\text{tr}}_{\mathcal{D}}}(1-\alpha)\cdot\ell(f(x_i,\theta),y_i)+\alpha\cdot\ell(f(x_i,\theta),f(x_i,\theta_0))$.
Regularise to zero-shot: add $\lambda\lVert\mathbf{W}-\mathbf{W}_{\text{zero-shot}}\rVert_F^2$ (linear classifier), or
$\lambda\lVert\theta-\theta_0\rVert_2^2$ (end-to-end, Figure 3 caption).

**EMA (pp. 28–29).** $\mu_t\leftarrow\beta\cdot\mu_{t-1}+(1-\beta)\cdot\theta_t$. Debiased: $\mu_0\leftarrow0$, final
$\mu_T/(1-\beta^T)$. Biased to initialisation: $\mu_0\leftarrow\theta_0$, final $\mu_T$. Applying WiSE-FT with
coefficient $1-\beta^T$ to the debiased variant exactly recovers the second.

**Diversity (pp. 47–48).** Prediction diversity:
$\mathrm{PD}(f,g,\mathcal{S})=\frac1N\sum_{1\le i\le N}\mathbb{1}[d_f\vee d_g]$,
$d_f=(\hat y_f^{(i)}=y^{(i)}\wedge\hat y_g^{(i)}\ne y^{(i)})$,
$d_g=(\hat y_f^{(i)}\ne y^{(i)}\wedge\hat y_g^{(i)}=y^{(i)})$.

**Algorithm 1 (p. 21), structure verbatim.**

```python
def wse(model, zeroshot_checkpoint, finetuned_checkpoint, alpha):
    # load state dicts from checkpoints
    theta_0 = torch.load(zeroshot_checkpoint)["state_dict"]
    theta_1 = torch.load(finetuned_checkpoint)["state_dict"]
    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())
    # interpolate between all weights in the checkpoints
    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }
    # update the model (in-place) according to the new weights
    model.load_state_dict(theta)

def wise_ft(model, dataset, zeroshot_checkpoint, alpha, hparams):
    # load the zero-shot weights
    theta_0 = torch.load(zeroshot_checkpoint)["state_dict"]
    model.load_state_dict(theta_0)
    # standard fine-tuning
    finetuned_checkpoint = finetune(model, dataset, hparams)
    # perform weight-space ensembling (in-place)
    wse(model, zeroshot_checkpoint, finetuned_checkpoint, alpha)
```

## 4. Exact hyperparameters

End-to-end, CLIP (appendix D.2, pp. 45–46), "unless otherwise mentioned":

| Item | Value |
|---|---|
| Optimiser | AdamW, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ |
| Learning rate | $3\times10^{-5}$ |
| Schedule, warm-up | cosine annealing, 500 warm-up steps |
| Epochs | 10 |
| Batch size | largest that fits 8 GPUs (512 for ViT-B/16) |
| Weight decay | 0.1 |
| Gradient clipping | global norm 1 |
| Augmentation | as CLIP: random square crop from the resized image (336 px for ViT-L/14@336px, 224 otherwise). Stronger augmentation (random crop, flip, colour jitter) gave "similar trends" (p. 30) |
| Label smoothing (end-to-end) | not stated in the paper |
| Head initialisation | the zero-shot classifier, scaled by CLIP's pre-trained temperature; no temperature parameter during fine-tuning; encoder outputs stay normalised |
| Default model | CLIP ViT-L/14@336px |

Other models. ALIGN: AdamW, 40,000 steps, batch 512, maximum rate $5\times10^{-6}$, weight decay 0.1, 500 linear
warm-up steps then cosine; zero-shot weights temperature-calibrated on ImageNet train before WiSE-FT (pp. 37–38).
JFT ViT-H/14: as ALIGN but 20k iterations and sigmoid cross-entropy (p. 40). BASIC: both encoders, contrastive loss,
half of ImageNet train (p. 42). 16-shot comparison: 10 epochs, rate $10^{-5}$ (p. 30).

Linear classifier (appendix D.3, pp. 46–47): AdamW, cosine schedule, 64 random configurations per regulariser; batch
uniform in $\{64,128,256\}$; rate $10^{-\beta}$, $\beta\sim U[0,6]$; weight decay $10^{-\lambda}$, $\lambda\sim U[0,4]$;
L1 $10^{-\lambda}$, $\lambda\sim U[4,8]$; label smoothing $\lambda\sim U[0,0.25]$. The configuration with the best
in-distribution accuracy is used. Low-data: the maximum number of epochs $T$ is scaled approximately inversely with
the data so the iteration count is consistent ($T$ found with default AdamW, rate 0.001, weight decay 0.01, doubling
epochs until performance saturates); each random run draws its epochs uniformly from $\{1,\dots,T\}$.

**$\alpha$ grid:** $\{0,0.05,0.1,\dots,1\}$ (p. 29). **How $\alpha$ is chosen:** "optimal $\alpha$" is the single
value that maximises each table column on the evaluation data; there is no held-out selection. The recommendation is
fixed $\alpha=0.5$ "when no domain knowledge is available". **Cost of fixing $\alpha=0.5$ (Table 3, p. 22):** on the
two averaged columns it is 0.0 to 0.5 pp below the per-column optimum; on single datasets up to 2.4 pp (ViT-B/16
end-to-end, ImageNet-A) and 1.8 pp (ViT-B/16 linear classifier, ImageNet).

## 5. Results

**Table 1 (p. 7), CLIP ViT-L/14@336px, top-1 %.**

| Method | ImageNet | IN-V2 | IN-R | IN-Sketch | ObjectNet | IN-A | Avg shifts | Avg ref., shifts |
|---|---|---|---|---|---|---|---|---|
| Zero-shot (PyTorch) | 76.6 | 70.5 | 89.0 | 60.9 | 69.1 | 77.7 | 73.4 | 75.0 |
| Fine-tuned LC | 85.2 | 75.8 | 85.3 | 58.7 | 67.2 | 76.1 | 72.6 | 78.9 |
| Fine-tuned E2E | 86.2 | 76.8 | 79.8 | 57.9 | 63.3 | 65.4 | 68.6 | 77.4 |
| WiSE-FT LC, $\alpha=0.5$ | 83.7 | 76.3 | 89.6 | 63.0 | 70.7 | 79.7 | 75.9 | 79.8 |
| WiSE-FT LC, optimal $\alpha$ | 85.3 | 76.9 | 89.8 | 63.0 | 70.7 | 79.7 | 75.9 | 80.2 |
| WiSE-FT E2E, $\alpha=0.5$ | 86.8 | 79.5 | 89.4 | 64.7 | 71.1 | 79.9 | 76.9 | 81.8 |
| WiSE-FT E2E, optimal $\alpha$ | 87.1 | 79.5 | 90.3 | 65.0 | 72.1 | 81.0 | 77.4 | 81.9 |

**$\alpha$ sweep, same model, end-to-end (Table 4, p. 24): ImageNet / Avg shifts.** 0.00: 76.6 / 73.3. 0.10: 80.4 /
75.5. 0.20: 83.2 / 76.7. 0.30: 85.1 / 77.3. 0.35: 85.7 / 77.4 (shift peak). 0.40: 86.2 / 77.3. 0.50: 86.8 / 76.9.
0.60–0.70: 87.1 (ImageNet peak) / 76.0–74.9. 0.80: 86.9 / 73.3. 0.90: 86.7 / 71.2. 1.00: 86.2 / 68.6. The shift
optimum sits at a smaller $\alpha$ than the reference optimum; $\alpha=0.4$ matches fine-tuned ImageNet accuracy
with +8.7 pp on shifts. ViT-B/16 (Table 5, p. 25): shift peak 63.5 at 0.40, ImageNet peak 82.6 at
0.65–0.75.

**Other models, end-to-end (Table 8, p. 37): ImageNet / Avg shifts.**

| Model | Zero-shot | Fine-tuned | WiSE-FT 0.5 | WiSE-FT opt. |
|---|---|---|---|---|
| CLIP ViT-B/16 | 68.3 / 58.1 | 81.3 / 53.8 | 81.7 / 63.0 | 82.6 / 63.5 |
| ALIGN | 76.4 / 74.6 | 88.2 / 75.0 | 86.3 / 78.4 | 88.3 / 78.4 |
| JFT ViT-H | 72.9 / 65.3 | 85.4 / 69.8 | 82.9 / 72.1 | 85.4 / 72.5 |
| BASIC-M | 81.4 / 74.5 | 86.2 / 73.2 | 85.6 / 77.3 | 86.2 / 77.4 |
| BASIC-L | 85.6 / 84.1 | 87.5 / 76.3 | 87.9 / 83.4 | 87.9 / 85.0 |

BASIC-L's shift optimum is at $\alpha=0.10$–$0.20$ (Table 12, p. 44); JFT at $\alpha=0.8$ gives +2.2 pp shifts within
0.2 pp of fine-tuned ImageNet (p. 8).

**Six further shifts (Table 6, p. 27): zero-shot / fine-tuned / $\alpha=0.5$ / optimal.** ImageNet-Vid-Robust 95.9 /
86.5 / 95.5 / 96.5. YTBB-Robust 95.8 / 66.5 / 89.7 / 96.0. CIFAR-10.1 92.5 / 95.9 / 97.6 / 98.0. CIFAR-10.2 88.8 /
91.3 / 93.4 / 94.4. FMoW in-distribution 28.0 / 73.3 / 73.0 / 74.8; FMoW worst region 23.8 / 46.0 / 49.5 / 49.7.
iWildCam in-distribution F1 15.1 / 52.1 / 55.8 / 55.8; out-of-distribution F1 15.5 / 39.9 / 46.1 / 46.4. (The two
video rows are linear classifier; the rest end-to-end.)

**Reference-distribution gains over end-to-end fine-tuning (Table 2, p. 9), $\alpha=0.5$ / optimal, pp.** ImageNet
+0.6 / +0.9; CIFAR-10 +0.7 / +0.8; CIFAR-100 +1.1 / +1.2; Cars +1.7 / +2.0; DTD +2.8 / +3.3; SUN397 +2.5 / +2.6;
Food-101 +1.6 / +1.8.

**Low data.** 16 examples per class, ViT-B/16 (Table 7, p. 30), ImageNet / IN-V2 / IN-R / IN-A / IN-Sketch: CoOp 71.73
/ 64.56 / 75.28 / 49.93 / 47.89; WiSE-FT LC 0.5: 73.02 / 65.19 / 77.63 / 49.81 / 49.09; WiSE-FT E2E 0.5: 72.38 /
65.29 / 78.47 / 51.07 / 49.72. Linear classifier, mean of seven datasets (Figure 17, p. 33, read off the figure),
gain over the better of the two endpoints:

| $k$ per class | $\alpha=0.25$ | $\alpha=0.5$ | $\alpha=0.75$ | optimal |
|---|---|---|---|---|
| 1 | +1.5 | +1.0 | about −4 | +1.6 |
| 5 | +2.9 | +4.7 | +4.1 | +4.7 |
| 10 | +0.3 | +2.2 | +2.2 | +2.5 |
| 25 | below −2 | 0.0 | +0.8 | +0.8 |
| 50 | below −2 | −0.8 | +0.3 | +0.4 |

At $k=1$ the zero-shot model beats the fine-tuned classifier and small $\alpha$ wins; with more data larger $\alpha$
wins and the gain over the best endpoint shrinks. Figures 18–19 (pp. 34–35) show the same per dataset.

## 6. Ablations and failure modes

- **Hyperparameter sensitivity (Figure 3, p. 8, with the numbers in the text on p. 7; ViT-B/16).** Rates $\{10^{-7},10^{-6},3\cdot10^{-6},10^{-5},2\cdot10^{-5},3\cdot10^{-5}\}$,
  epochs $\{2,4,10\}$. Ten epochs at $3\cdot10^{-5}$ against $3\cdot10^{-6}$: 0.3 pp apart on ImageNet, up to 8 pp
  apart on shifts. $10^{-7}\to3\cdot10^{-5}$: +5 pp ImageNet, −8 pp shifts. Robustness cannot be read from
  reference accuracy.
- **Aggressiveness against interpolation.** Lower rate, fewer epochs and early stopping (iterations 250, 1000, 2500)
  all land below the WiSE-FT curve. Interpolating the aggressive ($3\cdot10^{-5}$) run "removes the trade-off";
  "only $\alpha$ needs to be considered". Only weight averaging (WiSE-FT, EMA) reaches the frontier.
- **Regularisation toward the start.** End-to-end $\lambda\lVert\theta-\theta_0\rVert_2^2$ points lie on or below the
  curve. Linear classifier (Figure 13, p. 29, read off the figure): WiSE-FT peaks near 75.7 avg shifts; distillation
  near 75.0; regularise-to-zero-shot near 74.7 and "slightly worse than distillation"; weight decay, L1, label
  smoothing at most about 73; warm-started classifiers at most about 74.7; random interpolation lies on the straight
  line between the endpoints.
- **Zero-shot initialisation.** The head is always initialised from the zero-shot classifier; a randomly initialised
  head is tested only for the linear classifier. Interpolation "is only possible for a pre-trained model capable of
  zero-shot inference".
- **Output against weight ensemble.** End-to-end, ViT-B/16, rate $3\cdot10^{-5}$ (Figure 10, p. 27, read off the
  figure): weight-space peaks near 63.3 avg shifts, output-space near 62.5; the curves coincide near $\alpha=1$.
  Output-space needs twice the compute. Softmax ensembling of a linear classifier "performs comparably". With an
  independent model (Table 13, p. 46): NoisyStudent EfficientNet-B6 alone 86.5 / 62.3; output ensemble with zero-shot
  CLIP at 0.5: 86.2 / 75.9, close to WiSE-FT's 86.8 / 76.9. EfficientNet-L2: 88.3 / 71.5 to 87.4 / 78.8.
- **Complementarity (Figure 5, p. 10; read off the figure).** Prediction diversity between the zero-shot model and the
  fine-tuned linear classifier is about 0.10–0.18, against 0.04–0.07 for two linear classifiers trained on
  half-splits of ImageNet; each model is more confident (larger top-two margin on unnormalised outputs) where it is
  better: zero-shot under shift, the linear classifier on ImageNet.
- **Where it helps least or hurts.** $\alpha=0.5$ costs reference accuracy for ALIGN (−1.9), JFT (−2.5), BASIC-M
  (−0.6), CLIP linear classifier (−1.5), FMoW (−0.3). Where zero-shot is far ahead on the shift (YTBB-Robust,
  BASIC-L; also ImageNet-Vid-Robust, 95.5 against 95.9), $\alpha=0.5$ ends below zero-shot and the optimum is small.
  No shifted dataset is reported where no $\alpha$ helps. On the reference set the optimum only ties the fine-tuned
  endpoint for JFT ViT-H (85.4) and BASIC-M (86.2) in Table 8, and for the linear classifier on full training sets
  the gain over the better endpoint falls to about zero on several datasets (Figures 18–19, read off the figure).
- **Inconsistency.** Table 13 gives optimal averages 77.6 / 82.3 against 77.4 / 81.9 in Tables 1 and 8; 77.6 is the
  mean of per-column optima, not one $\alpha$.

## 7. Recipe for our case

1. **Interpolation is adapter scaling. (Algebra exact; application to LoRA is our extrapolation.)** For each adapted
   matrix $\theta_0$ holds $W_0$ and $\theta_1$ holds $W_0+\frac{s}{r}BA$, so
   $$(1-\alpha)W_0+\alpha\Big(W_0+\frac{s}{r}BA\Big)=W_0+\alpha\frac{s}{r}BA .$$
   Unadapted parameters are equal at both ends and do not move. Implement by multiplying every $B$ by $\alpha$, or by
   replacing $s$ with $\alpha s$. Conditions: (i) plain LoRA or rsLoRA (linear in $s$); DoRA is not linear, so merge
   first and interpolate full weights; (ii) any fully trained module must be interpolated explicitly; (iii) the base
   $W_0$ at inference must be the base the adapter was trained on. A QLoRA adapter trained on a 4-bit base and served
   on the FP8 base is not on the paper's path; re-measure $\alpha=0$ and $\alpha=1$ on the serving base.
2. **Start exactly at the zero-shot model. (Supported by the paper for CLIP; our extrapolation for a VLM.)** LoRA with
   $B=0$ gives $\theta=\theta_0$ at step 0, the analogue of zero-shot head initialisation. It holds only if the
   training prompt, code list, output format and thinking setting are those of the validated zero-shot run.
3. **Fine-tune normally, then scale back. (Supported by the paper for CLIP; our extrapolation for LoRA.)** Do not protect hard rows with a tiny rate or
   early stopping; those lie below the interpolation curve. Paper's settings: AdamW, cosine schedule with warm-up,
   weight decay 0.1, clipping at norm 1. The LoRA rate, rank and epoch count have no counterpart in the paper.
4. **Supervision target. (Our extrapolation; the largest risk.)** The paper has no reasoning chain. Loss on the answer
   only, after the model's own thinking, keeps the format identical to zero-shot inference; answer-only training with
   thinking off trains a different function from the one evaluated. Our Stage B reason-first student failed (review
   5.4), so this needs its own dev gate.
5. **Choosing $\alpha$. (Default 0.5 supported by the paper; coarse grid and slice weighting are our extrapolation.)** Evaluate
   $\mathcal{A}=\{0,0.25,0.5,0.75,1\}$, refine by 0.05–0.1 around the best. For slices $g\in\{N,S,H\}$ (no error, seen
   pair, hard) with test-mix weights $\pi_g$ (dev: $101/301$, $94/301$, $106/301$):
   $$M(\alpha)=\sum_g\pi_g\,M_g(\theta_\alpha),\qquad \alpha^{*}=\arg\max_{\alpha\in\mathcal{A}}M^{\text{fused}}(\alpha),$$
   where $M^{\text{fused}}$ is dev MAP@3 after fusing $\theta_\alpha$ with the submitted system, since that is what
   ships. Expect the hard-row optimum at a smaller $\alpha$ than the seen-pair optimum (Table 4: 0.35 against
   0.60–0.70). With about 100 rows per slice, keep $\alpha=0.5$ unless another value wins on a paired bootstrap; the
   paper's "optimal $\alpha$" is an oracle and would overfit 301 rows.
6. **Monitors.**
   - Barrier check per slice (Observation 1; supported):
     $M_g(\theta_\alpha)\ge(1-\alpha)M_g(\theta_0)+\alpha M_g(\theta_1)$. A violation means the path is not in the
     paper's regime; fall back to the output ensemble.
   - Output-ensemble fallback (identity is ours): $p(c)\propto p_{\theta_0}(c)^{1-\alpha}p_{\theta_1}(c)^{\alpha}$
     equals $\mathrm{softmax}((1-\alpha)z_0+\alpha z_1)$, the paper's score ensemble, at twice the inference cost.
   - Hard rows against 0.608 and seen-pair rows against 0.709 on the same 270 rows; no-error rows against 0.924.
   - Prediction diversity $\mathrm{PD}$ between $\theta_\alpha$ and the submitted system; fusion gain should fall as
     $\alpha$ makes Qwen resemble our fine-tuned system (paper section 5.1; application ours).
   - Share of rows answered within the time limit, and thinking length, at each $\alpha$ (ours).
7. **Expectation. (Our reading.)** The paper supports the claim that some $\alpha<1$ beats the fine-tuned endpoint on
   shifted data for CLIP-like classifiers. It gives no basis for a numerical forecast for LoRA, a generative model, about 1,000
   rows or unseen label-context pairs. Each $\alpha$ is a paid dev pass and needs the user's permission.

## 8. Pages inspected as images

2, 5, 6, 7, 8, 9, 10, 11, 21, 22, 24, 27, 28, 29, 30, 32, 33, 34, 35, 37, 45, 46, 47, 48, 51. Pages 2 and 8 rendered
with invisible body text; their figures were read from the image and their text from `text.txt`, which was read in
full. Printed oddity, transcribed as is: the regularisation grid on p. 30 reads
$\lambda\in\{1\cdot10^{-8},5\cdot10^{-8},1\cdot10^{7},\dots,5\cdot10^{2}\}$.

Verified against page images: 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 40, 42, 44, 45, 46, 47, 48, 51
