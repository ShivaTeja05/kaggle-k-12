# Research report: Q1-Q5 (K-12 misconception MAP@3)

**Baseline:** dev MAP@3 0.718 (public 0.7085). Claude Opus zero-shot scores 0.819.

**How to read the numbers:** all dev numbers are on the 301-row fold-0 dev set. The paired standard error of a MAP@3 difference there is about 0.013, so one run cannot confirm an effect smaller than about +0.02.

**Budget left:** about $17 GPU and about $13 API. The GPU figure is shrinking because the v4 A100 pod is still running.

---

## Q1 - The maths paper (Guo 2025, EfficientNet + GNN formula recognizer)

**What it is:** J. Guo, *Systems and Soft Computing* 7 (2025) 200347. It is an offline recognizer for single handwritten formulas: it reads them and does not diagnose errors. It works in two stages:
- EfficientNet-B1 with BiFPN and an EAST-style head detects formula regions.
- A YOLOv5-style symbol detector feeds a Line-of-Sight graph, and a 3-layer edge-aware GAT classifies symbols and their spatial relations.

It also adds a small amount of confidence-ranked self-training (1-4% pseudo-labels). No code is released, and the data is available only on request.

**Does it help us?** No. We lose points in diagnosis, not in reading, and this recognizer is weaker than existing readers.
- On our dev set, gold transcripts beat machine transcripts by only 0.035 (95% CI 0.009-0.062).
- Even with gold text, rows with a real error score only 0.547.
- On CROHME the paper is below CoMER on all three sets: 59.14 / 56.14 / 54.63 vs 59.33 / 59.81 / 62.97. Uni-MuMER on Qwen2.5-VL-3B, trained on CROHME only, averages 73.29.
- Its tables contradict themselves. Its "F" values are higher than both P and R, so they are not F1, and the ablation rows do not match the headline numbers.

**What we can try:**

| Idea | Where in pipeline | Expected effect | Cost | Confidence |
|---|---|---|---|---|
| Train the text reasoner on corrupted gold transcripts. This moves the paper's random character erasing to the text side: drop or merge lines, swap adjacent lines, change about 5-15% of math tokens, matched to the measured machine noise (token F1 about 0.81). The fitting rows have no machine transcripts, so synthetic corruption is the only option. | `src/k12/text_diagnose.py`, where fitting transcripts load (`--fit-transcripts` defaults to gold) | Estimate 0 to +0.015, bounded by the 0.035 reading gap. One third of the gap is on NONE rows, so report NONE-row and error-row MAP separately. Adopt only on a clear gain (> +0.02, or consistent across seeds or epochs). | CPU coding is free. One 8B LoRA dev run costs a few dollars (estimate; 8B timing is not logged). | Low |
| Average the reasoner's probabilities over 2 transcriptions per photo (greedy plus one sampled reading). This runs offline, so it is allowed at test time. | `src/k12/transcriber.py` second pass, then the blend | Estimate 0 to +0.01; too small to detect on dev | Extra transcription passes on dev and test; cost unknown | Low |
| Add an auxiliary head that predicts the root-step line, trained from the annotation `step` field. Use it only as an auxiliary loss, never to crop the input. | `text_diagnose.py`, a second head beside the code head | Estimate 0 to +0.01 | One extra dev training run | Low |

**What won't work:**
- Re-implementing the EfficientNet + YOLO + LOS-GNN reader. It needs character boxes and relation labels we do not have, it handles single formulas rather than multi-line Indonesian workings, and it is weaker than CoMER and VLM readers.
- Expecting better reading to reach 0.82. It adds at most about 0.035 (CI up to about 0.06).
- Hard-cropping to formula or error regions. An oracle crop lowered MAP from 0.609 to 0.601.
- Citing the paper's numbers, because its tables are inconsistent.

---

## Q2 - Ultramarine 1 Flash (open-weight model)

**What it is:** The real name is "Ultramarine 1 Flash". Paper Instruments released it on 14 Oct 2026. It is not a new base model: it is Z.ai's GLM-5.3-Flash (a 320B-total / 18B-active MoE) with LoRA-based RL post-training on finance, legal and consulting agent tasks.
- **Weights:** about 331 GB in FP8, gated (approval is automatic), under a custom licence. No quantised version exists.
- **Vision:** the RL stage was text-only. The report says "We did not train with vision".
- **Benchmarks:** only vendor-reported knowledge-work scores; none for maths, OCR or handwriting. On the vendor's own table, Claude Opus 5.5 scores higher (61.9 vs 59.3).

**Does it help us?** No.
- It cannot run in offline Kaggle inference. Even the smallest community quant of the base model is about 93 GB, while a T4x2 has 32 GB VRAM plus about 29 GB RAM.
- It breaks our RunPod rules and budget, because it needs 5 or more 80 GB GPUs.
- It does not target the error-family mistakes where we lose points.
- As a teacher it has no advantage over Opus.

**What we can try:**

| Idea | Where in pipeline | Expected effect | Cost | Confidence |
|---|---|---|---|---|
| Do not adopt it, and keep the budget for the current pipeline. | Planning / budget | Avoids spending the remaining GPU budget on a model that cannot run at final inference | $0 | High |
| Borrow the idea of structured scoring: make the classifier aware of the taxonomy's code families. The code prefixes (PV, OPS, ...) and the `section` column define families, and `text_diagnose.py` does not use them today. First, on CPU, re-rank the existing dev probabilities with a family prior (sum code probabilities per family, then combine family and code scores). Train an auxiliary family head on the 8B only if the re-rank helps. | Post-hoc blend first; then `text_diagnose.py` / `pod_train.py` | Unknown, but it targets wrong-family errors directly. Adopt only on a measured fold-0 gain. | $0 for the CPU re-rank; about $2-5 for an 8B ablation (estimate) | Low |

**What won't work:**
- Running Ultramarine or its GLM-5.3-Flash base offline on Kaggle (331 GB in FP8; about 93 GB at 1-bit).
- Fine-tuning it within about $17. One allowed GPU (80 GB at most) holds about a quarter of the FP8 weights.
- Using it as the handwriting reader. Vision was not RL-trained, and the base ranks 43rd of 52 on Roboflow's OCR evaluations.
- Using it as a teacher instead of Opus. It needs a multi-GPU node and a new above-cap approval, and Opus scores higher.
- Using its hosted API, which would send competition images to an unapproved third party.
- Expecting a bigger general model to fix wrong-family errors. Our Qwen3-235B few-shot run scored 0.650, below the 0.718 pipeline.
- Running it on the Mac (36 GB RAM, about 12 GiB free disk).
- Licence: nobody has checked whether the custom licence fits the competition rules. The MIT-licensed base avoids that question but is just as impractical.

---

## Q3 - kimi-k3-in-c (trillion-parameter model on CPU) and Rust/C++ rewrites

**What it is:** An independent pure-C CPU inference engine for Moonshot's open-weights Kimi K3, under Apache-2.0 and about 10.5k lines long. By the repo's count the model has 2.78T parameters, takes 1.56 TB on disk and activates about 104B per token. The engine is built to make the model *fit* in memory, not to make it fast:
- Routed experts stay on disk, and only a 113 GB bf16 core stays in RAM.
- Experts are stored in MXFP4 and multiplied straight from packed nibbles.
- Weights are read with O_DIRECT, and layers stream through a prefetch ring.
- The model state can be saved and resumed.
- The kernels are deterministic, and every change is checked against PyTorch (max logit difference 7.87e-6).

It generates 5.6 to 26.5 s per token on a 124-vCPU EPYC server, and 80% of that time is disk wait.

**Does it help us?** Partly. The engine itself does not help, and neither would a Rust or C++ rewrite.
- **The engine:** our 3,498 test rows are about 4.1M prompt tokens, which would take weeks on a CPU engine like this.
- **A rewrite:** I re-measured the CPU-side Python. Tokenising takes 0.3 ms per prompt (the tokenizer is already Rust), and image preprocessing takes 6.4 ms per photo (already C, and prefetched). Against 1-2 s of GPU time per training row, a rewrite saves under 1% of wall-clock, about $0.
- **What transfers is the approach:**
  - Avoid recomputing the shared prompt prefix.
  - Profile before optimising.
  - Parity-gate every speed change.
  - Keep the inference base identical to the training base.
- **The real lever is the prompt.** The 921-token taxonomy list is about 72% of every training row (about 1,290 tokens) and about 78% of every inference prompt.

**What we can try:**

| Idea | Where in pipeline | Expected effect | Cost | Confidence |
|---|---|---|---|---|
| Remove the taxonomy list from the reasoner prompt, or keep code ids only. Test it first on the 8B with the scale_25 recipe against the 0.6916 gold-dev baseline. Caveat: the explanation targets start "ROOT: <CODE> (<name>)", so this also changes the explanation loss. Fix a tolerance in advance and use a paired bootstrap. | `text_diagnose.py`: `prompt()` and the definitions string in `run()` | About 3.5x fewer tokens per row (1,290 to 367). Estimated 2-3x faster training, so a 32B epoch drops from about 7 h ($12.6) to about 2.5-3.5 h ($4.5-6), which makes a second 32B run affordable. Effect on accuracy unknown. | About $1.5 (8B, 4k rows, RTX 6000 Ada) | Medium |
| Reuse the prefix at inference: compute the taxonomy prefix's KV cache once and reuse it for every row. Require at least 99% top-3 agreement and identical dev MAP before using it. | `text_diagnose.score()`, the pod_predict path | About 78% fewer prefill tokens; saves about $1 per 32B test pass | $0-1 | Medium |
| Exact shared-prefix training (prefix KV cache with gradients). Only worth trying if removing the taxonomy hurts accuracy. | `TextSystem.explain_loss` / `hidden` | About 2.2x fewer tokens per batch. Unproven, and risky under HF gradient checkpointing. | About $1 | Low |
| Put LoRA on the top half of the layers only. Also remove `enable_input_require_grads()`, otherwise backprop still runs through every layer; `vision.py` first needs `use_reentrant=False`. | LoraConfig `layers_to_transform`; checkpointing setup in `text_diagnose.py`, `pod_train.py`, `vision.py` | Estimated 1.3-1.5x faster training; effect on accuracy unknown | About $1.5 (8B ablation) | Low |
| Parity-gate every speed change: a shorter prompt, prefix reuse, a different quantisation, or a GGUF or MLX export. Keep the 32B inference base on NF4, the same as in training. | `scripts/evaluate_*` and any new inference path | No score gain; stops speed work from silently changing predictions | $0 | High |
| Cut pod setup time: download the model while pip installs, keep a prebuilt environment on a network volume, and terminate the pod automatically when the COMPLETE marker appears. | `scripts/runpod_pod.py`, pod `run.sh` | About 10-15 min saved per launch (about $0.3-0.45 each) | Cents of storage | Medium |
| Batch the image classifier's training forward pass (it currently runs 1 row per pass). | `src/k12/pod_train.py` `run()` / `loss_terms()` | Estimated 1.2-1.6x faster; matters only if another vision run is planned | About $0.5 | Low |

**What won't work:**
- Running Kimi K3, or any MoE of 235B or more, on CPU or by streaming weights from SSD over the test set. It would take weeks, and the engine has no vision encoder. K3 also exceeds the competition's model-size allowance.
- Expecting a bigger model plus inference tricks to close the gap. The 235B model beat the 32B by +0.054 with the same information, but alone it scores below our ensemble, and adding it to the ensemble gave +0.012 (CI spans 0).
- Rewriting the Python glue (tokenising, PIL preprocessing, blending, stacking) in Rust or C++. It is under 1% of wall-clock.
- Hand-written C/SIMD kernels, expert LRU caching, O_DIRECT, hugepages and layer pinning. These solve "the model does not fit in RAM", and our models already fit on the GPU.
- MXFP4 weights. They need a checkpoint trained for that quantisation, and the A100 has no FP4 tensor cores.
- n-gram speculative decoding, because our reasoners are classifiers.
- Using the repo's int4 study to judge our 4-bit setup. It measured only the reconstruction error of naive per-row int4, not NF4 or GPTQ, and our NF4 QLoRA already works.

---

## Q4 - arXiv 2508.06107 ("Mask & Match")

**What it is:** "Mask & Match: Learning to Recognize Handwritten Math with Self-Supervised Attention" (Mitra, Chakraborty, Sahu). It converts one formula image into LaTeX in three steps:
1. Global and patch-level MoCo contrastive pretraining of a DenseNet encoder.
2. A masking curriculum that trains attention to stay consistent while the visible area grows from 30% to 100%.
3. Fine-tuning with a CoMER decoder.

It reports 63.35 / 63.8 / 65.9 ExpRate on CROHME 2014/16/19, but its own tables give two different 2019 figures (64.20 and 65.90). The authors withdrew the paper on 28 Aug 2025, saying the results need "substantial improvement and further validation". The GitHub repo holds only a README, with no code, weights or licence.

**Does it help us?** No. It solves a different task (reading single formulas), it is withdrawn, and it ships nothing. Its obvious text analogue, training on a second reader's noisy transcripts, has already been tried here. Mixing in 11,426 Claude-transcribed rows left the gold-vs-machine gap at 0.025-0.028.

**What we can try:**

| Idea | Where in pipeline | Expected effect | Cost | Confidence |
|---|---|---|---|---|
| Final-answer lookup for the same question (our idea, not the paper's). Map (question_id, final value from the machine transcript) to a smoothed code distribution built from the 26,202 Claude-labelled unlabelled photos, and add it at low weight to the blend. Choose the weight by cross-fitting or on another fold. It stays within the rules: no test traces are in the Claude pool, and test values come from the offline transcriber. It covers 44.8% of test rows. | Combining / blend (CPU) | Measured +0.001 to +0.004 on the stronger 0.7143 base, with intervals that include 0. Realistic expectation 0 to +0.005. | $0 CPU | Medium |
| Root-line lookup: match every machine-transcribed line, not just the final one, against Claude's normalised root lines for that question. Optionally pass a match as a hint in the prompt. | Root-step localisation and code choice (CPU feature) | Unmeasured; estimate 0 to +0.01 on top of the final-answer lookup. Measure coverage on CPU first. | $0 CPU | Low |
| Synthetic corruption curriculum for the text reasoner: drop or merge lines, swap digits and operators, mark lines [unclear], easing from heavy noise to clean text. This is the same lever as the Q1 corruption idea, so run them as one experiment, and only if budget remains. | `text_diagnose.py`, where `fit_text` is built | Estimate 0 to +0.01. It cannot fix error-family mistakes that also happen on gold text. | About $1-2 (8B retrain) | Low |
| Masked-line counterfactuals at inference: delete each line and measure how far each of the top 3-5 codes' probabilities drops. Down-weight codes that depend on no line. | Combining, using the 8B reasoner | Estimate -0.005 to +0.01; risks pushing predictions toward NONE | About $4 for test (about 35k forward passes) plus about $0.4 to validate on dev | Low |

**What won't work:**
- Reusing Mask & Match code or weights. None exist, and the paper is withdrawn.
- Replacing our Qwen3-VL-8B reader with a DenseNet formula reader. Reading is worth about 0.03, and the PP-FormulaNet pilot gave no diagnosis gain (0.656 vs 0.664).
- Contrastive pretraining on the 50k unlabelled photos. It needs many GPU hours, far beyond $17, and would only improve reading.
- Mixing in a second reader's transcripts to close the reading gap. This was already done, and the gap did not shrink.
- Expecting any of these ideas to deliver the error-family and rare-code gains that 0.82 needs.

---

## Q5 - Training on different or additional datasets

**What it is:** I checked two kinds of extra training data:
1. **More in-domain data:** more of the 50k unlabelled competition photos, labelled by Claude, by our own ensemble, or by synthetic error injection.
2. **External public datasets:**
   - Handwriting reading: MathWriting, HME100K, CROHME.
   - LLM step verification: ProcessBench, PRM800K.
   - Student errors and misconceptions: Eedi 2024, MAP 2025, ScratchMath, FERMAT, ErrorRadar, MaE.
   - Indonesian handwriting data.

Three findings:
- More in-domain data is close to saturation.
- No external set uses our 75-code label space.
- The unlabelled pool is skewed: Claude labels 71% of it NONE, against 29% in train.

**Does it help us?** Partly.
- All 11,426 Claude rows add only about +0.025 over a properly trained 8B model that uses real rows only.
- Correcting for the pool's label skew after training helps a single pseudo-trained model (+0.027) but gives about 0 on the submitted ensemble (cross-fit -0.010 to +0.001).
- External sets do not match our labels or input type, and several are non-commercial.

**What we can try:**

| Idea | Where in pipeline | Expected effect | Cost | Confidence |
|---|---|---|---|---|
| Use the label-skew correction only where it measurably helps: on a future ensemble whose members are all pseudo-trained. Compute each member's correction from its own training mix, never correct a member trained without pseudo rows, and re-measure the full blend with cross-fitting grouped by question. | Calibration before the blend (`scripts/blend_geometric.py`) | About 0 on the current ensemble (measured). Positive only if the ensemble's members change; unmeasured. | $0 CPU | Low |
| If an 8B retrain happens anyway, clean its targets first. Remove NOT-01, NOT-02 and REL-01 from the soft secondary slots (NOT-01 is about 57x over-represented). Re-weight pseudo rows toward the train code mix, keep NONE at about 1/3, and train more than 1 epoch on the real rows. Check on dev before paying for a full-data run. | `scripts/build_claude_distill.py`, then `text_diagnose.py` | Unmeasured; estimate 0 to +0.01 for the ensemble | About $6.3 per 1-epoch run. A dev-checked run plus a full-data run costs $14-27, at or above the budget left. | Low |
| If the last ~$13 of API is spent, target it rather than sampling at random: unlabelled photos where the ensemble's P(NONE) < 0.4, from questions with wrong-family dev errors or rare codes. Worth it only together with a retrain. | `select_unlabelled.py` -> `claude_label.py` -> `build_claude_distill.py` | Estimate under +0.01 (random sampling: about 0) | About $13 (about 1,830 photos) | Low |
| Generate rule-based synthetic errors for rare and easily confused codes (sign flips, adding numerators and denominators, left-to-right order of operations, distribution errors). Apply them to correct transcriptions and label each with the injected code. | New script feeding rows into `text_diagnose.py` | Perfect rare-code handling is worth at most about +0.015 overall; the realistic gain is small and unmeasured | $0 to generate; shares the retrain cost above | Low |

**What won't work:**
- Label-skew correction on the current 0.718 ensemble. Cross-fit gives 0.708-0.719, and the CI spans 0.
- Random Claude labelling of more photos. 11.4k rows gave about +0.025 in total; $13 buys about 1,830 photos, and labelling the whole pool would cost about $174.
- Self-training on our own ensemble's labels. It is a weaker teacher (0.718 vs 0.819) and risks reinforcing its own mistakes.
- A per-question prior from the 26k teacher labels: 0.410 on its own, +0.000 when blended.
- Handwriting datasets. MathWriting is CC BY-NC-SA, and the HME100K and CROHME licences are unverified. Their value is capped by the reading gain (about +0.03), they contain single expressions, and a non-commercial licence is a risk.
- ProcessBench and PRM800K. They label LLM errors as correct or incorrect per step and carry no misconception codes.
- Eedi 2024, MAP 2025, MaE, ScratchMath, FERMAT, ErrorRadar, MathCCS and DrawEduMath. Each has a different label space, typed text or deliberately perturbed solutions, is small or Chinese-only, or has an unverified licence.
- Indonesian data (arXiv 2506.04822): 14K maths and English answers, with no release stated.

---

## Bottom line

Ranked by expected gain per dollar. All gains are estimates unless marked as measured.

1. **Remove the taxonomy list from the prompt and test it on the 8B (Q3), about $1.5.** It is the only efficiency lever that matters. About 3.5x fewer tokens per row could make a second 32B run fit in the remaining $17. Check it on dev with a paired bootstrap.
2. **Re-rank existing probabilities with a code-family prior (Q2), $0 on CPU.** It is the only idea aimed directly at wrong-family errors, our main source of lost points. Measure it on fold 0 before any training.
3. **Final-answer and root-line lookups from the Claude-labelled pool (Q4), $0 on CPU.** The final-answer lookup measured 0 to +0.004; the root-line lookup is unmeasured. Give it a low weight in the blend, chosen by cross-fitting.
4. **If a retrain happens anyway, combine Q1, Q4 and Q5 into one run:** cleaned soft targets, re-weighted pseudo rows, synthetic transcript corruption, and more than 1 epoch on the real rows. Estimate 0 to +0.015 for about $6-13, so only one such run fits the budget.
5. **Process hygiene (Q3), $0-1:** parity-gate every speed change, reuse the prefix at inference, and cut pod setup time. No score gain, but it protects the budget and the predictions.
6. **Do not do:** Ultramarine or GLM-5.3-Flash, the Kimi CPU engine, Rust or C++ rewrites, the Guo or Mask & Match readers, external datasets, label-skew correction on the current ensemble, or random API labelling. None of them targets error-family or rare-code mistakes, and better reading is capped at about +0.03.

## Sources

**Q1**
- J. Guo, *Systems and Soft Computing* 7 (2025) 200347, https://doi.org/10.1016/j.sasc.2025.200347 (local text: /Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/research/maths-paper.txt)
- CoMER: https://arxiv.org/abs/2207.04410
- Uni-MuMER: https://arxiv.org/html/2505.23566
- https://arxiv.org/html/2511.02288
- CROHME test sizes: https://arxiv.org/pdf/2207.11463 and https://www.cs.rit.edu/~rlaz/files/CROHME+TFD%e2%80%932019.pdf

**Q2**
- https://huggingface.co/paperinstruments/Ultramarine-1-Flash
- https://huggingface.co/api/models/paperinstruments/Ultramarine-1-Flash (and /tree/main)
- https://huggingface.co/api/models/paperinstruments/Ultramarine-1-Flash-LoRA
- https://paperinstruments.com/blog/ultramarine
- https://huggingface.co/zai-org/GLM-5.3-Flash
- https://playground.roboflow.com/models/z-ai/glm-5-3-flash
- https://llmrun.dev/model/zai-org-glm-5-3-flash/benchmarks
- https://unsloth.ai/docs/models/glm-5.3-flash
- https://aiweekly.co/alerts/zai-glm-53-flash-runs-local-in-93gb-1-bit-via-unsloth
- https://dev.classmethod.jp/en/articles/dgx-spark-glm-5-3-flash-first-touch/
- https://www.ycombinator.com/companies/paper-instruments

**Q3**
- https://github.com/FareedKhan-dev/kimi-k3-in-c (README.md, CHANGELOG.md, docs/PERFORMANCE.md, docs/TUNING.md, docs/data/speed-2026-08.md)
- https://api.github.com/repos/FareedKhan-dev/kimi-k3-in-c
- https://github.com/ggml-org/llama.cpp/discussions/4167
- https://www.crowdfundinsider.com/2026/07/294116-moonshot-ai-wraps-up-kimi-k3-rollout-with-full-model-weight-release/
- https://aiweekly.co/alerts/moonshot-open-sources-kimi-k3-a-28t-parameter-moe-model

**Q4**
- https://arxiv.org/abs/2508.06107 (v1: https://arxiv.org/html/2508.06107v1; withdrawal: https://arxiv.org/abs/2508.06107v2)
- https://github.com/shreemitra/Mask-Match

**Q5**
- MathWriting: https://arxiv.org/abs/2404.10690 and https://github.com/google-research/google-research/tree/master/mathwriting
- ProcessBench: https://huggingface.co/datasets/Qwen/ProcessBench
- PRM800K: https://github.com/openai/prm800k
- ScratchMath: https://huggingface.co/datasets/songdj/ScratchMath
- FERMAT: https://arxiv.org/abs/2501.07244
- ErrorRadar: https://arxiv.org/abs/2410.04509
- MAP 2025 mirror: https://huggingface.co/datasets/kaggle-map/data
- MaE: https://huggingface.co/datasets/nanote/algebra_misconceptions
- Correct Answer Trap: https://arxiv.org/abs/2606.23205
- Indonesian classroom assessment: https://arxiv.org/abs/2506.04822
- Not re-verified: MathCCS https://arxiv.org/abs/2502.13789, DrawEduMath https://arxiv.org/abs/2501.14877

**Local evidence** (under /Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/)
- src/k12/text_diagnose.py, pod_train.py, vision.py
- scripts/build_dataset_v3.py, scripts/build_claude_distill.py
- reports/claude_scaling.json, reports/runpod_budget.json, reports/claude_budget.json
- runs/pod/{text32,tx,dev-distill,scaling,probe235}/
- data/raw/taxonomy_v1.csv
- docs/htr_survey_takeaways.md, docs/competition_research.md
- Fact-check scripts: runs/research/q1-factcheck/, runs/research/q4-arxiv-2508/, runs/research/q5-factcheck/
