# Scale probe: is a 200B+ model worth it? (11 Nov 2026)

One H200 (pod `09klqgmxtmpqhn`, $5.29/hr, 14:20–15:34 UTC, about $6.50). vLLM 0.33, thinking mode, temperature 0.6.
All runs read the machine transcripts `runs/pod/tx/tx_dev.json` of the 301 fold-0 development rows. Nothing was trained.
Few-shot = up to 4 labelled training rows (non-development folds only: same question first, then same topic) with gold
transcripts, root code and the annotator's reason, shown before each row. Scores are rank shares 0.6/0.25/0.15.

| Model (301 dev rows) | Zero-shot | Few-shot (4 ex.) | Fine-tuned (ours) |
|---|---|---|---|
| Qwen3-32B (bf16) | 0.518 | 0.596 | 0.652 (QLoRA reasoner, epoch 1) |
| Qwen3-235B-A22B (GPTQ-Int4) | 0.584 | **0.650** | not tried |
| Qwen3-8B | | | 0.667 |
| Offline 3-way ensemble | | | 0.713 equal weights / 0.718 tuned |
| Claude Opus 5.5 | | | 0.819 (teacher) |

- Scale at equal information: 235B − 32B few-shot = +0.054, paired bootstrap 95% CI [+0.014, +0.094]. Zero-shot gap +0.066.
- Fine-tuning at 32B is worth about +0.056 over few-shot (0.596 → 0.652). If a 235B gained the same, it would reach about
  0.70–0.71 as a single model. That is an extrapolation, not a measurement.
- 235B few-shot: errors 0.568 (n=200), NONE 0.810 (n=101); unseen questions (n=39) 0.701.
- Zero-shot 235B with a 3000-token limit truncated 42 rows (0.525). The rerun with 9000 tokens gave 0.584 with nothing unparsed.
- Ensemble on dev (equal-weight 3-way 0.713): adding 235B few-shot gives 0.725–0.730 (w=1 gain +0.012, CI [−0.014, +0.037]).
  The zero-shot runs and the 32B few-shot run do not help (−0.016 to −0.032).

Recommendation: do not fine-tune a 235B model now. A fine-tune is estimated at $150–300 on multiple GPUs, which exceeds the
remaining RunPod cap ($27.2 of $70 left). The deadline is 17 Nov, there is no ready LoRA path for the GPTQ MoE checkpoint,
and the expected single-model gain (about 0.70) only matches the current ensemble. The one cheap use is 235B few-shot as a
4th ensemble member on test. 3,498 rows at the measured 2.6 s/row would take about 2.8 h on one H200 with setup, about $15.
It needs the user's approval, because the gain on dev is not statistically secure.

Pod setup lessons (fixed in `scripts/probe_setup.sh`): vLLM 0.33 needs the venv's CUDA 13 NVRTC on `LD_LIBRARY_PATH`, the
venv `bin` on `PATH` (ninja), and `CUDA_HOME` pointed at `site-packages/nvidia/cu13` or `VLLM_USE_FLASHINFER_SAMPLER=0`
(the image's nvcc 12.4 cannot build FlashInfer). Hugging Face downloads reach about 520 MB/s with 16 workers, while pip
runs at about 9 MB/s, so start the model download in parallel with `pip install`.

Files: `zeroshot235/`, `fewshot235/`, `fewshot32/`, `zeroshot32/` (validation.npz bundles, responses.jsonl, metrics.json,
run.log); `pass1/` = zero-shot with the 3000-token limit; `logs/` = the three failed launches and setup.log.
