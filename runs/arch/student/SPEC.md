# Candidate-aware 8B student ("Stage B"), build spec

Agreed plan: `runs/arch/PLAN.md` Stage B, merged with `runs/research/ARCHITECTURE_FROM_PAPERS.md` changes #2 and #3.

**Goal.** Fix the measured failure: on familiar questions, an error type never seen for that question gets the question's usual codes (dev 0.431 vs Claude 0.712 on 84 rows). The student reads the question, the answer key and the transcript. It sees ~10 candidate codes, each with its meaning and a same-question example. It writes a short diagnosis first, then picks a letter. The letter probabilities become a 76-way bundle, which we fuse with the current blend and then label-balance.

## Files (new; do not edit existing src/ or scripts/ files)

| File | Role |
|---|---|
| `scripts/build_candidate_dataset.py` | Writes `data/v5/cand_train.jsonl`, `data/v5/cand_dev.jsonl`, `data/v5/cand_test.jsonl` and `reports/cand_dataset.json` |
| `src/k12/candidate_student.py` | `train` and `predict` subcommands, run as `python -m k12.candidate_student ...` |
| `scripts/fuse_candidate.py` | Dev fusion, gates and test staging (CPU) |

## Row format (jsonl)

The prompt is fully rendered by the builder, so the trainer stays generic.

**Train row:**
```json
{"trace_id", "source": "claude"|"gold", "phase": "a"|"b", "prompt": str, "target": str,
 "letters": ["A",...], "codes": ["EXP-01",...], "soft": {"B": .7, "E": .2, "A": .1}, "weight": float}
```

**Dev/test row:**
```json
{"trace_id", "prompts": [prompt_order1, prompt_order2], "codes_per_prompt": [[...],[...]], "letters": [...]}
```
- order2 lists the candidates in reverse order, with letters reassigned.
- For dev, also include `"truth"` (the gold root_code, used only for reporting).

## Prompt template

```
Grade: {grade}
Topic: {topic_name}
Question: {question_text}
Answer key: {correct_answer}
Student working (line by line):
{transcript}

Candidate error codes:
A. {code}: {name} - {description, cut to 140 chars}{example}
B. ...
Find the root major error the rest of the working follows from (or NONE if the working has no major error) and choose the matching candidate.
```

- **`{example}`:** ` On this question, e.g. "{span, cut to 60 chars}"` when an example exists; otherwise empty.
- **NONE line:** `NONE: No major error - the working is correct or has only minor slips.`
- **Language:** question text and transcripts are Indonesian. Keep them verbatim.

## Target

Target (train only; reason first, letter last; about 120 tokens or fewer):
```
ROOT_LINE: {Ln or -}
WHY: {diagnosis, at most 70 words}
ANSWER: {letter}
```
- **Claude rows:** ROOT_LINE comes from `answer.root_line` (`-` if empty). WHY is the first sentences of `answer.reasoning`, up to 70 words, cut at a sentence end where possible. The letter is the one for `first_code`.
- **Gold rows:** ROOT_LINE is the `step` of the misconceptions entry with `root: true` (`-` for NONE). WHY comes from `runs/claude/train_explain_opus.jsonl` `answer.reasoning`, first ≤70 words (if missing, use the root misconception's `text`). The letter is the one for the gold `root_code`.
- **`soft`:**
  - Claude rows: `.7/.2/.1` over the letters of first/second/third code. Skip codes not in the candidates and renormalise.
  - Gold rows: `{letter: 1.0}`.

## Candidates

k = 10 codes, always including NONE.

**Train rows** (no model predictions exist for these rows). Fill in this order, skipping duplicates, until there are 10:
1. truth (first_code / root_code)
2. NONE
3. Claude's 2nd and 3rd codes (Claude rows only)
4. codes seen for the same `question_id` in the fit gold rows and in the Claude rows (excluding this row), most frequent first, at most 4
5. same-section siblings of the truth: the taxonomy `section` column, random order
6. random codes

Then shuffle the order with a per-row seeded RNG, and assign letters A–J in display order.

**Dev rows:** the top 10 of the Sinkhorn-balanced dev blend.
- Base blend (geometric .35/.30/.35):
  - `runs/pod/text32/validation_tx_dev_epoch_1.npz`
  - `runs/pod/tx/text_tx_dev_e3.npz`
  - `runs/pod/dev-distill/validation_epoch_2.npz`
- Balancing: Sinkhorn with λ = .5 toward the fit-row label marginals (+0.5 smoothing).
- The same code is in `runs/arch/label-structure/s7_sinkhorn.py` and `common.py`.
- If NONE is not in the top 10, it replaces the 10th.
- Order 1 = by descending blend score; order 2 = reversed.
- Report recall@10 of the truth (expect about 0.94).

**Test rows:** the same, from the test blend.
- Test bundles: `runs/pod/text32-full/test.npz`, `runs/pod/text-full/test.npz`, `runs/pod/full-distill/test.npz`.
- Balance toward the marginals of all 1,500 train rows (as in `runs/arch/label-structure/s11_none_falarm_and_test.py`).

## Examples (exemplar spans)

**For candidate code c on question q:**
- Prefer a gold row (one with root_code == c on question q): use the `span` of its root misconception.
- Otherwise use a Claude row (first_code == c on question q): take the text of its `root_line` line from `answer.transcription`, with the `Lk | ` prefix stripped.
- Never use the row itself.
- Choose deterministically, e.g. the first match after seeded sorting.

**Which gold rows each split may use:**
- train and dev: fit gold rows only (`make_folds(train)` fold != 0).
- test: all 1,500 gold rows.

Claude rows (unlabelled photos) can serve any split.

## Training data selection

**Claude rows**
- Source: `runs/claude/unl_all_v4.jsonl` (26,202 rows), with metadata from `data/semisupervised/competition_data/unlabelled.csv`.
- Keep a row only if its first_code is in the label space and its transcription is non-empty.
- Cap: `--per-code-cap` 200 rows per first_code; `--none-cap` 2,000 NONE-first rows. Choose with a seeded RNG.
- Weights: 0.8, or 0.8 × 0.3 when first_code == OTH-00.
- Transcript: Claude's `answer.transcription`.
- Phase: "a".

**Gold fit rows**
- 1,199 rows, using the gold `transcription_steps` as transcript.
- Weight 1.0.
- Phase "b", repeated `--gold-repeats` 2 times, each copy with a different candidate shuffle.

**Dev and test transcripts:** machine transcripts `runs/pod/tx/tx_dev.json` and `runs/pod/tx/tx_test.json`.
- Cap any transcript at 1,500 characters, keeping the last part if it is longer, since some test transcripts loop.

**Hard rule.** No dev trace_id (fold 0) may appear anywhere in cand_train: not as a row, not as an exemplar source.

## Trainer (`src/k12/candidate_student.py`)

**Model**
- Qwen3-8B base plus LoRA (r=32, alpha 64, dropout .05, all attention and MLP projections). bf16 on CUDA.
- `--device cpu` with float32 for smoke tests: the cached tiny model `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` at `~/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/*`.
- Optional `--load-4bit`.

**Sequence**
- Input = `prompt + "\n" + target + eos`.
- Right padding. Left-truncate only the prompt's working section if over `--max-length` (default 2048); never truncate the candidate block or the target.

**Loss** (per row, times the row weight; mean over the batch)
- `lm_weight` (default 1.0) × token cross-entropy on the target tokens, excluding the answer letter token.
- `+ letter_weight` (default 1.0) × soft cross-entropy at the answer position, restricted to the candidate letter tokens. Take the logits at the position predicting the letter token after `"ANSWER:"`, gather the ids of the candidate letter tokens (each `" A"`, `" B"`, … must be one token; assert it), log-softmax over those, and dot with `soft`.

**Schedule**
- AdamW, lr 1e-4, cosine with 5% warmup.
- Phase "a" rows first (shuffled), then phase "b" rows (shuffled), within each epoch. Phase b uses `--phase-b-lr-mult` (default .3) × the scheduled lr.
- `--epochs` 1; batch size and grad accumulation configurable; gradient checkpointing with `use_reentrant=False`.
- Log a JSON line every 25 steps: loss, letter loss, rows, seconds, rows/s.
- Save the LoRA adapter at the end.

**Predict**, for each prompt order:
1. Greedy-generate up to `--max-new-tokens` (default 160), stopping after `"ANSWER:"` is produced (or at EOS).
2. If `"ANSWER:"` appears, run one forward on prompt + generated text up to and including `"ANSWER:"`, and read the logits over the candidate letter tokens.
3. If it does not appear, append `"\nANSWER:"` and read the logits there.
4. Softmax over the candidate letters gives code probabilities.

Then:
- Average the 2 orders' code probabilities (`--orders` 1|2).
- Fill non-candidate codes with 1e-4 and renormalise. Save with `k12.core.save_bundle` (same format as the other bundles: trace_ids, classes, p).
- Also save `generations.jsonl` (trace_id, order, text, letter_probs).
- Batch the generation, and log the parse rate and rows/s.

## Fusion and gates (`scripts/fuse_candidate.py`, CPU)

1. P0 = the raw dev blend. Fused = normalize(P0 · P_B^w); then Sinkhorn (λ ∈ {0, .25, .5, .75}).
2. Choose w ∈ {0, .25, .5, .75, 1, 1.5} and λ jointly by 5-fold question-grouped cross-fitting (as `s7_sinkhorn.py` does).
3. Report the cross-fitted MAP@3 against the blend+Sinkhorn reference (0.737), with the question-grouped paired bootstrap 95% CI (reuse `boot_ci` in `runs/arch/label-structure/common.py`).

**Gates (all must pass):**
- (a) Δ ≥ +0.015 over blend+Sinkhorn, with the CI's lower bound > 0.
- (b) MAP@3 ≥ 0.50 on the dev error rows whose code is new for the question, plus the unseen-question error rows (blend 0.431). Definitions as in `runs/arch/ensemble-anatomy/`.
- (c) Top-1 agreement with Claude (`runs/claude/dev_opus.jsonl`) ≥ 0.65 on the unseen-pair rows.
- (d) Top-1 agreement with Claude ≥ 0.45 on the rows where the blend and Claude disagree.

Also report the student alone, its parse rate, and per-order scores.

**`--stage-test`:** apply the chosen w and λ to the test bundle. Write `artifacts/submission-3way-student/submission.csv` and `metrics.json` (with sha256 and `"submitted": false`). Never submit.

## Constraints

- CPU only on this Mac. No pods, no APIs, no Kaggle.
- Run with `PYTHONPATH=src:scripts .venv/bin/python`.
- Match the repo's code style: compact, few comments; see `src/k12/text_diagnose.py`.
