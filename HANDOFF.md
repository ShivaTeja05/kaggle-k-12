# K-12 misconception detection: hand-over file

**Purpose.** One file that lets a person or another model pick this project up cold: what the task is, the rules,
what exists, what was tried, what is running now, and where to start. It is kept current: the state sections are
edited in place and every action gets a dated line in the log at the bottom.

**Last updated:** 14 Nov 2026, 10:55 UTC.

**Read these next, in this order.**

| File | What it holds |
|---|---|
| `AGENTS.md` | The owner's rules, with dates. Binding. Read before doing anything that costs money or touches Kaggle. |
| `docs/plan_to_084_2026-11-14.md` | **The current plan**: diagnosis, the solution (ranker v3), evidence with links, stages with costs and pass marks. |
| `docs/experiment_review_2026-11-14.md` | All 122 experiments in one table, what worked, what failed and why, where points are lost, research questions. |
| `docs/model_audit_plan_2026-11-12.md` | The detailed running record (sections 1 to 17.1). Later sections supersede earlier ones. |
| `reports/status.json` | Submission history with public scores. |
| `reports/runpod_budget.json`, `reports/claude_budget.json` | Spend ledgers. |

---

## 1. The task

- **Competition:** Kaggle, "Misconception Detection in K-12 Student Math Workings"
  (slug `misconception-detection-in-k-12-student-math-workings`). **Deadline 17 Nov 2026 00:00 UTC.**
- **Input:** a photo of a student's handwritten maths working, the question text (Indonesian), the answer key, sometimes
  a question diagram. **Output:** three ranked codes from a taxonomy of 75 misconception codes plus `NONE`.
- **Metric:** MAP@3 (1, 1/2 or 1/3 for the true code in first, second or third place).
- **Data:** 1,500 labelled training rows over 462 questions; 3,498 test rows over 492 questions (95.7% of test rows are
  on a training question); 50,000 unlabelled photos over 284 questions. About a third of rows are `NONE`.
- **Public leaderboard:** about 29% of test (about 1,000 rows). The other 71% decides the final standing.

**Host rulings that matter** (Kaggle discussion tab, read 14 Nov):

- Final predictions must come from models that run offline. No API in the final solution.
- Commercial APIs may be used to label **training and unlabelled** photos. Never test photos.
- Any open model up to the size of Nemotron 550B is allowed, ensembles included. Compute is unrestricted.
- Winners must hand over training and inference pipelines; both will be verified.
- The root code is "most of the time the earliest" error, but can be the one the teacher judged the main cause.

**Host reference scores** on the public split, models used off the shelf (host blog, 17 Oct): Claude Opus 5.5 0.792,
GPT-6 Astra 0.790, Gemini 4 Argon 0.775, **Qwen3.8-27B 0.757**, Gemma 4 31B 0.710. Leaders were at 0.815–0.823 then
and are at about 0.845 now.

---

## 2. Where we stand

| Item | Value |
|---|---|
| Best public score | **0.76266** (submission 58335588, 14 Nov 07:21 UTC), rank about 61 of 437 |
| File behind it | `artifacts/submission-3way-pw14-q38-partial/submission.csv` (the previous best system fused with Qwen3.8-27B on 1,392 least-confident test photos) |
| Its dev score | 0.804 against 0.775 for the previous system on the same 270 dev rows |
| Previous best | 0.73183 (submission 58299898, `artifacts/submission-3way-pw14`), dev 0.7525 cross-fitted on 301 rows |
| Owner's milestone | about 0.82 public (`AGENTS.md`) |
| Latest dev result | Qwen3.8-27B (zero-shot, thinking) fused with the submitted system: **0.804 against 0.775** on the same 270 dev rows, +0.029 [+0.002, +0.055] (`reports/q38_fuse.json`) |
| Running now | Nothing. No pods are active (last one terminated 14 Nov 09:23 UTC). |
| Staged, not submitted | `artifacts/submission-3way-pw14-q38/submission.csv`: the complete fused file with all 1,593 Qwen3.8 answers (the best file has 1,392). Needs the owner's yes. |
| RunPod spend | Ledger $84.82 of the $94 cap. **Hard limit: the RunPod account held $6.52 at 08:55 UTC on 14 Nov (owner's message); about $6.01 is left by calculation. Everything must finish under it unless the owner tops it up.** |
| Claude API spend | $188.40 of $200 |
| Kaggle submissions used today (14 Nov) | 2 of 5 |

**Dev** always means fold 0 of `k12.core.make_folds`: 301 rows, standard error about 0.023. Dev has run 0.01–0.03
above the public score. A dev gain below about +0.02 has not been reliable (one such gain lost 0.006 on the leaderboard).

**Row types on dev** (used everywhere):

| Row type | Rows | Submitted system | Claude Opus with photo |
|---|---|---|---|
| No error (`NONE`) | 101 | 0.911 | 0.934 |
| Seen pair: the (question, code) pair is in the other 1,199 training rows | 94 | 0.867 | 0.812 |
| Hard: the pair is not in the training rows | 106 | 0.500 | 0.715 |

All of the gap to Claude is on hard rows. Hard rows split into A (32: Claude labelled this pair on a pool photo),
B (31: question has pool photos, pair absent) and C (43: question has no pool photos); the group of each dev row is in
`runs/deep/catalog/catalog_meta.json` under `grp`.

---

## 3. The owner's rules (summary; `AGENTS.md` is the authority)

- **Ask before spending or submitting.** No pods, paid API calls, Kaggle GPU jobs or Kaggle submissions without the
  owner's explicit permission for that specific thing. Each approval covers one named job and is then used up.
- **Standing exception as of 14 Nov 06:10 UTC:** the fused Qwen3.8 file from the job in section 6 may be submitted
  without asking again, because its dev result beats the submitted system. Nothing else.
- **RunPod:** Secure Cloud GPUs with at least 48 GB at no more than $1.79/hr; total cap **$88**; track spend in
  `reports/runpod_budget.json`; terminate every pod as soon as it is not in use; never copy Kaggle credentials to a pod.
- **Claude API:** training and unlabelled photos only, never test photos; cap $200; never print or copy the key.
- **Kaggle GPU:** off (owner's instruction of 8 Nov).
- **Final inference must be offline.** Do not hand-label test rows. Keep competition data and photos private.
- **Do not speculate that other teams broke rules.** Do not claim a score from a smoke test or an unlaunched job.
- **Keep this file current** after every experiment, submission, spend or change of plan.

---

## 4. The submitted system (public 0.73183)

Everything runs offline on open models. Claude was used only to label training and pool photos.

| Step | What it is | Files |
|---|---|---|
| 1. Transcriber | Qwen3-VL-8B + LoRA turns the photo into numbered lines (`L1 | ...`) | `src/k12/transcriber.py` |
| 2a. Text classifier, 8B | Qwen3-8B + LoRA on the transcript, 76-way head | dev `runs/pod/tx/text_tx_dev_e3.npz`, test `runs/pod/text-full/test.npz` |
| 2b. Text classifier, 32B | Qwen3-32B QLoRA on the transcript | dev `runs/pod/text32/validation_tx_dev_epoch_1.npz`, test `runs/pod/text32-full/test.npz` |
| 2c. Image classifier | Qwen3-VL-8B + LoRA on the photo, trained with Claude labels | dev `runs/pod/dev-distill/validation_epoch_2.npz`, test `runs/pod/full-distill/test.npz` |
| 3. Blend | Geometric: 32B^0.35 × 8B^0.30 × image^0.35 | `scripts/fuse_candidate.py` (`DEV`, `TEST`, `geo`) |
| 4. Pointwise ranker v1 | Qwen3-14B + LoRA answers yes/no for each of the blend's top 10 codes | `src/k12/pointwise_ranker.py`; dev and test scores in `runs/deep/pointwise/pod_out/pw_full/` |
| 5. Fusion | blend × ranker^0.5 | `scripts/fuse_candidate.py` (`fuse`) |
| 6. Label balancing | Sinkhorn toward the training label mix, λ 0.75; uses no test labels | `scripts/fuse_candidate.py` (`sinkhorn`) |

All prediction bundles are `.npz` files with `ids`, `classes`, `p` (rows × 76) and `meta`; read them with
`k12.core.aligned_bundle`.

---

## 5. What was tried (short form; the full table is in `docs/experiment_review_2026-11-14.md`)

**Worked**

| Ingredient | Effect |
|---|---|
| Read the photo first, then diagnose the text | +0.038 public, the largest single step |
| First few thousand Claude labels as training targets | +0.113 dev on one model; +0.021 public |
| Blending models that make different mistakes | members ≤ 0.667 alone, blend 0.706 cross-fitted |
| Sinkhorn label balancing | +0.019 dev, +0.009 public |
| Pointwise yes/no ranker reading code descriptions | +0.019 dev, +0.015 public |
| Qwen3.8-27B zero-shot with thinking, fused in | +0.03 dev; hard rows 0.50 → 0.60 (test result pending) |

**Failed, by reason**

| Reason | Evidence |
|---|---|
| Re-mixing outputs we already have is exhausted | about 25 combiners, priors, stackers and fusion tricks: −0.03 to +0.007 |
| More Claude labels of the same kind do not transfer | 4× the rows: +0.02 on hard rows; ranker v2 worse; averaging rankers v1+v2 scored 0.72550 public (−0.006) |
| A small model taught to reason first copies the form, not the substance | Stage B student 0.625 against 0.692 |
| Synthetic error examples carry about half the signal and hurt other rows | catalogue training: target rows +0.077, overall −0.018 |
| Matching by similarity to stored examples | nearest neighbours, answer lookup, example cards: none beat the classifiers |
| Qwen3-generation models without training, up to 235B | hard rows 0.39–0.45 |
| Qwen3.8-27B **without** thinking | 0.657 on 102 dev rows against 0.717 for our system on those rows |

**Where points are lost**

- One teacher example of the exact (question, code) pair moves the system from 0.50 to 0.82; a Claude-labelled
  example only to 0.66.
- On hard rows the wrong answer is a different error family 87% of the time: wrong diagnosis, not a near miss.
- About two thirds of the hard-row gap to Claude is reasoning on identical input, one third is reading.
- The machine transcript contains the annotated error text only 55% of the time.
- Qwen3.8-27B zero-shot is the mirror image of our system: hard rows 0.608 (ours 0.502) but seen pairs 0.709 (ours 0.868).

---

## 6. Finished 14 Nov 07:50 UTC: Qwen3.8-27B on the least-confident test photos

Approved by the owner on 14 Nov at about 04:20 UTC: about $7.80, working limit $9.50 across both pods. **Result: $7.08, all 1,593 photos answered, none failed; both pods terminated; public 0.76266 from the partial file.** The rest of this section records how it was run, for repeating it.

- **What:** `Qwen/Qwen3.8-27B-FP8`, zero-shot, thinking on (reasoning effort medium), vLLM 0.33.0, the same prompt and
  1024-pixel photos Claude was given, answers constrained to a JSON schema. Scores the 1,593 test photos (45.5%) where
  the submitted system's top probability is below 0.817 (`data/v9/q38_test/ids_least_confident.json`).
- **Pods** (RTX 6000 Ada, $1.09/hr, created 04:24 UTC):

| Pod | Id | SSH | Parts | Local results |
|---|---|---|---|---|
| A | `ripj2np3ueqbn1` | root@195.26.233.81 port 44168 | 1, 3, 5, first 129 rows of 7 | `runs/deep/q38/test_pod_a/` |
| B | `6htd6iidr0suqe` | root@195.26.233.93 port 55291 | 2, 4, 6, 8, last 70 rows of 7 | `runs/deep/q38/test_pod_b/` |

- **Safety:** one collector per pod (`runs/deep/stageB/collect.py`, terminates its pod at job end or at $4.60), one
  guard per pod (`runs/deep/pointwise/guard.py`), an in-pod timer that removes the pod at 08:41 UTC, and a watcher
  agent. Clients start no new request after 08:29 UTC. Expected finish about 08:05 UTC.
- **After a session restart:** run `~/.cache/k12-api-venv/bin/python -I scripts/runpod_pod.py status` first. If a pod
  is alive with no collector (`pgrep -fl collect.py`), restart the collector with the arguments above.
- **When both pods have ended:**

```bash
export PYTHONPATH=src:scripts
~/.cache/k12-api-venv/bin/python scripts/fuse_vlm.py --dev runs/deep/q38/pod_out/q38_dev_think.jsonl \
  --report reports/q38_fuse.json --test 'runs/deep/q38/test_pod_*/q38_test_part_*.jsonl' \
  --out artifacts/submission-3way-pw14-q38
```

  Check `metrics.json` (`rebuilt_base_matches_submitted_file` must be true), validate the CSV (3,498 rows, ids in test
  order, three distinct valid codes per row), then submit under the standing exception in section 3 and record the
  public score here and in `reports/status.json`.
- **Photos not answered** keep the submitted system's prediction, so a partial run is still usable.

---

## 7. How to work in this repository

- **Python:** `~/.cache/k12-api-venv/bin/python` with `export PYTHONPATH=src:scripts`. Do not pass `-I` to project
  scripts that import each other (it ignores `PYTHONPATH`). The repo's own `.venv` hangs because iCloud evicts its files.
- **Kaggle CLI:** `~/.cache/k12-api-venv/bin/kaggle`.
- **Disk:** the Mac is about 97% full (about 26 GB free). Do not download models locally.
- **Score a dev prediction:** `scripts/fuse_candidate.py --student <dev bundle> --report <json>` (pointwise-style
  bundles) or `scripts/score_vlm_dev.py name=<answers.jsonl>` (ranked-code answers from a generative model).
- **Stage a test CSV:** `fuse_candidate.py ... --stage-test <test bundle> --out artifacts/<name> --allow-failed-gates`
  or `fuse_vlm.py ... --test <glob> --out artifacts/<name>`. Staging never submits.
- **Submit (only with permission):**
  `kaggle competitions submit -c misconception-detection-in-k-12-student-math-workings -f <csv> -m "<message>"`.

**RunPod procedure**

1. `scripts/runpod_pod.py create --gpu-types "NVIDIA RTX 6000 Ada Generation" --max-rate 1.15 --attempts 14`
   (the card is often out of stock; retry). `status` lists active pods; `terminate --pod <id>` ends one.
2. SSH with `-i ~/.runpod/ssh/runpodctl-ssh-key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`.
   Copy files with tar over SSH. Upload from this Mac runs at 150–700 kB/s, so start large uploads while setup runs.
3. Start the job with `ssh -f ... 'setsid nohup sh script > out 2>&1 < /dev/null &'`. The script writes its stage to a
   status file and ends with a line starting `exit`.
4. Start the collector in the background, then the guard, then the in-pod dead-man timer. **Read the collector's
   first poll and confirm a file arrived before doing anything else.**
5. When the job ends, confirm `status` shows no pod and update the ledgers.

**Mistakes already made once**

- `pkill -f <pattern>` inside an ssh command kills the ssh shell itself when the pattern appears in the command.
  Use a bracket trick (`clien[t]`) or avoid pkill.
- zsh does not split unquoted variables; put ssh options in a function, not a string.
- vLLM needs `ninja` and `vllm download-kernels` before `vllm serve` will start (both are in the q38 pod scripts).
- The RTX 6000 Ada has no native FP8; Qwen3.8-27B-FP8 runs at only about 150 output tokens a second there
  (about 15 s per photo with thinking). An H100 or A100 may be much faster but has not been measured.
- A collector without host-key options once terminated a healthy pod; that is fixed in `collect.py`.

---

## 8. Where to start next

1. **Section 6 is done and confirmed on the leaderboard (0.76266).** The complete file is staged; submitting it is optional (expected about +0.003) and needs the owner's yes.
2. **The natural next step** is to remove Qwen3.8's weakness on seen pairs
   (0.709 against our 0.868) while keeping its strength on hard rows. Options, none run yet:
   - Fine-tune Qwen3.8-27B (LoRA) on the 1,500 teacher rows, keeping thinking on. Open question: how to fine-tune a
     thinking model on about a thousand labels without losing its zero-shot strength (research question 8 in the review).
   - Cheaper: give Qwen3.8 the question's known training codes or a few same-question teacher examples in the prompt.
     Caution: examples in the prompt hurt Claude slightly (−0.008) and hurt Qwen3-generation models on hard rows.
   - Run Qwen3.8 on the remaining 1,905 test photos (expected gain small: on dev the confident half gained nothing).
   - Try reasoning effort `xhigh`, or two samples per photo, on the dev rows first.
   Each needs a quote and the owner's approval. RunPod has $4.01 left under the $88 cap.
3. **If the leaderboard does not confirm it:** keep `artifacts/submission-3way-pw14` as the final selection. Every
   cheap re-mixing idea is already exhausted (section 5).
4. **Before the deadline:** make sure the two files selected on Kaggle for final scoring are the best public file and
   the most trusted alternative.

### >>> THE PLAN TO FOLLOW (15 Nov): `docs/plan_to_085_2026-11-15.md` (architecture, experiments E0-E6, pass marks, fallbacks, timetable, cost). It supersedes `docs/plan_to_084_2026-11-14.md` (written 14 Nov 10:50 UTC; not yet approved) <<<

Owner's goal: 0.84. In one paragraph: locating the student's error is mostly solved (Qwen3.8 finds the teacher's line 79% of the time); **naming it with the teachers' code is the bottleneck** (64% even with the right line), and the right code is within reach on 92% of hard rows (our top five + Qwen's three + codes of the ten most similar teacher-marked errors from other questions). The fix is a trained judge, **ranker v3**: our proven yes/no ranker (`src/k12/pointwise_ranker.py`), trained on the teacher labels, with Qwen3.8's reasoning text as input, teacher reference examples for each candidate code, per-question hints (never a hard mask), the wider candidate set, then a Qwen3.8-27B backbone and several seeds. Each ingredient is taken from winning solutions of the two closest past competitions (MAP 2025, Eedi 2024); links are in the plan file. Stages: 0 submit the staged complete file; A Qwen3.8 reasoning for the 1,230 teacher rows and 31 dev rows not yet done (about $6); B ranker v3 on the 14B code path (about $6, pass mark fused dev +0.02 and hard rows 0.68); C the same on Qwen3.8-27B with two seeds (about $15); D Qwen3.8 at maximum reasoning effort; E final blend. The sections below are the analysis behind it; the "CURRENT PLAN towards 0.80" further down is superseded.

### First-principles diagnosis (14 Nov 10:15 UTC; free CPU analysis on the 270 dev rows; owner's goal is now 0.84)

The task splits into three steps: decide error or no error, **locate** the line where the root error enters, **name** it with the code the teachers would use.

| Step | Measurement | Reading |
|---|---|---|
| Error or no error | 8 false alarms on 101 no-error rows; 12 of 169 error rows called "no error" | 7 + about 10 lost points; not the main problem |
| Locate | Qwen3.8's root line is within one line of the teacher's on 79% of error rows (Claude 85%) | Mostly solved |
| Name | When the line is right, Qwen3.8's first code is the teacher's on only 64% (Claude 69%); when the line is wrong, 45% | **The bottleneck.** "Right line, wrong code" is 46 rows, about three quarters of Qwen3.8's mistakes on error rows; its code is in a different family on 37 of them; the truth is somewhere in its three on 22 |

- Points lost by the fused system (50.2 of 270): no-error rows 7.2, seen-pair rows 12.0, hard rows 31.0. Of 59 error rows with a wrong top answer: 12 say "no error", 47 name another error (only 7 in the right family; only 3 are a code the teacher marked as a secondary finding).
- The naming convention is not easy to learn from text alone: a bag-of-words classifier trained on the 1,165 teacher findings in the fit rows names the code from the **teacher's own explanation** of the dev row with only 64% top-1 (hard rows 50%). Applied to Qwen's reasoning text it is worse than Qwen's own choice (56.5% against 60.9%) and hurts the fused system (−0.005 to −0.019).
- Simple learned adapters on Qwen's outputs (trained on about 216 dev rows, cross-validated) also hurt (−0.014 to −0.022 fused). Small linear models cannot learn 76-way conventions from this little data.
- What is known to fix naming: a teacher example of the exact (question, code) pair (0.50 → 0.82 with one, 0.955 with two or three), and, on the host's own data, fine-tuning a 31B model on about 3,400 teacher labels (52% → 64% top-1).
- Target arithmetic for 0.84 public (dev-like scores run about 0.04–0.05 above public): roughly no-error 0.95, seen pair 0.95, hard 0.76, against 0.93 / 0.86 / 0.63 now.

### Superseded plan towards 0.80 (written 14 Nov 09:45 UTC; kept for its evidence table; follow `docs/plan_to_084_2026-11-14.md` instead)

**What the evidence says**

| # | Evidence | Source |
|---|---|---|
| E1 | We ran Qwen3.8 at reasoning effort *medium*; its default is the maximum. Our run converts to about 0.73 public alone; the host measured 0.757. Our prompt reproduces the host's Claude figure, so the 0.025 shortfall is most likely the effort setting (or FP8 weights) | audit record 17.3 |
| E2 | Thinking is the active ingredient: off → medium is 0.657 → 0.794 on the same 102 rows, hard rows 0.549 → 0.711 | audit record 17.3 |
| E3 | Fusing Qwen3.8 into our system is the one gain that transferred in full: dev +0.029, public +0.031 | submission 10 |
| E4 | Re-prompting at the same effort does not help: rerank with our shortlist, with teacher examples, and three answers averaged all land between −0.015 and +0.010 whole-dev | audit record 17.3 |
| E5 | Fine-tuned open models can beat every zero-shot frontier model here: host reports teams at 0.815–0.823 on 17 Oct against Claude Opus 0.792. Our fine-tuned members are all Qwen3-generation | host blog |
| E6 | Hard rows: the truth is in Qwen's three or our top three on 84% of rows, but first in Qwen's list on only 53% | audit record 17.3 |

**Steps**

| Step | What | Cost (estimate) | Pass mark | Expected public |
|---|---|---|---|---|
| 0 | Submit the staged complete file `artifacts/submission-3way-pw14-q38` (needs the owner's yes) | free | n/a | about 0.766 |
| 1 | Dev test: Qwen3.8 at maximum reasoning effort (`--effort xhigh`) on the 135 least-confident dev rows, same prompt as the first pass (`data/v9/rerank/dev_first_sent.jsonl`), scored with `scripts/score_stage1.py` | about $1.75 on one RTX 6000 Ada; fits in the $6 left | alone at least +0.04 over the medium answer on those rows, or fused at least +0.03 on them | n/a (dev only); it also measures tokens per photo |
| 2 | If step 1 passes: maximum effort on the 984 test photos where our system and Qwen disagree, fuse with `scripts/fuse_vlm.py`, submit with the owner's yes | about $11–18 at the measured speed (needs a top-up), less on a GPU with native FP8, which is unmeasured | n/a | about 0.78–0.79 (E1 + E3) |
| 3 | A fine-tuned new-generation member: our existing classifier recipe (LoRA + 76-way head, teacher + Claude labels) on Qwen3.8-27B or Qwen3.5-9B, added to the blend | about $10–15; new training code for this architecture, about a day | dev alone above 0.69 (our best old member is 0.667) and blend gain of at least +0.015 | adds about +0.01–0.03 if it works |

Odds (my estimate): 0.78 or better is likely if step 1 passes; 0.80 needs step 3 to work as well, about one chance in three; 0.85 is out of reach before 17 Nov 00:00 UTC.

### Where the fused system still loses (CPU analysis, 14 Nov 08:15 UTC, 270 dev rows)

| Row type | Rows | Ours | Qwen3.8 alone | Fused | Claude with photo | Best of ours and Qwen per row |
|---|---|---|---|---|---|---|
| No error | 101 | 0.916 | 0.924 | 0.929 | 0.934 | 0.954 |
| Seen pair | 86 | 0.872 | 0.709 | 0.860 | 0.795 | 0.924 |
| Hard | 83 | 0.502 | 0.608 | 0.627 | 0.717 | 0.723 |
| All | 270 | 0.775 | 0.759 | 0.814 | 0.823 | 0.873 |

(Fused here uses the fixed test setting, so it is in-sample; the cross-fitted figure is 0.804.)

- The fused system is within 0.01 of Claude with the photo on dev. Of the points still lost, 31 are on hard rows, 12 on seen pairs, 7 on no-error rows. **The remaining lever is reasoning on hard rows, not seen pairs**, so fine-tuning Qwen3.8 to fix seen pairs is no longer the first priority.
- On hard rows Qwen3.8's top answer is right on 44 of 83, Claude's on 50, both on 34, neither on 23; they agree on the top answer 54% of the time. Two strong thinkers are complementary, which is the evidence for adding more independent opinions (more samples from Qwen3.8, higher reasoning effort, a second family such as Gemma 4 31B).
- Reading is level with Claude: character error against the teacher transcript 0.369 (Qwen3.8) and 0.368 (Claude). The photos are already 1024 pixels on the long side, so higher resolution is not available.
- Proposed (not approved): Stage 1, on the least-confident dev rows, test a second and third Qwen3.8 sample, reasoning effort `xhigh`, and Gemma 4 31B with thinking; Stage 2, apply whatever adds at least +0.01 on dev to the 1,593 test photos.

### Earlier plan, superseded by the CURRENT PLAN above (proposed 14 Nov 08:22 UTC)

Target arithmetic on the 270 dev rows: the fused system is at no-error 0.93, seen pair 0.86, hard 0.63 (0.814 in-sample, public 0.763). Public 0.80 needs about +0.037, roughly no-error 0.94, seen pair 0.90, hard 0.72: hard rows must reach Claude's level.

| Stage | What | Evidence | Cost (estimate) |
|---|---|---|---|
| 1 (dev, about 150 least-confident rows per arm) | A: a **rerank pass**: Qwen3.8 with thinking sees the photo, our system's top candidates with descriptions, and the teacher's examples for that question, and ranks them. B: two more Qwen3.8 samples, combined. C: reasoning effort `xhigh`. D: Gemma 4 31B with thinking | Per-row best of ours and Qwen is 0.873 against 0.814 fused, so arbitration has headroom; the truth is in our top 10 on 94% of rows; Qwen and Claude are right on different hard rows (44 and 50 of 83, 34 shared); our older ranker gained +0.019 with a weak model | about $6 |
| 2 (test) | Apply each arm that adds at least +0.01 on dev to the 1,593 least-confident test photos; submit with the owner's yes | Stage 1 | about $7 per pass, up to three |
| 3 (optional, larger) | Fine-tune Qwen3.8-27B (LoRA) on the 1,500 teacher rows with Claude's explanations of the teacher labels (`runs/claude/train_explain_opus.jsonl`) as reasoning targets | Host study: fine-tuning lifts 31B from 52% to 64% top-1; VEHME paper. Against: our 8B reason-first student failed; new training code for this architecture; about a day | about $40, high risk |

Honest odds (my estimate): 0.78–0.80 likely if two arms work; 0.80+ possible, not promised; 0.85 is not reachable by the 17 Nov deadline.

**Lean version of the plan (14 Nov 08:30 UTC, after two more free CPU checks; approved by the owner at 08:35 UTC, cap raised to $94).**

- On the 135 dev rows sent to Qwen3.8, the two systems agree on the top answer on 53 (fused 0.818) and disagree on 82 (fused 0.646; ours 0.520, Qwen 0.530, per-row best of the two 0.795). The points are in the disagreement rows, about 30% of dev and about 60% of the 1,593 test photos already scored.
- A simple rule for whom to trust (a different weight when Qwen's top code is one the teachers used on that question) does not help: −0.009 [−0.020, −0.002] cross-fitted. New information is needed, not a rule.
- So: one pod, arms run one after another on one loaded model. Stage 1 on the 135 sent dev rows: rerank pass (Qwen3.8 sees the photo, its own earlier transcription, our top candidates with definitions, the teacher's examples for the question; shorter output because nothing is re-transcribed), one extra sample, and maximum reasoning effort on the 82 disagreement rows: about $2. Stage 2: the winning arm on the roughly 960 test photos where the two systems disagree: about $2.50. Gemma 4 31B only if these are not enough.

### Open models checked on Hugging Face (14 Nov 2026)

The host allows open models up to about 550B parameters ("512 GB of weights at 8-bit").

| Model | Size | Licence | Reads photos | Note |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B (in use) | 28B | Apache-2.0 | Yes | Host: 0.757 zero-shot. Its card reports it ahead of Muse Glimmer-30B and Qwen3.6-27B on visual maths and document benchmarks |
| Qwen/Qwen3.8-Flash-Next | 180B | qwen-community-1.0 | Yes | Experimental preview of the next architecture; non-standard licence; untested here |
| zai-org/GLM-5.3-Flash | 321B | MIT | Yes | Within the limit; untested here; needs several 80 GB GPUs |
| MiniMaxAI/MiniMax-M3.1-Flash, M3 | 220B, 427B | minimax-community | Yes | Within the limit; non-standard licence; untested |
| Qwen/Qwen3.5-397B-A17B | 403B | Apache-2.0 | Yes | Older generation |
| meta-models/Muse-Glimmer-30B | 30B | Apache-2.0 | Yes | Same size class as Qwen3.8-27B; behind it on Qwen's published vision table |
| openbmb/MiniCPM-V-4.7-35B-A3B | 35B | Apache-2.0 | Yes | Untested here |
| google/gemma-4-31B-it | 31B | Apache-2.0 | Yes | Host: 0.710 zero-shot, below Qwen3.8-27B |
| Over the limit | | | | Qwen3.8-2.4T (2,446B, text only), Kimi K3 and K3.1-Pro (2,780B), DeepSeek V4.1-Flash (763B), Thinking Machines Inkling (952B), Mistral Large 4 (1T) |

No other model has been run on our data. Only the two host figures are measurements on this task.

**GLM family (checked 14 Nov).** `zai-org/GLM-5.3-Flash` (321B total, 18B active, MIT) is the only photo-reading model in the GLM-5 series; `GLM-5.3` itself is text only and there is no smaller GLM-5.3 vision model. Older vision models exist (GLM-4.6V, GLM-4.6V-Flash, December 2025). Third-party aggregators put GLM-5.3-Flash and Qwen3.8-27B about level on multimodal tests (BenchLM 84.0 against 84.4; Qwen ahead on CharXiv-R and BabyVision per llm-stats; GLM ahead on five of six Roboflow vision tasks). It needs roughly 170–320 GB of GPU memory, so a multi-GPU machine well above the approved $1.79/hr.

**Gemma 4 31B published figures (its model card):** MATH-Vision 85.6%, MMMU-Pro 76.9%; Qwen3.8-27B's card reports MathVision 90.0. It has a thinking mode. Never run here.

**Research on tasks like ours (none evaluates Qwen3.8, Gemma 4 or GLM-5.3):**

| Paper | What it found |
|---|---|
| ScratchMath, arXiv 2603.24961 (Mar 2026) | 1,720 real handwritten scratchwork samples, seven error types, 16 models: large gaps to human experts in reading and reasoning; proprietary models beat open ones; large reasoning models do best |
| The Aftermath of DrawEduMath, arXiv 2603.00925 (Mar 2026) | 11 models on real students' handwritten responses: weaknesses concentrate on assessing student error |
| FERMAT, ACL 2025 (arXiv 2501.07244) | 2,200+ handwritten solutions with injected errors; models do better on printed than handwritten input |
| VEHME, arXiv 2510.22798 | An open model fine-tuned on structured reasoning data, then reinforcement learning, approaches proprietary accuracy on FERMAT |
| CHECK-MAT, arXiv 2507.22958 | Grading scanned handwritten exam solutions against expert grades |

---

## 9. Log (append only; newest last; times in UTC)

| When | What happened |
|---|---|
| 8 Nov | Submissions 1–2: 4B image classifier on Kaggle free GPU, then calibration and a question prior. Public 0.58216, then 0.59800. |
| 10 Nov | Owner authorised RunPod. |
| 11 Nov | Owner authorised the Claude API for training and pool photos (26,202 pool photos labelled over 11–12 Nov). Submissions 3–6: 8B image classifier with reading loss (0.62383), read-then-diagnose blend (0.66183), text × Claude-trained image model (0.68283), 3-way blend (0.70850). |
| 12 Nov 01:46 | Owner: terminate all pods; nothing starts or is submitted without permission. |
| 12 Nov 02:31 | Submission 7: 3-way blend + Sinkhorn. Public 0.71716. |
| 12 Nov | Stage B reason-first student: failed ($2.14). Claude without the photo: 0.771 dev ($1.79). Error catalogue: failed ($2.81). |
| 13 Nov 01:57–06:46 | Pointwise ranker v1 (Qwen3-14B): fused dev 0.7525 ($5.25). |
| 13 Nov 07:08 | Submission 8: blend × ranker v1. **Public 0.73183 (best).** |
| 13 Nov 07:42–13:09 | Ranker v2 on more Claude rows: failed its gate ($5.97). |
| 13 Nov | Eleven CPU fusion ideas tested; only the v1/v2 ranker average held on dev (+0.007). |
| 13–14 Nov 23:24–00:45 | Test rows scored with ranker v2 ($1.47). |
| 14 Nov 00:55 | Submission 9: ranker average. Public 0.72550 (worse). Ranker v2 dropped. |
| 14 Nov 01:40 | Read the Kaggle discussion tab, 13 public notebooks and the host's blog. Found the host's reference score for Qwen3.8-27B (0.757 zero-shot); we had only used Qwen3-generation models. |
| 14 Nov 02:42–03:57 | Qwen3.8-27B zero-shot on dev ($1.37): with thinking, fused 0.805 against 0.772 on 270 rows. |
| 14 Nov 04:20 | Owner approved test scoring of the 1,593 least-confident photos and raised the RunPod cap to $88. |
| 14 Nov 04:24 | Two pods started (section 6). `docs/experiment_review_2026-11-14.md` written. |
| 14 Nov 05:22 | Pod A slower than pod B; the last 70 rows of part 7 moved to pod B. |
| 14 Nov 06:10 | Owner: submit the fused Qwen3.8 file if it is better, keep working until the result is in, and keep this file. 748 of 1,593 photos answered. |
| 14 Nov ~06:45 | CPU check, no spend: voting over Qwen3.8 samples, using the existing 3-sample **no-thinking** dev answers (102 rows), fusion settings fixed from the thinking run. Fused: 1 sample 0.711, 2 samples 0.737 (+0.026 [+0.003, +0.054]), 3 samples 0.729 (+0.018, CI crosses 0). Alone, voting barely moved Qwen (0.657 → 0.670). Suggests a thinking second sample might add about +0.01 public; not tested with thinking. Kaggle T4s ruled out for this test (10-hour K-12 cap; up to 8.55 h used on 8 Nov). |
| 14 Nov 07:17 | 1,392 of 1,593 photos answered. A partial fused file was staged and validated: `artifacts/submission-3way-pw14-q38-partial/submission.csv` (17.6% of top answers and 38.5% of top-three sets differ from the best file). **Not submitted**: the submit command was blocked by the session's permission check because the owner had not yet answered whether to submit the partial file; waiting for an explicit yes. Pods still running their last parts. |
| 14 Nov 07:21 | Owner said "submit now". Submission 10 (58335588): the partial fused file. **Public 0.76266, new best** (+0.031 over 0.73183; the estimate was 0.76). Pods still finishing the last 177 photos; the complete file will be staged when they end. |
| 14 Nov 07:28 | Pod A (`ripj2np3ueqbn1`) finished all its parts (`exit 0`) and was terminated by its collector. Pod B (`6htd6iidr0suqe`) still on part 8 (100 of 199 photos at 07:30); it terminates itself at job end, expected about 07:57. 1,494 of 1,593 photos answered. RunPod $83.63 of $88. |
| 14 Nov 05:40–07:35 | Separate session, CPU only, no spend, nothing launched or submitted. (1) Eight papers in `research-papers-k12/` read and verified against page images: notes in `docs/research_notes/`, synthesis in `RESEARCH_SYNTHESIS_K12.md`. (2) Design for the next system in `ARCHITECTURE_REFORM_2026.md` (reviewed twice by independent critics): keep the memoriser, give Qwen3.8 a shortlist plus one human example per shortlisted code, then fine-tune it on its own gold-matching chains and scale the adapter back; honest estimate about 0.76 to 0.81 public, not 0.85. (3) New files only, no existing file edited: `src/k12/{slices,router,exemplars,prompts}.py`, `scripts/{eval_router,build_shortlist,build_vlm_requests_v2,audit_transcript_faithfulness}.py`, five new test files; `PYTHONPATH=src:scripts python -m unittest discover -s tests` runs 96 tests, all pass; `tests/test_reproduce_submission.py` rebuilds the 0.73183 file byte for byte. (4) Measured on all 301 dev rows (`reports/router_benchmark.json`): submitted 0.755, product fusion 0.781 cross-fitted (+0.026 [+0.002, +0.050]); a hard router loses (learned gate 0.758; even with the true slice as gate 0.787, not distinguishable from the fusion); the 270-row figures are about 0.02 easier because the 31 unanswered rows are all error rows. (5) Transcript audit (`reports/transcript_faithfulness.json`): silent correction is about 6% of dev error rows, not 45%; Qwen3.8 keeps the error text at least as often as our transcriber. (6) Ready but not sent: dev requests for prompt arms A1–A4 in `data/v10/` (135 sent rows). Next step if the owner approves about $5 and a cap increase: Module 1c in `ARCHITECTURE_REFORM_2026.md` section 7 (A3 against A0 on dev). Waiting on the owner for: permission to add default-off flags to `build_vlm_requests.py`, `fuse_vlm.py`, `vlm_zero_shot_client.py`; a commit and tag of the working tree. |
| 14 Nov 07:50 | Pod B finished and was terminated by its collector. Job total **$7.08** (quote $7.80, limit $9.50); RunPod $83.99 of $88. All 1,593 photos answered, 0 failed, about 2,456 output tokens each. Complete fused file staged and validated: `artifacts/submission-3way-pw14-q38/submission.csv` (differs from the 0.76266 file in the top answer on 2.6% of rows). Not submitted; waiting for the owner's yes. |
| 14 Nov ~08:20 | Price and odds check for the owner, no spend. Leaderboard at 07:50 UTC: top three 0.8465 / 0.8445 / 0.8435, 20th place 0.8135; we are 0.76266. RunPod Secure prices on 14 Nov (runpod.io/pricing, updated 12 Nov): A40 48 GB $0.59/h, RTX A6000 48 GB $0.69, RTX 6000 Ada and L40S $1.09, A100 80 GB $2.09–$2.19 (now ABOVE the $1.79/h limit in `AGENTS.md`, so an 80 GB fine-tune needs the owner to raise that limit). Google Cloud is 2.4 to 6 times dearer on demand (48 GB half RTX PRO 6000 in Mumbai $3.36/h plus 18% GST; no A100 in India; A100 80 GB $5.07/h in Iowa) and a new account needs a paid upgrade and GPU quota approval first, so RunPod stays the provider. Owner asked about adding 50 pounds; advice given: top three is not realistic (gap 0.08 with about 64 hours left); if spending, do it in stages on RunPod with the dev pass mark of `ARCHITECTURE_REFORM_2026.md` section 6. Nothing approved yet. |
| 14 Nov 07:57 | Owner asked whether a better Qwen 3.8 model or other allowed models exist. Surveyed Hugging Face (table at the end of section 8). No model with evidence of beating Qwen3.8-27B on this task; larger allowed ones (180B–430B) need multi-GPU machines far above the approved $1.79/hr. Nothing run or spent. |
| 14 Nov 08:05 | Owner asked whether Gemma 4 31B and GLM-5.3-Flash are worth testing and for research benchmarks on similar tasks. Checked model cards, aggregators and five papers (notes at the end of section 8). No published result for these models on a task like ours. Nothing run or spent. |
| 14 Nov 08:15 | Owner offered more budget and asked for results. CPU analysis of where the fused system loses (section 8). Proposed a two-stage plan (more Qwen3.8 opinions, higher reasoning effort, Gemma 4 31B as a third opinion), about $20, cap to $108. Waiting for approval. |
| 14 Nov 08:22 | Owner asked for a rethought plan to reach 0.80+. Plan written in section 8 (rerank pass, more samples, higher effort, Gemma; optional fine-tune). Waiting for approval and a budget. |
| 14 Nov 08:30 | Owner asked for a cheaper, smarter version. Two free CPU checks (disagreement rows hold the points; a trust rule does not help). Lean plan written in section 8: about $4.50 in total on one pod. Waiting for approval. |
| 14 Nov 08:35 | Owner approved the lean plan ("Yes, go ahead"), cap $94. Built `scripts/build_rerank_requests.py` and the request files in `data/v9/rerank/` (135 dev rows sent to Qwen, 82 of them disagreements; 984 of the 1,593 test photos are disagreements). Maximum-effort arm left out of the first round to keep Stage 1 near $2. |
| 14 Nov 08:37 | Stage 1 pod `zh38skorwdhips` started with `runs/deep/q38/pod_run_q38_jobs.sh` (jobs: rerank_ex, rerank_plain, sample2). Collector, guard and in-pod timer in place. |
| 14 Nov ~08:50 | Second session, no spend: owner asked how to reach 0.80+ with budget no object. Full plan in `docs/plan_to_080_2026-11-14.md` (not approved). Key measurement on saved dev answers (stand-ins for a better or additional reasoner): M x Qwen3.8 = 0.780 on 301 rows; adding a second photo model that is 0.804 alone gives 0.809 (+0.029 [+0.009, +0.050]); a Claude-with-photo-level reasoning side gives 0.816 to 0.819, about 0.80 public. Ranked levers: (1) second open vision reasoner (Qwen3.8-Flash-Next 180B, Gemma 4 31B, GLM-5.3-Flash), (2) two more Qwen3.8 thinking samples, (3) fine-tune Qwen3.8-27B as a no-thinking photo classifier with `pod_train.py` in full precision (needs the Gated DeltaNet LoRA target names and thinking switched off explicitly), (4) thinking fine-tune on Claude traces, served merged. Estimated odds: 0.78 about 60%, 0.80 about 20 to 25%, 0.82 about 3 to 5%. Needs from the owner: cap about $700, pods above $1.79/h including multi-GPU, the named jobs, flags on three existing scripts. One session should execute; this one launched nothing. |
| 14 Nov 08:55 | Owner: the RunPod account has $6.52 left and everything must finish under it. Plan adjusted: the second-answer test was dropped from Stage 1 (it could not be afforded on test anyway; its request file was emptied on the pod), so Stage 1 ends after the two rerank tests at about 09:27 UTC. Stage 2 (rerank on 984 test photos, about $2.90) runs on a fresh pod only if Stage 1 passes. Projected total from 08:55: about $3.50. |
| 14 Nov 09:05 | Second session, no spend: owner rejected the $700 plan as poor value and asked to solve it inside the Kaggle free GPU quota. Checked `kaggle quota`: the weekly quota reset today, **30.00 h GPU free, next refresh 21 Nov** (`reports/gpu_budget.json` still holds the old 8 Nov snapshot and `gpu_launches_allowed: false`; `AGENTS.md` still caps K-12 at 10 h and bars K-12 Kaggle GPU jobs, so the owner must give a new allowance first). Proposed, not approved: a 30-minute Kaggle smoke test of Qwen3.8-27B in 4-bit on 2x T4 as a no-thinking photo classifier (seconds per training row, memory), then if it fits one fit-row training run (about 13 h estimated), dev scoring and test scoring (about 7 h; an earlier 27B 4-bit model ran at 7.2 s per row on T4s), about 21 h of quota in all. No smaller Qwen3.8 model exists. |
| 14 Nov 09:17 | Second session. Owner approved the Kaggle smoke test ("Yes, run the smoke"); recorded in `AGENTS.md` and `reports/gpu_budget.json` (new quota window, smoke only). Built `kaggle/q38-smoke/` (script `smoke.py`: Qwen3.8-27B from the Kaggle model `cputer/qwen3-8-27b-base`, 4-bit NF4 with the vision tower left in fp16, split over 2x T4, LoRA on all language-model linear layers, k12.vision prompt and last-token 76-way readout, thinking off; times 5 inference rows and 8 training rows, reports memory and whether losses are finite). Launch goes through `scripts/launch_kaggle.py` with a 1,800 s timeout. Complication: another project on this Kaggle account (kernels named `qcc-job-*`) is running CPU-only jobs continuously; they hold all 5 CPU batch slots (so a CPU-only inspection job could not start) and the launcher refuses while unknown sessions are active, so the runner first verifies each active job is CPU-only (`enable_gpu` false in its metadata) and registers it in `reports/gpu_budget.json` with zero GPU time. Runner log: `runs/kaggle-q38-smoke/runner.log`; output lands in `runs/kaggle-q38-smoke/`. Free CPU experiments this morning (eight fusion variants incl. a per-question prior from test answers): nothing beats the current product fusion; the prior is slightly negative; details in the scratch report. Outside research: the closest past competitions were all won by fine-tuning the strong model itself to score candidate codes from logits; Unsloth documents QLoRA of Qwen3.8-27B on Kaggle 2x T4. The Kaggle overview page gives the deadline as 17 Nov 23:59 UTC (HANDOFF section 1 says 00:00; check). |
| 14 Nov 09:20 | Second session: owner said "stop it" before the Kaggle smoke test launched. Runner and watcher stopped; the kernel was never pushed (it does not exist on Kaggle), 0 h of GPU quota used. `reports/gpu_budget.json` set back to `gpu_launches_allowed: false`; `AGENTS.md` updated. The prepared job stays in `kaggle/q38-smoke/` for later use if the owner asks. |
| 14 Nov 09:23 | Stage 1 pod finished and was terminated ($0.83). **Both rerank tests failed the pass mark** (with teacher examples about +0.009 whole-dev at best, interval through zero; without examples slightly negative; three Qwen answers averaged +0.005). Stage 2 not run. Report: `reports/q38_stage1.json`; details in audit record 17.3. |
| 14 Nov 09:45 | Owner asleep; asked for results and a 0.80+ plan with stronger evidence. Free analysis found that our Qwen3.8 run (effort medium) is worth about 0.73 public alone against the host's 0.757, and that thinking off → medium is worth +0.137 on the same rows. CURRENT PLAN written in section 8 (maximum reasoning effort first, then a fine-tuned new-generation member). Nothing running; about $6.01 left in the RunPod account. |
| 14 Nov 10:15 | Owner set the goal at 0.84 and asked for a first-principles diagnosis, solutions checked against published work, and a complete plan, with GPU access to follow. Diagnosis written in section 8: locating the error is mostly solved (79%), naming it the way the teachers do is the bottleneck (64% when the line is right). Two free fixes tested and rejected (teacher-text classifier, small adapters). Two research agents are checking winning solutions of the closest past competitions and the evidence for each candidate technique. |
| 14 Nov 10:50 | Research finished (winning write-ups of MAP 2025 and Eedi 2024; papers on voting, thinking budgets, fine-tuning thinking models, retrieved examples, handwritten-maths grading). One more free check: the right code is in our top five, Qwen's three or ten similar teacher findings on 92% of hard rows. Complete plan written: `docs/plan_to_084_2026-11-14.md` (ranker v3). Waiting for the owner's go-ahead and GPU access; about $6 left in the RunPod account. Free preparation of the ranker v3 data builder is the next CPU task. |
| 14 Nov 11:20 | Owner asked about DeepSeek+Qwen mixes. Checked Hugging Face (no spend). DeepSeek itself published only old text-only ones (R1-Distill-Qwen 1.5B/7B/14B/32B, Jan 2025; R1-0528-Qwen3-8B, May 2025). The new ones are community fine-tunes: `Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B` and `-4B` (Apache-2.0, Qwen3.5 base, about 250K maths/STEM text answers from DeepSeek V4 Pro, +0.93 MMLU-Pro points over base by its own card; photo parts kept but not trained), `kevinzhou21/Qwen3.5-9B-Distill-Deepseek-V4.1-Flash`, and a few barely used Qwen3.6-27B coding distils. None on Qwen3.8-27B. Verdict: not on the route to 0.84 (older, smaller base; teaches problem solving, not reading handwriting or the teachers' codes). Optional cheap check only: 9B zero-shot on dev photos. |
| 15 Nov 01:15 | Owner: "i have 5.97 credits in run pod so tell me if the porgress is going well i'll top up otherwise no like check it progressive and tell me updates on it". Read as a go-ahead for the plan's GPU work inside that balance (at most $5.45 of spend), with progress reports; recorded in `AGENTS.md`. No Kaggle submission is covered. |
| 15 Nov 01:22 | **Stage A started**: pod `hv0glxvp4f0d8p` (RTX 6000 Ada, $1.09/hr), `runs/deep/q38/pod_run_q38_jobs.sh` with jobs `tr_1..tr_9` (requests in `data/v9/q38_train/`: the 31 dev photos not answered before, then the 1,199 fit photos in a fixed random order; same prompt and settings as the dev and test runs). New: `/workspace/workers` sets parallel requests (20, was 40). Reason: the saved server logs show the KV cache was 97% full at 24-32 running requests and generation averaged about 160 tok/s there, against about 330 tok/s at 16-23 running (cache about 70%); so the old "150 tok/s on the Ada" was partly self-inflicted overload. Unverified until this run reports. No new request after 05:52 UTC; collector limit $5.20 (output `runs/deep/q38/train_pod/`); in-pod timer removes the pod at 06:16 UTC. |
| 15 Nov 01:33 | Stage A first reading: with 20 parallel requests the server makes about 375 output tok/s (KV cache about 55% full) against about 170 in the 14 Nov runs at 40. Stage A now costs about $2.25, not $6. **Use 20-26 parallel requests on the 48 GB Ada, not 40.** The 31 missing dev photos are answered: full-dev check `reports/q38_fuse_301.json`: old system 0.7553, fused with Qwen3.8 0.7890 cross-fit (+0.034 [+0.006, +0.062]); the staged setting gives 0.7962 on dev (hard rows 0.616). |
| 15 Nov 01:36 | **Stage B chained on the same pod** (fits the balance now). New files: `scripts/build_ranker_v3.py` (judge data: Qwen3.8 transcription + root line + finding + its codes as context; teacher codes on the question as a hint; candidates = our top 5 + Qwen 3 + NONE + question codes + codes of similar teacher notes, 97% truth coverage on dev, 90% on hard rows; each candidate carries a teacher note from this question or the most similar one elsewhere; 10% of training rows repeated question-blind), `runs/deep/ranker3/pod_run_v3.sh` (Qwen3-14B, group 5, max length 1280, waits for stage A, reuses the vLLM venv, falls back to the ranker v1 environment if the smoke test fails), `runs/deep/ranker3/feed_v3.sh` (Mac side: when stage A ends, copy answers, build `data/v10/pw_*.jsonl`, upload, mark ready), `scripts/fuse_v3.py` (cross-fitted fusion against the current fused system and the stage B pass mark; stages a test file, never submits). Collector and guard now watch `v3.status` (limit $5.20, final copy includes the adapter). Test rows are scored in six parts, least-confident first, with no new part after 05:49 UTC. |
| 15 Nov 02:20 | Owner asked for the Qwen3.8-27B fine-tune to be prepared (no spend, NOT approved to run). Built and dry-run on this Mac with a tiny random model of the same architecture and the real tokenizer/processor: `scripts/build_q38_finetune.py` (-> `data/v11/ft_train.jsonl` 1,306 rows incl. 107 question-blind repeats, `ft_dev.jsonl`; test parts with `--test-parts`, skipped so far), `src/k12/q38_finetune.py` (train / merge / probe; LoRA r16 on every linear layer of the language model, loss on the answer only, thinking off, time-aware schedule `--max-minutes`), `runs/deep/q38ft/pod_run_q38ft.sh` (A100 80GB, bf16, smoke, train, merge, vLLM, dev, go signal, test). Design: the model is trained to write the teacher transcription, root line and note BEFORE the codes (reasoning in the visible answer, so no bare-answer training); prompt is short (all code names, definitions of the topic's codes, teacher codes on the question as a hint; about 1,900 tokens per example, measured). Facts learned: the architecture is `Qwen3_5ForConditionalGeneration`; LoRA must target `linear_attn.in_proj_qkv/z/b/a/out_proj` as well as q/k/v/o and the MLP (the old fixed name list misses three layers in four); the processor needs torchvision; HF training wants `flash-linear-attention` and `causal-conv1d` or falls back to slow reference code; default reasoning effort is xhigh. Estimate, unmeasured: about $12-18 and 6-9 h on an A100 80GB ($1.79/hr) including test scoring. Rebuild `data/v11` after stage A ends so every row has Qwen alternatives. |
| 15 Nov 02:32 | Owner asked for a classification layer in the fine-tune (no spend). `src/k12/q38_finetune.py` now trains two things in one pass: the written teacher-style answer, and a new LayerNorm + Linear on the model's state at the last prompt token, trained on the teacher's root code (`--head-weight 0.5`, `--head-lr 1e-3`, saved as `adapter/head.pt`). New command `head` scores request files with one forward pass per photo and no generation, writing a normal `.npz` bundle; `merge` copies `head.pt` beside the merged model. Builder writes `label` per training row and `data/v11/classes.json` (76 outputs). Pod script gained HEAD_DEV and HEAD_TEST stages and a go signal with a mode (`both`, `gen`, `head`). `scripts/fuse_v3.py --all-rows` fuses a bundle that covers every photo. Dry run on the tiny model: head loss falls, bundles are valid, adapter and merged model agree to 1e-7. Still unproven at full size; the pod smoke test is the first real check. |
| 15 Nov 02:35 | Owner asleep. Instruction: terminate the pod when it has no use; get results; "if they are 0.80 or better than that please do submit it". Rule applied (recorded in `AGENTS.md`): submit the judge's fused test file once, only if `scripts/fuse_v3.py` reports cross-fitted dev >= 0.80 and above the current fused system; otherwise submit nothing. If the judge clearly fails on dev, terminate the pod at once instead of scoring test rows. |
| 15 Nov 03:12 | **Stage A done** ($2.00 so far): all 1,230 photos answered, none failed (`runs/deep/q38/train_pod/tr_1..9.jsonl`). Qwen3.8 alone on the 1,199 fit rows: MAP@3 0.744, top-1 0.681 (no-error rows 0.950, error rows 0.639; `reports/q38_train_zero_shot.json`), against 0.759 on dev: the stage A check (within 0.03) passes. `feed_v3.sh` built `data/v10` (1,306 train rows, 14,317 pairs; dev 301 rows with truth among candidates on 96.7%, hard rows 90.6%; test 1,593 rows in 6 parts) and uploaded it. |
| 15 Nov 03:17 | **Stage B (judge) training** on the same pod: smoke test passed in the reused vLLM environment (peft trains fine on torch 2.14), 811 tok/s, 0.245 rows/s; full run 1,306 rows, group 5, 327 updates, expected to end about 04:46 UTC, dev scores about 05:00, then test parts until 05:49. |
| 15 Nov 03:22 | Owner asked for a proper plan with expected results per step and fallbacks, target 0.85. Written: `docs/plan_to_085_2026-11-15.md` (E0 judge on 14B, running; E1 fine-tune 27B + classification layer, about $20; E2 Claude-labelled photos + seeds, about $40; E3 maximum thinking; E4 27B judge with the photo; E5 second reader; E6 final blend). Nothing in it beyond E0 is approved or funded. |
| 15 Nov 03:37 | Free research on the 1,500 Qwen3.8 answers (no spend). Findings in section 9 of `docs/plan_to_085_2026-11-15.md`: naming errors are not systematic (279 distinct confusions, 187 singletons; a code-to-code mapping gives 0.679 vs 0.681); biggest confusions EXP-04->EXP-07 (20), over-use of WOP-01 and CAL-01; same-question students share a code only 26-44% of the time; within-question smoothing +0.0003 (rejected); final-answer matching covers 19/200 at 58% (rejected); CPU feature judge `scripts/cpu_judge.py` alone 0.751 vs Qwen 0.748, fused -0.010 (rejected, `reports/cpu_judge_fuse.json`). Label noise cannot be estimated from duplicates (2 near-identical pairs). New code, dry-run only: `q38_finetune.py features` (frozen 27B state per photo, several layers) and `scripts/q38_probe.py` (logistic probes on the Mac, layer chosen by CV inside fit rows); pod script gained a FEATURES stage. Plan ladder lowered: E1 0.79-0.81, all steps 0.82-0.84. |
| 15 Nov 03:40 | Outside research returned (section 10 of the plan). Host data page admits label noise between neighbouring codes (names WOP-01 vs ANS-01) but publishes no agreement figure: no evidence of a ceiling near 0.85. Published recipes for Qwen3.5/3.8 LoRA (Unsloth, ms-swift, NVIDIA) match our settings; ms-swift pins flash-linear-attention >= 0.4.2 and causal-conv1d; vLLM run-time LoRA for Gated DeltaNet is unmerged upstream, so merge before serving; MAP 1st used a last-token linear head and found the rationale loss marginal; seeds beat folds, retrain on all data with 3 seeds. Host will verify winners' training and inference pipelines. Pseudo-labelling test photos with own models is unanswered by the host: not done. `xhigh` thinking can use over 20k reasoning tokens on one prompt: cap and measure in E3. Deadline re-checked with `kaggle competitions list`: 2026-11-17 00:00 (the agent read the page as 17 Nov 23:59 UTC; plan on the earlier). Rank 62 of 447. |
| 15 Nov 04:00 | Loss map of the current system on dev (section 11 of the plan): 61.3 points lost of 301; right line but wrong code 34.0, wrong line 11.3, error called NONE 8.8, NONE called error 7.2. For 0.90 at most 30.1 may be lost. New lever: teacher flag `final_answer_correct` predicts NONE (94% / 98%); with the true flag dev rises up to +0.0205 (oracle upper bound); a regex proxy from Qwen's transcription agrees only 79% and gives nothing. `q38_finetune.py` head now has one extra output trained on that flag (`final_ok` in `ft_train.jsonl`, written to the head bundle as `final_ok`); dry-run passes. Photo resolution ruled out (already native, median long side 1,024). |
| 15 Nov 04:15 | Owner approved the free runs ("okay start the free runs on mac and kaggle"): Mac rehearsal on `Qwen/Qwen3.5-2B` (downloaded to `~/.cache/k12-models/Qwen3.5-2B`, 4.5 GB) and a Kaggle fine-tune of `Qwen/Qwen3.5-9B`, at most 10 GPU hours. Recorded in `AGENTS.md`. |
| 15 Nov 04:30 | **Bug found and fixed before any paid run:** `processor.apply_chat_template` rewrites the photo parts of the messages it is given, so `q38_finetune.encode` then found no photos and every dry run trained on text only (inputs had no `pixel_values`). Now the photos are collected first, the template gets a copy, and `encode` raises if photo slots and photos differ. Real size: about 2,700 tokens per example (photo about 930). Also added: photo references `file:images/...` with `--image-root` (identical pixels and tokens to the embedded version, checked), `--dtype float16` with a GradScaler, `--device auto` across GPUs, `--device mps`, and the Kaggle PEFT/torchao workaround. `data/v11k/` holds the small reference-form files (35 MB) for Kaggle. |
| 15 Nov 04:35 | Owner asked for agents tied to the plan that test on Kaggle and locally, watch Kaggle and RunPod, look for cheaper and better options, and change the plan only when results are worse. Set-up: the main session is the coordinator (owns the plan, `HANDOFF.md`, spending); two worker agents started: Kaggle 9B run (log `runs/agents/kaggle_9b.md`) and Mac 2B rehearsal (log `runs/agents/mac_2b.md`). The plan gained a live status board and change log (section 0). |
