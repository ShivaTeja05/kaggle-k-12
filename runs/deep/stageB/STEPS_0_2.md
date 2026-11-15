# Stage B, steps 0-2 (CPU, $0), 12 Nov 2026

## Step 0: reference (`reference.py` → `reference.json`)
- Cross-fitted blend+Sinkhorn dev MAP@3 **0.7337** (gate 0.7337 ± 0.001: pass). In-sample λ=.5: 0.737; raw blend 0.7182.
- Slices (cross-fitted): NONE 0.926 (101), seen pair 0.856 (94), new pair 0.443 (84), unseen question 0.439 (22), new/unseen 0.442 (106).
- Label-free test Simpson of the blend top-1: 0.421 (the plan quoted 0.423).
- Fallback final: `artifacts/submission-3way-sinkhorn/submission.csv`, public 0.71716.

## Step 1: v5b data (`build_v5b.py` → `data/`)
9,705 train rows (6,107 Claude + 2,398 gold main; 709 + 491 contrast), 301 dev, 3,498 test.

| Change | Result |
|---|---|
| Verdict phrase stripped | 0 left in train/dev/test transcripts |
| Code IDs in WHY | 0 (was 1,847 rows incl. NONE); "matches X: meaning" keeps the meaning, other ID sentences dropped or reworded |
| Gold non-root codes as distractors | yes |
| Confusion-neighbour distractors | top 8 (Claude co-rank + 3× gold co-annotation, fit only), ≤3 generic; test co-listing as fallback |
| Contrast rows | 1,200 (2–3 candidates), generic/specific convention stated in WHY |
| Truth-example masking | p = 0: after weighting, the visible-example ratio is already 0.53 (in band) |
| Weights | K=1.5 for unseen pairs, no pair-count factor, label shift^0.5. The plan's K=2 with 1/√pair overshot (LOO ratio 0.86, ANS-01 −46%), so the scan picked the closest recipe that passes |
| Smart candidate fill | **dropped**: it lowered dev new/unseen recall@10 (0.849 vs 0.859) |

## Step 2: gates (`gate_v5b.py` → `gate_v5b.json`): all pass
- Confusion coverage 0.774 (≥ 0.70).
- Unseen likelihood ratios: visible 0.53, LOO 0.53. Dev ≈ 0.49; v5 had 0.99 and 0.37.
- Kish ESS 7,431 (≥ 4,000).
- ANS-01 share ×0.85 (within ±20%).
- Claude NONE share 0.338.
- p99 tokens 1,120 (≤ 1,600).
- ROOT_LINE references OK.
- Dev recall unchanged (0.950).
- Hygiene clean, no dev leakage.

**Known leftover:** 137 error rows have ROOT_LINE "-", as in v5 (source rows with no root line).

## CPU smoke (tiny Qwen2, `cpu_smoke/`)
- Trains on main, contrast and gold rows, and predicts in both orders.
- `fuse_candidate.py --subset` reads the bundle.
- The interim-abort check works: it correctly says no-go on the random tiny model.

## Prepared, not launched
`pod_run_v5b.sh` covers step 3 and step 4:
- **Step 3 (smoke):** 400 rows plus 50 dev rows. It continues only if parse rate ≥ .95, ≥ 1.5 rows/s, loss falling and epoch cost < $4.
- **Step 4 (full run):** full epoch with an automatic interim abort at 50% (new/unseen < .35 or parse < .9), then dev and test predict.
- **Estimate:** about 4.5 h on an RTX 6000 Ada at $0.99/h ≈ $4.5–5.

Needs your explicit approval.
