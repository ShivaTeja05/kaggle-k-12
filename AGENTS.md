User constraints for this competition:

- On 8 November 2026 at approximately 16:28 UTC, after a shared-quota conflict was discovered, the user chose: "Keep Gemma running; continue K-12 on CPU". Do not launch further K-12 GPU jobs unless the user changes that instruction. Do not stop the unrelated Gemma job.
- Use Kaggle free GPUs only. No paid compute or paid model APIs.
- Hard cap: do not consume more than 10 hours of the account's 30-hour weekly GPU quota. Count quota already used; preserve at least 20 hours. The user's instruction was given on 8 November 2026.
- On 10 November 2026 the user also required at least 4 hours of unused Kaggle GPU quota. Check this reserve after accounting for the worst case of active and proposed jobs, including overhead and margin.
- A weekly quota reset does not renew this task's 10-hour allowance. Reconcile prior usage before any run after the quota window changes.
- Before every new GPU run, use `scripts/launch_kaggle.py`, which checks `reports/gpu_budget.json`, current Kaggle quota, and reservations for all active jobs. Do not bypass that check with a direct GPU `kaggle kernels push`.
- Keep explicit job timeouts. Reserve time for active jobs and a safety margin; do not schedule a run that could exceed the cap if every active job uses its remaining timeout.
- CPU-only dependency preparation is allowed and does not consume GPU quota.
- Target approximately 0.82 public MAP@3 as the immediate milestone. Do not claim a score from a smoke test, in-sample predictions or a prepared/unlaunched job.
- Final competition inference must run offline. Keep competition data and student images private, and do not hand-label test records.
