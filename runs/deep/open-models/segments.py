"""Open-model evidence table: per-segment MAP@3 of every measured open model vs blend and Claude on dev fold 0.
CPU only. Segments follow runs/arch/teacher-gap/bin_ci.py (seen-pair / unseen-pair / unseen-question errors, NONE).
Also: question-shortcut rate (top-1 is one of the question's fit codes on unseen-pair rows), top-1 agreement with Claude,
recall@3 rescue (truth in model top-3 but not blend top-3), and question-grouped bootstrap CIs vs blend."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path[:0] = ['src', 'scripts', 'runs/arch/label-structure']
from common import load, blend, boot_ci
from k12.core import aligned_bundle, topk, row_scores, normalize
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy(); q = dev.question_id.to_numpy()
qc = fit.groupby('question_id').root_code.apply(set).to_dict()
bins = np.where(truth == 'NONE', 'NONE', np.where(~pd.Series(q).isin(qc).to_numpy(), 'q_unseen',
        np.where([t in qc.get(a, set()) for t, a in zip(truth, q)], 'pair_seen', 'pair_unseen')))
P = {
 'blend(.35/.30/.35)': blend(ids, classes),
 'ft Qwen3-32B QLoRA tx': aligned_bundle('runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes),
 'ft Qwen3-8B LoRA tx e3': aligned_bundle('runs/pod/tx/text_tx_dev_e3.npz', ids, classes),
 'ft Qwen3-VL-8B distill': aligned_bundle('runs/pod/dev-distill/validation_epoch_2.npz', ids, classes),
 'ft Qwen3-8B scale_100 gold': aligned_bundle('runs/pod/scaling/scale_100/validation_gold_epoch_1.npz', ids, classes),
}
for name, d in [('zs Qwen3-32B think', 'zeroshot32'), ('fs4 Qwen3-32B think', 'fewshot32'),
                ('zs Qwen3-235B-A22B think', 'zeroshot235'), ('fs4 Qwen3-235B-A22B think', 'fewshot235')]:
    P[name] = aligned_bundle(f'runs/pod/probe235/{d}/validation.npz', ids, classes)
cl, _ = claude_scores('runs/claude/dev_opus.jsonl', ids, classes); P['Claude Opus 5.5 zs (teacher)'] = cl

ref = row_scores(truth, topk(P['blend(.35/.30/.35)'], classes))
ref_top3 = [r.split()[:3] if isinstance(r, str) else list(r)[:3] for r in topk(P['blend(.35/.30/.35)'], classes)]
cl_top1 = [ (r.split() if isinstance(r, str) else list(r))[0] for r in topk(cl, classes)]
out = {}
for name, p in P.items():
    rk = topk(p, classes); s = row_scores(truth, rk)
    lst = [r.split() if isinstance(r, str) else list(r) for r in rk]
    row = {'all': round(float(s.mean()), 4)}
    for b in ['pair_seen', 'pair_unseen', 'q_unseen', 'NONE']:
        m = bins == b; row[b] = round(float(s[m].mean()), 3); row['n_' + b] = int(m.sum())
    m = bins == 'pair_unseen'
    row['pair_unseen_top1_is_question_fit_code'] = round(float(np.mean([l[0] in qc.get(a, set()) for l, a, k in zip(lst, q, m) if k])), 3)
    row['pair_unseen_top1_agree_claude'] = round(float(np.mean([l[0] == c for l, c, k in zip(lst, cl_top1, m) if k])), 3)
    err = bins != 'NONE'
    row['rescue_rows_truth_in_top3_not_in_blend_top3'] = int(sum((t in l[:3]) and (t not in r) for t, l, r, k in zip(truth, lst, ref_top3, err) if k))
    row['delta_vs_blend_all'] = boot_ci(s, ref, q)
    mm = (bins == 'pair_unseen') | (bins == 'q_unseen')
    row['delta_vs_blend_unseen106'] = boot_ci(s[mm], ref[mm], q[mm])
    out[name] = row
# oracle complementarity: max(blend, model) per row on the 106 unseen rows -> upper bound of a perfect router
mm = (bins == 'pair_unseen') | (bins == 'q_unseen')
for name, p in P.items():
    s = row_scores(truth, topk(p, classes))
    out[name]['oracle_max_with_blend_unseen106'] = round(float(np.maximum(s, ref)[mm].mean()), 3)
Path('runs/deep/open-models/segments.json').write_text(json.dumps(out, indent=1))
df = pd.DataFrame(out).T[['all', 'pair_seen', 'pair_unseen', 'q_unseen', 'NONE', 'pair_unseen_top1_is_question_fit_code',
     'pair_unseen_top1_agree_claude', 'rescue_rows_truth_in_top3_not_in_blend_top3', 'oracle_max_with_blend_unseen106']]
pd.set_option('display.width', 250); print(df.to_string())
print({k: (v['delta_vs_blend_all'], v['delta_vs_blend_unseen106']) for k, v in out.items()})
print('bins', pd.Series(bins).value_counts().to_dict())
