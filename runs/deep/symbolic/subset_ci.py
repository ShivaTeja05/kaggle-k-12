"""Paired question-grouped bootstrap CIs for selected oracle contrasts, overall and on unseen-pair rows."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import boot_ci
import value_oracle as vo
out = {}
for reg in ['fit_only', 'fit+unl']:
    p = HERE / f'value_oracle_pts_{reg}.npz'
    if not p.exists():
        continue
    Z = np.load(p)
    for a, b in [('topic+gold_rootline', 'topic+transcript'), ('Q+transcript+gold_rootline', 'Q+transcript'),
                 ('topic+sym_line', 'topic+transcript'), ('Q+transcript+sym_line+symtok', 'Q+transcript'),
                 ('gold_rootline_only', 'transcript_only'), ('sym_line_only', 'transcript_only')]:
        for t in ['gold_tx', 'machine_tx']:
            ka, kb = f'{reg}__{a}__{t}', f'{reg}__{b}__{t}'
            if ka not in Z or kb not in Z:
                continue
            for sub in ['all', 'unseen_pair', 'unseen_q', 'seen_pair']:
                m = np.ones(len(vo.cat), bool) if sub == 'all' else vo.cat == sub
                d, ci = boot_ci(Z[ka][m], Z[kb][m], vo.qg[m])
                out[f'{reg}|{a} vs {b}|{t}|{sub}'] = dict(a=round(float(Z[ka][m].mean()), 3), b=round(float(Z[kb][m].mean()), 3), delta=round(d, 3), ci95=[round(x, 3) for x in ci])
json.dump(out, open(HERE / 'oracle_contrasts.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v)
