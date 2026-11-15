"""Does the gold ROOT misconception sit on the FIRST erroneous step (ProcessBench/PRM 'first error' target)?
Aggregate over all 1,500 train rows' step-level annotations."""
import json, re, pandas as pd
from common import load, ROOT
tr, tax, classes, folds, dev, fit = load()
def ln(s):
    m = re.match(r'L(\d+)', str(s or '')); return int(m.group(1)) if m else None
n_err = n_multi = root_first = root_first_major = n_multi_major = 0; root_is_major = 0; n_spans = []
root_line_none = 0
for ms, code in zip(tr.misconceptions, tr.root_code):
    if code == 'NONE': continue
    try: L = json.loads(ms)
    except Exception: continue
    n_err += 1; n_spans.append(len(L))
    roots = [m for m in L if m.get('root')]
    if not roots: continue
    r = roots[0]; rl = ln(r.get('step'))
    if rl is None: root_line_none += 1; continue
    lines = [ln(m.get('step')) for m in L if ln(m.get('step')) is not None]
    major = [ln(m.get('step')) for m in L if m.get('severity') == 'major' and ln(m.get('step')) is not None]
    root_is_major += int(r.get('severity') == 'major')
    if len(L) >= 2:
        n_multi += 1; root_first += int(rl == min(lines))
    if len(major) >= 2:
        n_multi_major += 1; root_first_major += int(rl == min(major))
res = {'error_rows': n_err, 'mean_spans_per_error_row': round(sum(n_spans) / len(n_spans), 2),
       'share_error_rows_with_2plus_spans': round(n_multi / n_err, 3),
       'root_is_earliest_span_given_2plus': round(root_first / max(n_multi, 1), 3),
       'root_is_earliest_major_given_2plus_major': [round(root_first_major / max(n_multi_major, 1), 3), n_multi_major],
       'root_severity_major_share': round(root_is_major / n_err, 3), 'root_without_line': root_line_none}
json.dump(res, open(ROOT/'runs/deep/literature/p3_root_vs_first_error.json', 'w'), indent=1); print(json.dumps(res, indent=1))
