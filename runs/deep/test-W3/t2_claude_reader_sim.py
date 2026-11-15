"""W3 test 2: simulate a reader at CLAUDE's fidelity (the A1 teacher) rather than gold, including rows where Claude reads
the root span WORSE than our machine reader. Per error row, f = fraction of the machine->gold root-span edit distance that
Claude's transcript closes: f = (m_dist - c_dist) / m_dist (f in [-1, 1]; m_dist=0 & c_dist>0 -> f=-1 i.e. Claude regresses).
The 32B member's log-probs are moved l = l_machine + f * (l_gold - l_machine) (linear interpolation in log space: an
ASSUMPTION; f<0 extrapolates away from gold as a crude stand-in for a worse read). NONE rows unchanged (no root span;
Claude page CER is worse than machine on dev). Student-reader variants scale f by k (k = share of teacher fidelity
the distilled transcriber reaches). Fixed weights/lambda (nothing fitted on dev); question-grouped bootstrap vs 0.737."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W3')
import importlib.util
spec = importlib.util.spec_from_file_location('t1', '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W3/t1_reader_ceiling.py')
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
from common import boot_ci, ROOT
L, md, cd, err, y, qid, ref = t1.L, t1.md, t1.cd, t1.err, t1.y, t1.qid, t1.ref
f = np.zeros(len(y))
for i in np.flatnonzero(err):
    if np.isnan(md[i]) or np.isnan(cd[i]): continue
    if md[i] > 0: f[i] = np.clip((md[i] - cd[i]) / md[i], -1, 1)
    elif cd[i] > 0: f[i] = -1.0
out = dict(f_stats=dict(n_pos=int((f > 0).sum()), n_neg=int((f < 0).sum()), n_zero_err=int(((f == 0) & err).sum()), mean_f_err=round(float(f[err].mean()), 3)))
def run(k, all_text=False, pos_only=False):
    ff = k * (np.clip(f, 0, None) if pos_only else f)
    l32 = L['m32'] + ff[:, None] * (L['g32'] - L['m32'])
    l8 = L['m8'] + (ff[:, None] * (L['s100g'] - L['s100m']) if all_text else 0)
    s = t1.score(t1.sinkhorn(t1.blendP(l32, l8))); d, c = boot_ci(s, ref, qid)
    return dict(map3=round(float(s.mean()), 4), delta=round(d, 4), ci95=[round(x, 4) for x in c], up=int((s > ref).sum()), down=int((s < ref).sum()))
res = {}
for k in (1.0, .75, .5):
    for at in (False, True):
        for po in (False, True):
            key = f'k={k} {"allText(s100 ESTIMATE)" if at else "32B only"}{" gains-only(no regressions)" if po else ""}'
            res[key] = run(k, at, po); print(f'{key:55s}', res[key])
out['sim'] = res
json.dump(out, open(ROOT/'runs/deep/test-W3/t2_claude_reader_sim.json', 'w'), indent=1); print(out['f_stats'])
