"""T6. Put REAL offline model outputs through the proposal's gate and its fusion, to see whether 'passing the gate'
(masked single model >= 0.715 and unseen-pair >= 0.48) implies a fused gain over blend+Sinkhorn when the student's
errors have the real correlation structure (not gate_sim's random mixing).
s100 scored on GOLD transcripts is not deployable (gold transcripts are not available on test); it is used only as a
real 8B output that happens to sit at the gate thresholds."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; mf = qfolds(qid)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
LAMS = [0, .25, .5, .75]; W = [0, .2, .35, .5, .65, .8]
bs = {l: row_scores(y, topk(sinkhorn(B, pi, l) if l else B, classes)) for l in LAMS}
bsink_cf, _ = crossfit(bs, mf)
cands = {'s100_tx': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
         's100_gold_tx(not deployable)': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz',
         'q32_gold_tx(not deployable)': 'runs/pod/text32/validation_gold_epoch_1.npz',
         's25_gold_tx(not deployable)': 'runs/pod/scaling/scale_25/validation_gold_epoch_1.npz',
         'fewshot235': 'runs/pod/probe235/fewshot235/validation.npz'}
res = {'ref': ref.mean(), 'blend_sink_cf': bsink_cf.mean()}
for n, p in cands.items():
    St = normalize(aligned_bundle(ROOT/p, ids, classes)*mask + 1e-9)
    a = row_scores(y, topk(St, classes))
    sc = {}
    for w in W:
        F = geo([(B, 1-w), (St, w)]) if w else B
        for l in LAMS: sc[(w, l)] = row_scores(y, topk(sinkhorn(F, pi, l) if l else F, classes))
    o, ch = crossfit(sc, mf)
    res[n] = dict(alone_top10=a.mean(), unseen_pair=a[up].mean(), passes_gate=bool(a.mean() >= .715 and a[up].mean() >= .48),
                  fused_joint_cf=o.mean(), ci_vs_ref=boot_ci(o, ref, qid), ci_vs_blend_sink_cf=boot_ci(o, bsink_cf, qid),
                  cfg=[list(c) for c in ch])
dump(res, 't6_real_gate.json')
