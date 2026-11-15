"""T4. How much of Claude's dev advantage did existing distillation (8B trained on 0/25/100% of Claude-labelled rows)
actually transfer? 'Teachable' rows: Claude's top-1 is the gold code and the reference blend's top-1 is not.
If 26k Claude labels moved recovery of teachable rows only a little, the proposal needs reason-first generation alone
to supply the q~0.25-0.5 'Claude-fidelity' that its gate_sim maps to a fused gain."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; cls = np.array(classes)
B = blend(ids, classes)
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
c1 = cls[C.argmax(1)]; b1 = cls[B.argmax(1)]
teach = (c1 == y) & (b1 != y)
models = {'blend': B,
          's0': aligned_bundle(ROOT/'runs/pod/scaling/scale_0/validation_tx_dev_epoch_1.npz', ids, classes),
          's25': aligned_bundle(ROOT/'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz', ids, classes),
          's100': aligned_bundle(ROOT/'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz', ids, classes),
          's100_gold_tx': aligned_bundle(ROOT/'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz', ids, classes),
          'q32': aligned_bundle(ROOT/'runs/pod/text32/validation_tx_dev_epoch_1.npz', ids, classes),
          'q8_e3': aligned_bundle(ROOT/'runs/pod/tx/text_tx_dev_e3.npz', ids, classes),
          'vl_distill': aligned_bundle(ROOT/'runs/pod/dev-distill/validation_epoch_2.npz', ids, classes),
          'fewshot235': aligned_bundle(ROOT/'runs/pod/probe235/fewshot235/validation.npz', ids, classes)}
res = {'n_teachable': int(teach.sum()), 'n_teachable_unseen_pair': int((teach & up).sum()), 'n_unseen_pair': int(up.sum()),
       'claude_top1_acc': float((c1 == y).mean()), 'claude_top1_acc_unseen_pair': float((c1 == y)[up].mean())}
for n, P in models.items():
    p1 = cls[P.argmax(1)]; t3 = np.argsort(-P, 1)[:, :3]
    in3 = np.array([ci[yy] in t for yy, t in zip(y, t3)])
    c_in3 = np.array([ci[c] in t for c, t in zip(c1, t3)])
    pts = row_scores(y, topk(P, classes))
    res[n] = dict(map3=pts.mean(), unseen_pair=pts[up].mean(),
                  agree_top1_with_claude=float((p1 == c1).mean()), agree_top1_unseen_pair=float((p1 == c1)[up].mean()),
                  claude_top1_in_model_top3=float(c_in3.mean()),
                  teachable_top1_hit=float((p1 == y)[teach].mean()), teachable_top3_hit=float(in3[teach].mean()),
                  teachable_unseen_top3_hit=float(in3[teach & up].mean()))
dump(res, 't4_fidelity.json')
