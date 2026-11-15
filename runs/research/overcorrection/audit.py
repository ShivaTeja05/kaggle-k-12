"""Over-correction audit (PINK, EMNLP 2026): does the machine transcriber turn a wrong final answer into the right one?
Dev fold 0 (301 rows). Gold = annotator transcription; machine = runs/pod/tx/tx_dev.json. Uses the crude checker k=2."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
from checker import check
tr, tax, classes, folds, dev, fit = load()
print([c for c in dev.columns])
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
B = blend(ids, classes); pts = row_scores(y, topk(B, classes))
g = np.array([check(t, a, 2)[0] >= .999 for t, a in zip(dev.transcription_steps, dev.correct_answer)])
m = np.array([check(tx[i], a, 2)[0] >= .999 for i, a in zip(ids, dev.correct_answer)])
fac = dev.final_answer_correct.astype(str).eq('True').to_numpy(); err = y != 'NONE'
over = (~g) & m; under = g & (~m)
out = {'n': len(ids),
       'gold_says_correct': int(g.sum()), 'machine_says_correct': int(m.sum()),
       'over_corrected_rows (gold wrong, machine right)': int(over.sum()),
       'over_rate_among_gold_wrong': float(over.sum() / (~g).sum()),
       'over_rows_that_are_errors': int((over & err).sum()),
       'over_rows_fac_false': int((over & ~fac).sum()),
       'under_rows (gold right, machine wrong)': int(under.sum()),
       'blend_map3_over_rows': float(pts[over].mean()) if over.any() else None,
       'blend_map3_other_err_rows': float(pts[~over & err].mean()),
       'blend_top1_NONE_on_over_err_rows': int(sum(classes[int(np.argmax(B[i]))] == 'NONE' for i in np.where(over & err)[0])),
       'map_points_lost_on_over_rows_per_301': float((1 - pts[over]).sum() / len(ids))}
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT/'runs/research/overcorrection/audit.json', 'w'), indent=1)
# Same check on Claude Opus dev transcriptions (the distillation targets on 26k photos come from the same prompt).
cl = {}
for line in open(ROOT/'runs/claude/dev_opus.jsonl'):
    r = json.loads(line); cl[r['trace_id']] = r['answer'].get('transcription') or ''
c = np.array([check(cl.get(i, ''), a, 2)[0] >= .999 for i, a in zip(ids, dev.correct_answer)])
oc = (~g) & c
out2 = {'claude_says_correct': int(c.sum()), 'claude_over_rows': int(oc.sum()),
        'claude_over_rate_among_gold_wrong': float(oc.sum() / (~g).sum()),
        'claude_over_rows_that_are_errors': int((oc & err).sum()),
        'claude_under_rows': int((g & ~c).sum())}
print(json.dumps(out2, indent=1)); out.update(out2)
json.dump(out, open(ROOT/'runs/research/overcorrection/audit.json', 'w'), indent=1)
