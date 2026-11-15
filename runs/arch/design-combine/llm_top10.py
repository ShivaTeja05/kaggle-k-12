"""CPU check: existing untuned LLM bundles (zero/few-shot 32B/235B, codes with descriptions in prompt?) used only as
a re-ranker inside the blend's top-10, fusion weight cross-fitted on 5 question folds; per bin."""
import sys, json
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import *
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
P = blend(ids, classes); ref = row_scores(y, topk(P, classes))
fitcodes = fit.groupby('question_id').root_code.agg(set).to_dict()
up = np.array([(yy != 'NONE') and (yy not in fitcodes.get(q, set())) for yy, q in zip(y, qid)])
qs = np.unique(qid).copy(); np.random.default_rng(7).shuffle(qs)
g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
top10 = np.argsort(-P, 1)[:, :10]
def sc(M): return row_scores(y, topk(M, classes))
res = {}
for name in ['zeroshot32', 'fewshot32', 'zeroshot235', 'fewshot235', 'pass1']:
    V = aligned_bundle(ROOT/f'runs/pod/probe235/{name}/validation.npz', ids, classes)
    V10 = np.full_like(V, 1e-6)
    for i in range(len(y)): V10[i, top10[i]] = V[i, top10[i]] + 1e-4
    V10 = normalize(V10)
    cands = {}
    for w in [0, .25, .5, 1]:
        lp = np.log(P) + w*np.log(V10); cands[w] = sc(normalize(np.exp(lp - lp.max(1, keepdims=True))))
    out = np.empty(len(y)); ch = []
    for f in range(5):
        m = mf != f; w = max(cands, key=lambda w: cands[w][m].mean()); ch.append(w); out[~m] = cands[w][~m]
    d, c = boot_ci(out, ref, qid)
    res[name] = dict(alone=float(sc(V).mean()), alone_top10=float(sc(V10).mean()), alone_unseen_pair=float(sc(V)[up].mean()),
                     fused=float(out.mean()), delta=d, ci95=c, w=ch, fused_unseen_pair=float(out[up].mean()))
res['ref_unseen_pair'] = float(ref[up].mean())
print(json.dumps(res, indent=1)); json.dump(res, open('llm_top10.json', 'w'), indent=1)
