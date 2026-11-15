import json, sys, difflib
from collections import Counter
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/recipe-audit')
from common import *
R = {}
ref = np.load(OUT / 'blend_ref.npy'); top = topk(ref, classes)
dc = claude_rank(ROOT / 'runs/claude/dev_opus.jsonl')
pairs = Counter(); claude_fixed = Counter()
for i, y in enumerate(truth):
    if top[i, 0] != y:
        pairs[(y, top[i, 0])] += 1
        claude_fixed[(y, top[i, 0])] += dc[ids[i]][0] == y
R['top_blend_confusions_truth_pred_count_claudeRight'] = [(a, b, n, claude_fixed[(a, b)]) for (a, b), n in pairs.most_common(15)]
fampairs = Counter((fam(a), fam(b)) for (a, b), n in pairs.items() for _ in range(n) if a != 'NONE' and b != 'NONE' and fam(a) != fam(b))
R['top_wrong_family_pairs'] = fampairs.most_common(10)
# Claude transcription closeness to gold vs our transcriber (fit text for pseudo rows is Claude's transcription)
strip = lambda s: '\n'.join(l.split('|', 1)[-1].strip() for l in str(s).splitlines() if l.strip())
cdev = {json.loads(l)['trace_id']: json.loads(l)['answer'].get('transcription', '') for l in (ROOT / 'runs/claude/dev_opus.jsonl').read_text().splitlines()}
sc = np.array([difflib.SequenceMatcher(None, strip(r.transcription_steps), strip(cdev[r.trace_id])).ratio() for r in dev.itertuples()])
st = np.load(OUT / 'tx_similarity.npy')
R['dev_similarity_to_gold'] = {'claude_median': round(float(np.median(sc)), 3), 'claude_q25': round(float(np.quantile(sc, .25)), 3),
                               'tx_median': round(float(np.median(st)), 3), 'tx_q25': round(float(np.quantile(st, .25)), 3),
                               'rows_claude_closer': int(sum(sc > st))}
# Rows where the true code was in our top-10 but not top-3, and Claude got it in top-3 (reranker headroom)
r10 = np.array([y in np.asarray(classes)[np.argsort(-ref[i])[:10]] for i, y in enumerate(truth)])
r3 = np.array([y in top[i] for i, y in enumerate(truth)])
c3 = np.array([y in dc[ids[i]] for i, y in enumerate(truth)])
R['in_top10_not_top3'] = int(sum(r10 & ~r3)); R['in_top10_not_top3_claude_top3'] = int(sum(r10 & ~r3 & c3))
R['rank_of_truth_hist'] = dict(Counter(int(np.where(np.argsort(-ref[i]) == ci[y])[0][0]) + 1 for i, y in enumerate(truth)).most_common(8))
(OUT / 'a6_confusions.json').write_text(json.dumps(R, indent=1, default=str)); print(json.dumps(R, indent=1, default=str))
