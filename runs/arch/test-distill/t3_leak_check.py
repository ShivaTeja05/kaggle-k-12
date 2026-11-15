"""T3. Is the d4 gain partly near-duplicate leakage? Nearest Claude-unlabelled transcription for each dev row
(char 3-5gram TF-IDF cosine over Claude's own transcriptions), and the d4 fused gain split by that similarity.
Also: coverage of dev / test questions by the 26k Claude-labelled unlabelled rows."""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import *

cf = np.load(OUT/'cf_claude_full.npy')
ut = [U[t].get('transcription') or '' for t in U_ids]; dt = [D.get(t, {}).get('transcription') or '' for t in ids]
v = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(ut)
Xu = v.transform(ut); Xd = v.transform(dt)
sim = (Xd @ Xu.T).max(axis=1).toarray().ravel()
res = {'nn_sim_percentiles_5_50_95_max': [round(float(x), 3) for x in np.percentile(sim, [5, 50, 95, 100])]}
for th in [0.8, 0.9, 0.95]:
    m = sim >= th
    res[f'rows_sim_ge_{th}'] = int(m.sum())
    if (~m).sum(): res[f'd4_delta_excluding_sim_ge_{th}'] = round(float(cf[~m].mean() - sb[~m].mean()), 4)
test = read_csv(ROOT/'data/raw/test.csv'); uqs = set(U_q)
res['dev_rows_question_in_claude_unl'] = round(float(np.mean([q in uqs for q in qid])), 4)
res['dev_q_unseen_rows_question_in_claude_unl'] = round(float(np.mean([q in uqs for q in qid[bins == 'q_unseen']])), 4)
res['test_rows_question_in_claude_unl'] = round(float(np.mean([q in uqs for q in test.question_id])), 4)
print(json.dumps(res, indent=1)); json.dump(res, open(OUT/'t3_leak_check.json', 'w'), indent=1)
