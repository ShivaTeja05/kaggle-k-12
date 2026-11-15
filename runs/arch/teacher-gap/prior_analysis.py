"""Is the blend's gap to Claude a 'question/topic prior vs mechanism' gap? Uses only non-dev rows for priors."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from k12.core import read_csv, make_folds, label_space
OUT = 'runs/arch/teacher-gap/'
train = read_csv('data/raw/train.csv'); folds = make_folds(train)
fit = train.loc[folds != 0]
d = pd.read_csv(OUT + 'rows.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
qc = {q: Counter(g.root_code) for q, g in fit.groupby('question_id')}
tc = {t: Counter(g.root_code) for t, g in fit.groupby('topic_name')}
code_topics = fit[fit.root_code != 'NONE'].groupby('root_code').topic_name.nunique()

def share(c, cnt):
    return np.nan if cnt is None else cnt[c] / sum(cnt.values())
d['q_seen'] = d.question_id.map(lambda q: q in qc)
d['q_share_truth'] = [share(t, qc.get(q)) for t, q in zip(d.truth, d.question_id)]
d['t_share_truth'] = [share(t, tc.get(tp)) for t, tp in zip(d.truth, d.topic)]
d['q_n'] = d.question_id.map(lambda q: sum(qc[q].values()) if q in qc else 0)
mode = {q: c.most_common(1)[0][0] for q, c in qc.items()}
d['q_mode'] = d.question_id.map(mode).fillna('')
b1, c1 = d.blend.str.split().str[0], d.claude.str.split().str[0]
res = {}
err = d.truth != 'NONE'
res['dev_questions_seen_in_fit'] = float(d.q_seen.mean())
res['median_fit_rows_per_dev_question'] = float(d.q_n.median())
res['top1_equals_question_mode'] = {'blend': float((b1 == d.q_mode)[d.q_seen].mean()), 'claude': float((c1 == d.q_mode)[d.q_seen].mean()),
                                    'truth': float((d.truth == d.q_mode)[d.q_seen].mean())}
res['top1_equals_question_mode_error_rows'] = {'blend': float((b1 == d.q_mode)[d.q_seen & err].mean()), 'claude': float((c1 == d.q_mode)[d.q_seen & err].mean()),
                                               'truth': float((d.truth == d.q_mode)[d.q_seen & err].mean())}
# gap by how typical the truth is for its question (error rows only; NONE handled separately)
e = d[err].copy()
bins = pd.cut(e.q_share_truth.fillna(-1), [-1.5, -.5, 0, .2, .5, 1.01], labels=['question_unseen', '0 (never seen for q)', '(0,.2]', '(.2,.5]', '(.5,1]'])
res['error_rows_by_question_share_of_truth'] = {str(k): {'rows': int(len(v)), 'blend': round(v.s_blend.mean(), 3), 'claude': round(v.s_claude.mean(), 3),
                                                          'gold32': round(v.s_gold32.mean(), 3), 'net_gain_per_301': round(float((v.s_claude - v.s_blend).sum() / 301), 4)}
                                                 for k, v in e.groupby(bins, observed=True)}
bins = pd.cut(e.t_share_truth.fillna(-1), [-1.5, -.5, 0, .1, .3, 1.01], labels=['topic_unseen', '0', '(0,.1]', '(.1,.3]', '(.3,1]'])
res['error_rows_by_topic_share_of_truth'] = {str(k): {'rows': int(len(v)), 'blend': round(v.s_blend.mean(), 3), 'claude': round(v.s_claude.mean(), 3),
                                                       'net_gain_per_301': round(float((v.s_claude - v.s_blend).sum() / 301), 4)}
                                              for k, v in e.groupby(bins, observed=True)}
# cross-topic ('mechanism') codes vs topic-bound codes
e['code_topic_span'] = e.truth.map(code_topics).fillna(0)
bins = pd.cut(e.code_topic_span, [-1, 3, 8, 100], labels=['topic-bound (<=3 topics)', 'mid (4-8)', 'cross-topic (>8)'])
res['error_rows_by_code_topic_span'] = {str(k): {'rows': int(len(v)), 'codes': sorted(set(v.truth)), 'blend': round(v.s_blend.mean(), 3), 'claude': round(v.s_claude.mean(), 3),
                                                  'net_gain_per_301': round(float((v.s_claude - v.s_blend).sum() / 301), 4)}
                                         for k, v in e.groupby(bins, observed=True)}
# fallback codes when wrong
wrongb, wrongc = d[b1 != d.truth], d[c1 != d.truth]
res['wrong_top1_code_distribution'] = {'blend': Counter(b1[b1 != d.truth]).most_common(8), 'claude': Counter(c1[c1 != d.truth]).most_common(8)}
# outcome x whether truth == question mode
d['truth_is_qmode'] = (d.truth == d.q_mode)
res['outcome_by_truth_is_question_mode'] = pd.crosstab(d.truth_is_qmode, d.outcome).to_dict()
res['mean_map_by_truth_is_question_mode'] = {str(k): {'rows': int(len(v)), 'blend': round(v.s_blend.mean(), 3), 'claude': round(v.s_claude.mean(), 3)} for k, v in d.groupby('truth_is_qmode')}
json.dump(res, open(OUT + 'prior_gap.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
