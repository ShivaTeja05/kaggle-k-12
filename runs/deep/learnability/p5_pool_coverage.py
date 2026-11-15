"""Part 5: coverage of test/dev questions and topics by the Claude-labelled pool vs the full unlabelled pool;
dev gap by coverage; how many extra Claude labels would be needed to cover uncovered test questions."""
from lcommon import *
D = get_all(); dev, test, unl, tr = D['dev'], D['test'], D['unl'], D['tr']; y = D['y']; preds = D['preds']
U50 = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv')
S = {k: score(y, v) for k, v in preds.items()}
res = {'unlabelled_rows': int(len(U50)), 'claude_labelled': int(len(unl)),
       'topics': {'train': int(tr.topic_name.nunique()), 'test': int(test.topic_name.nunique()), 'unl50k': int(U50.topic_name.nunique()), 'claude26k': int(unl.topic_name.nunique())},
       'questions': {'test': int(test.question_id.nunique()), 'unl50k': int(U50.question_id.nunique()), 'claude26k': int(unl.question_id.nunique())}}
for nm, pool in [('unl50k', U50), ('claude26k', unl)]:
    res[f'test_rows_q_covered_{nm}'] = round(float(test.question_id.isin(pool.question_id).mean()), 3)
    res[f'test_rows_topic_covered_{nm}'] = round(float(test.topic_name.isin(pool.topic_name).mean()), 3)
    res[f'dev_rows_q_covered_{nm}'] = round(float(dev.question_id.isin(pool.question_id).mean()), 3)
# uncovered test questions that DO have unlabelled photos waiting
unc = set(test.question_id) - set(unl.question_id)
avail = U50[U50.question_id.isin(unc) & ~U50.trace_id.isin(unl.trace_id)]
res['test_q_without_claude_labels'] = len(unc)
res['test_rows_in_those_q'] = int(test.question_id.isin(unc).sum())
res['of_which_q_have_unlabelled_photos'] = int(avail.question_id.nunique())
res['test_rows_coverable_by_labelling_more'] = int(test.question_id.isin(set(avail.question_id)).sum())
res['photos_needed_for_10_per_uncovered_q'] = int(avail.groupby('question_id').size().clip(upper=10).sum())
res['est_cost_usd_at_0.0069_per_photo_10_per_q'] = round(res['photos_needed_for_10_per_uncovered_q'] * 0.0069, 2)
# dev gap by Claude coverage of the question
cov = dev.question_id.isin(unl.question_id).to_numpy(); err = y != 'NONE'; q = dev.question_id.to_numpy()
for nm, m in [('covered', cov), ('uncovered', ~cov)]:
    for e_nm, e in [('all', np.ones_like(err)), ('err', err)]:
        mm = m & e
        d, ci = boot_ci(S['claude'][mm], S['blend_sk'][mm], q[mm])
        res[f'dev_{nm}_{e_nm}'] = {'rows': int(mm.sum()), 'blend_sk': round(float(S['blend_sk'][mm].mean()), 3), 'claude': round(float(S['claude'][mm].mean()), 3),
                                    'scale_0': round(float(S['scale_0'][mm].mean()), 3), 'scale_100': round(float(S['scale_100'][mm].mean()), 3),
                                    'gap': round(d, 3), 'ci': [round(x, 3) for x in ci]}
# within-question label shift: Claude pool NONE share vs train NONE share on the same questions
both = set(tr.question_id) & set(unl.question_id)
res['same_questions_NONE_share_train_vs_claude'] = [round(float((tr[tr.question_id.isin(both)].root_code == 'NONE').mean()), 3), round(float((unl[unl.question_id.isin(both)].c1 == 'NONE').mean()), 3)]
json.dump(res, open(OUT / 'p5_pool_coverage.json', 'w'), indent=1); print(json.dumps(res, indent=1))
