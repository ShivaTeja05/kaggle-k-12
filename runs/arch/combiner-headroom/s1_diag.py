from common import *
from priors import *
lab = claude_unl()
print('claude unl rows', len(lab), 'questions', lab.question_id.nunique())
print('dev qids in claude unl', dev.question_id.isin(lab.question_id).mean())
test = read_csv(R+'data/raw/test.csv')
print('test qids in claude unl', test.question_id.isin(lab.question_id).mean(), 'test qids in train', test.question_id.isin(train.question_id).mean())
# overlap of unlabelled trace ids with train/test
print('unl ids in train/test:', lab.trace_id.isin(train.trace_id).sum(), lab.trace_id.isin(test.trace_id).sum())
gp = gold_prior(); print('gold prior map3', pts(gp).mean())
for a in [1, 5, 20]:
    cp = claude_prior(a=a); print('claude prior a', a, pts(cp).mean())
ref = geo([load('t32'), load('t8e3'), load('vl')], [.35, .30, .35])
o = np.argsort(-ref, 1)
rank = np.argmax(o == y[:, None], 1)
for k in [1, 3, 5, 10, 20]: print('recall@%d' % k, (rank < k).mean())
print('NONE share truth', (np.array(classes)[y] == 'NONE').mean(), 'pred NONE first', (o[:, 0] == ci['NONE']).mean())
# per-question label entropy in fit rows: how deterministic is code given question?
g = fit.groupby('question_id').root_code.agg(lambda s: s.value_counts().iloc[0] / len(s))
print('fit questions', fit.question_id.nunique(), 'mean rows per q', len(fit) / fit.question_id.nunique(), 'mean majority share', g.mean())
