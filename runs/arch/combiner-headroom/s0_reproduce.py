from common import *
for n in BUNDLES:
    try:
        p = load(n); print(n, round(pts(p).mean(), 4))
    except Exception as e: print(n, 'ERR', e)
for t8 in ['t8e3', 't8e2']:
    p = geo([load('t32'), load(t8), load('vl')], [.35, .30, .35]); print('blend', t8, round(pts(p).mean(), 4))
print('rows', len(dev), 'K', K, 'fit rows', len(fit))
print('dev questions', dev.question_id.nunique(), 'dev qids seen in fit', dev.question_id.isin(fit.question_id).mean())
