"""C ceiling: which dev error rows could mal-rule injection even address? Model-free counts."""
import json
from collections import Counter
import numpy as np, pandas as pd
from common import load
tr, tax, classes, folds, dev, fit = load()
INJ = {'EXP-01','EXP-02','EXP-03','EXP-04','EXP-05','FRA-01','FRA-03','INT-02','INT-03','INT-04','OOO-01','RAT-02','ALG-02','ALG-04'}
fitq = fit.groupby('question_id').root_code.apply(set).to_dict()
sl = np.array(['none' if t=='NONE' else 'unseen_q' if q not in fitq else 'seen_pair' if t in fitq[q] else 'new_pair'
               for q,t in zip(dev.question_id, dev.root_code)])
dev['slice']=sl
print(Counter(sl))
err = dev[dev.root_code!='NONE']
print('dev error codes overall top:', Counter(err.root_code).most_common(15))
np_ = dev[np.isin(sl,['new_pair','unseen_q'])]
print('new_pair+unseen codes:', Counter(np_.root_code).most_common(30))
print('injectable among new_pair+unseen:', int(np_.root_code.isin(INJ).sum()), 'of', len(np_))
print('injectable among all dev errors:', int(err.root_code.isin(INJ).sum()), 'of', len(err))
# among fit gold: frequency of injectable codes
print('fit injectable error share', round(fit[fit.root_code!='NONE'].root_code.isin(INJ).mean(),3))
# does a fit NONE donor exist on the same question for the new-pair injectable rows?
fit_none_q = set(fit.question_id[fit.root_code=='NONE'])
x = np_[np_.root_code.isin(INJ)]
print('injectable new/unseen rows with a fit NONE donor on same question:', int(x.question_id.isin(fit_none_q).sum()))
print(x[['question_id','topic_name','root_code','slice']].to_string())
json.dump(dict(slices=Counter(sl), np_codes=Counter(np_.root_code), inj_np=int(np_.root_code.isin(INJ).sum()), n_np=len(np_),
               inj_err=int(err.root_code.isin(INJ).sum()), n_err=len(err), inj_np_with_donor=int(x.question_id.isin(fit_none_q).sum())),
          open('runs/deep/test-W5/c0_ceiling.json','w'), indent=1)
