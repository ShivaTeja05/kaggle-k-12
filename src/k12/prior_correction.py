"""CPU-only conditional-prior correction, screened with question-group cross-fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import HierarchicalPrior
from .core import (aligned_bundle, label_space, make_folds, map3, normalize,
                   read_csv, row_scores, save_bundle, topk, write_submission)
from .holdout_calibration import calibrated


def corrected(image, question_prior, global_prior, exponent, temperature):
    ratio=np.clip(question_prior/np.maximum(global_prior,1e-12),.1,10.)
    return normalize(np.maximum(image,1e-12)**(1/temperature)*ratio**exponent)


def run(args):
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');test=read_csv(data/'test.csv')
    classes=label_space(read_csv(data/'taxonomy_v1.csv'));folds=make_folds(train)
    fit=train.loc[folds!=0];val=train.loc[folds==0].reset_index(drop=True)
    image=aligned_bundle(args.validation,val.trace_id.tolist(),classes)
    prior=HierarchicalPrior(classes).fit(fit);q=prior.predict_proba(val)
    candidates=[(a,t) for a in [0.,.25,.5,1.] for t in [1.,2.]]
    # Hold entire questions out of parameter selection. This does not undo
    # previous image-epoch selection on the same development fold.
    questions=val.question_id.unique().copy();np.random.default_rng(2026).shuffle(questions)
    assignment={q:i%5 for i,q in enumerate(questions)}
    groups=np.asarray([assignment[q] for q in val.question_id]);crossfit=np.empty_like(image)
    choices=[]
    all_p=[corrected(image,q,prior.global_p,a,t) for a,t in candidates]
    for f in range(5):
        fitting=groups!=f;held=~fitting
        scores=[map3(val.loc[fitting,'root_code'],p[fitting],classes) for p in all_p]
        chosen=int(np.argmax(scores));crossfit[held]=all_p[chosen][held]
        choices.append({'meta_fold':f,'exponent':candidates[chosen][0],
                        'temperature':candidates[chosen][1],'rows':int(held.sum())})
    reference=calibrated(image,q,.25,2.)
    delta=row_scores(val.root_code,topk(crossfit,classes))-row_scores(val.root_code,topk(reference,classes))
    agg=pd.DataFrame({'question':val.question_id,'delta':delta}).groupby('question').delta.agg(['sum','size']).to_numpy()
    ix=np.random.default_rng(2026).integers(0,len(agg),(2000,len(agg)))
    interval=np.quantile(agg[ix,0].sum(1)/agg[ix,1].sum(1),[.025,.975]).tolist()
    scores=[map3(val.root_code,p,classes) for p in all_p];best=int(np.argmax(scores));a,t=candidates[best]
    promote=bool(delta.mean()>=.01 and interval[0]>0)
    report={'rows':len(val),'reference_fixed_calibration_map3':map3(val.root_code,reference,classes),
            'question_crossfit_map3':map3(val.root_code,crossfit,classes),'delta':float(delta.mean()),
            'paired_question_bootstrap_delta_95':interval,'full_development_tuned_map3':scores[best],
            'selected_exponent':a,'selected_temperature':t,'meta_fold_choices':choices,
            'candidates':[{'exponent':a,'temperature':t,'map3':s} for (a,t),s in zip(candidates,scores)],
            'promotion_rule':'crossfit delta >= 0.01 and paired question-bootstrap lower bound > 0',
            'promote':promote,'warning':'Exploratory reused development fold, conditional on prior image-epoch selection; not an untouched audit. No public leaderboard feedback used in selection.'}
    save_bundle(out/'crossfit.npz',val.trace_id.tolist(),classes,crossfit,kind='question_crossfit_prior_correction')
    test_image=aligned_bundle(args.test,test.trace_id.tolist(),classes)
    prior=HierarchicalPrior(classes).fit(train)
    p=corrected(test_image,prior.predict_proba(test),prior.global_p,a,t)
    save_bundle(out/'test.npz',test.trace_id.tolist(),classes,p,kind='conditional_prior_corrected_test')
    write_submission(test,p,classes,out/'submission.csv')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw')
    parser.add_argument('--validation',default='runs/vision-development/k12/validation_epoch_3.npz')
    parser.add_argument('--test',default='runs/vision-development/k12/test.npz')
    parser.add_argument('--output',default='artifacts/prior-correction');run(parser.parse_args())
