"""CPU-only prior blending; cross-fit calibration by question within development."""
import argparse
import json
from pathlib import Path

import numpy as np

from .core import read_csv,label_space,aligned_bundle,make_folds,map3,normalize,save_bundle,write_submission
from .baseline import HierarchicalPrior


def calibrated(image,prior,weight,temperature):
    softened=normalize(np.maximum(image,1e-12)**(1/temperature))
    return (1-weight)*softened+weight*prior


def run(args):
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');test=read_csv(data/'test.csv');classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    folds=make_folds(train);fit=train.loc[folds!=0];val=train.loc[folds==0].reset_index(drop=True)
    ids=val.trace_id.tolist();image=aligned_bundle(args.validation,ids,classes)
    prior=HierarchicalPrior(classes).fit(fit).predict_proba(val)
    candidates=[(w,t) for w in [0,.1,.25,.5] for t in [1.,2.]]
    questions=val.question_id.unique().copy();np.random.default_rng(2026).shuffle(questions)
    group={q:i%5 for i,q in enumerate(questions)};meta=np.asarray([group[q] for q in val.question_id])
    crossfit=np.empty_like(image);choices=[]
    for fold in range(5):
        a=meta!=fold;b=~a
        scores=[map3(val.loc[a,'root_code'],calibrated(image[a],prior[a],w,t),classes) for w,t in candidates]
        w,t=candidates[int(np.argmax(scores))];crossfit[b]=calibrated(image[b],prior[b],w,t)
        choices.append({'meta_fold':fold,'prior_weight':w,'temperature':t,'rows':int(b.sum())})
    scores=[map3(val.root_code,calibrated(image,prior,w,t),classes) for w,t in candidates]
    best=int(np.argmax(scores));w,t=candidates[best]
    result={'rows':len(val),'raw_image_map3':map3(val.root_code,image,classes),
            'question_crossfit_calibration_map3':map3(val.root_code,crossfit,classes),
            'full_development_tuned_map3':scores[best],'selected_prior_weight':w,'selected_temperature':t,
            'meta_fold_choices':choices,'candidates':[{'prior_weight':a,'temperature':b,'map3':s} for (a,b),s in zip(candidates,scores)],
            'warning':'Calibration cross-fitting is conditional on an image model already selected using this development fold; this is not an untouched audit or full image-model OOF.'}
    save_bundle(out/'meta_crossfit.npz',ids,classes,crossfit,kind='crossfit_calibration_within_development')
    save_bundle(out/'tuned_development.npz',ids,classes,calibrated(image,prior,w,t),kind='in_sample_calibration_development')
    test_image=aligned_bundle(args.test,test.trace_id.tolist(),classes)
    test_prior=HierarchicalPrior(classes).fit(train).predict_proba(test)
    p=calibrated(test_image,test_prior,w,t)
    save_bundle(out/'test.npz',test.trace_id.tolist(),classes,p,kind='calibrated_test')
    write_submission(test,p,classes,out/'submission.csv')
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw')
    parser.add_argument('--validation',default='runs/vision-development/k12/validation_epoch_3.npz')
    parser.add_argument('--test',default='runs/vision-development/k12/test.npz')
    parser.add_argument('--output',default='artifacts/holdout-calibration');run(parser.parse_args())
