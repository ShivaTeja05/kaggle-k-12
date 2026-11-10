"""Compare a held-out image experiment against its same-fold CPU baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import read_csv,label_space,aligned_bundle,topk,row_scores,make_folds
from .baseline import HierarchicalPrior
from .analyze import analyze


def evaluate(data,predictions,output,fold=0,split_seed=2026):
    data=Path(data);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    folds=make_folds(train,seed=split_seed)
    with np.load(predictions,allow_pickle=False) as b:ids=b['ids'].tolist()
    held_ids=set(train.loc[folds==fold,'trace_id'])
    if not set(ids)<=held_ids:raise ValueError('Predictions include rows outside declared held-out fold')
    target=train.set_index('trace_id').loc[ids].reset_index()
    p=aligned_bundle(predictions,ids,classes)
    prior=HierarchicalPrior(classes).fit(train.loc[folds!=fold]).predict_proba(target)
    scores=row_scores(target.root_code,topk(p,classes));reference=row_scores(target.root_code,topk(prior,classes))
    groups=pd.DataFrame({'question':target.question_id,'delta':scores-reference}).groupby('question').delta.agg(['sum','size']).to_numpy()
    rng=np.random.default_rng(2026);ix=rng.integers(0,len(groups),(2000,len(groups)))
    boot=groups[ix,0].sum(axis=1)/groups[ix,1].sum(axis=1)
    report={'rows':len(target),'full_fold_rows':len(held_ids),'full_fold_evaluated':set(ids)==held_ids,'fold':fold,
            'image_map3':float(scores.mean()),'same_rows_prior_map3':float(reference.mean()),
            'delta':float((scores-reference).mean()),'paired_question_bootstrap_delta_95':np.quantile(boot,[.025,.975]).tolist(),
            'target':.82,'gap_to_target':float(.82-scores.mean()),'evaluation':'local held-out images; not public leaderboard',
            'decision':'pilot only; expand before making a quality claim' if len(target)<150 else 'review error slices before promotion'}
    pd.DataFrame({'trace_id':train.trace_id,'fold':folds}).to_csv(out/'folds.csv',index=False)
    analyze(data,predictions,out/'errors',out/'folds.csv',allow_holdout=True)
    (out/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw');p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--fold',type=int,default=0);p.add_argument('--split-seed',type=int,default=2026)
    a=p.parse_args();evaluate(a.data,a.predictions,a.output,a.fold,a.split_seed)
