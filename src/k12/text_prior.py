"""Question-text prior using test-available context; CPU-only screening."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .baseline import HierarchicalPrior
from .core import (aligned_bundle, label_space, make_folds, map3, normalize,
                   read_csv, row_scores, save_bundle, topk, write_submission)
from .holdout_calibration import calibrated


class TextPrior:
    def __init__(self,classes):
        self.classes=list(classes)
        self.word=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=15000,sublinear_tf=True)
        self.char=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,max_features=25000,sublinear_tf=True)
        self.context=OneHotEncoder(handle_unknown='ignore')
        self.model=LogisticRegression(C=4,max_iter=500,solver='lbfgs')

    @staticmethod
    def text(frame):
        return frame.question_text+' || '+frame.correct_answer.astype(str)+' || '+frame.topic_name

    def fit(self,frame):
        text=self.text(frame)
        x=hstack([self.word.fit_transform(text),self.char.fit_transform(text),
                  2*self.context.fit_transform(frame[['question_id','grade','topic_name']].astype(str))]).tocsr()
        self.model.fit(x,frame.root_code);return self

    def predict_proba(self,frame):
        text=self.text(frame)
        x=hstack([self.word.transform(text),self.char.transform(text),
                  2*self.context.transform(frame[['question_id','grade','topic_name']].astype(str))]).tocsr()
        p=np.full((len(frame),len(self.classes)),1e-12)
        ci={c:i for i,c in enumerate(self.classes)}
        p[:,[ci[c] for c in self.model.classes_]]=self.model.predict_proba(x)
        return normalize(p)


def run(args):
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');test=read_csv(data/'test.csv');classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    folds=make_folds(train);fit=train.loc[folds!=0];val=train.loc[folds==0].reset_index(drop=True)
    image=aligned_bundle(args.validation,val.trace_id.tolist(),classes)
    model=TextPrior(classes).fit(fit);prior=model.predict_proba(val)
    reference=calibrated(image,HierarchicalPrior(classes).fit(fit).predict_proba(val),.25,2.)
    weights=[0.,.1,.25,.5];candidates=[calibrated(image,prior,w,2.) for w in weights]
    questions=val.question_id.unique().copy();np.random.default_rng(2026).shuffle(questions)
    groups={q:i%5 for i,q in enumerate(questions)};fold=np.asarray([groups[q] for q in val.question_id])
    p=np.empty_like(image);choices=[]
    for f in range(5):
        a=fold!=f;b=~a
        scores=[map3(val.loc[a,'root_code'],q[a],classes) for q in candidates]
        chosen=int(np.argmax(scores));p[b]=candidates[chosen][b];choices.append(weights[chosen])
    delta=row_scores(val.root_code,topk(p,classes))-row_scores(val.root_code,topk(reference,classes))
    agg=pd.DataFrame({'question':val.question_id,'delta':delta}).groupby('question').delta.agg(['sum','size']).to_numpy()
    ix=np.random.default_rng(2026).integers(0,len(agg),(2000,len(agg)))
    interval=np.quantile(agg[ix,0].sum(1)/agg[ix,1].sum(1),[.025,.975]).tolist()
    scores=[map3(val.root_code,q,classes) for q in candidates];chosen=int(np.argmax(scores))
    report={'rows':len(val),'text_prior_map3':map3(val.root_code,prior,classes),
            'reference_fixed_calibration_map3':map3(val.root_code,reference,classes),
            'question_crossfit_blend_map3':map3(val.root_code,p,classes),'delta':float(delta.mean()),
            'paired_question_bootstrap_delta_95':interval,'full_development_tuned_map3':scores[chosen],
            'selected_prior_weight':weights[chosen],'temperature':2.,'meta_fold_weights':choices,
            'candidates':[{'prior_weight':w,'map3':s} for w,s in zip(weights,scores)],
            'promote':bool(delta.mean()>=.01 and interval[0]>0),
            'warning':'Exploratory reused development fold, conditional on image-epoch selection. Only test-available context inputs. No public feedback used.'}
    save_bundle(out/'crossfit.npz',val.trace_id.tolist(),classes,p,kind='question_crossfit_text_prior_blend')
    test_image=aligned_bundle(args.test,test.trace_id.tolist(),classes)
    model=TextPrior(classes).fit(train);joblib.dump(model,out/'prior.joblib')
    p=calibrated(test_image,model.predict_proba(test),weights[chosen],2.)
    save_bundle(out/'test.npz',test.trace_id.tolist(),classes,p,kind='text_prior_image_blend_test')
    write_submission(test,p,classes,out/'submission.csv')
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',default='data/raw')
    parser.add_argument('--validation',default='runs/vision-development/k12/validation_epoch_3.npz')
    parser.add_argument('--test',default='runs/vision-development/k12/test.npz')
    parser.add_argument('--output',default='artifacts/text-prior');run(parser.parse_args())
