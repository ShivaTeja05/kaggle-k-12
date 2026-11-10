"""Diagnostic only: how far do word/character features get with gold handwriting text?

This uses a train-only input and cannot generate a valid deployment/test model.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .core import read_csv,label_space,make_folds,map3,save_bundle


def text(df):
    strip=lambda s:re.sub(r'(?m)^\s*L\d+\s*\|\s*','',str(s))
    return (df.topic_name+' || '+df.question_text+' || KEY '+df.correct_answer.astype(str)+
            ' || WORK '+df.transcription_steps.map(strip))


def run(args):
    data=Path(args.data);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    train=read_csv(data/'train.csv');classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    ci={c:i for i,c in enumerate(classes)};folds=make_folds(train);sentences=text(train)
    p=np.zeros((len(train),len(classes)));scores=[]
    for fold in range(5):
        a=folds!=fold;v=~a
        word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=25000,sublinear_tf=True)
        char=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=3,max_features=40000,sublinear_tf=True)
        x=hstack([word.fit_transform(sentences[a]),char.fit_transform(sentences[a])]).tocsr()
        xv=hstack([word.transform(sentences[v]),char.transform(sentences[v])]).tocsr()
        model=LogisticRegression(C=4,max_iter=1200).fit(x,train.loc[a,'root_code'].map(ci))
        p[np.ix_(v,model.classes_)]=model.predict_proba(xv)
        score=map3(train.loc[v,'root_code'],p[v]+1e-12,classes);scores.append(score)
        print(json.dumps({'fold':fold,'diagnostic_map3':score}),flush=True)
    save_bundle(out/'gold_text_diagnostic.npz',train.trace_id.tolist(),classes,p+1e-12,
                kind='gold_transcription_diagnostic',deployable=False)
    result={'map3':map3(train.root_code,p+1e-12,classes),'fold_map3':scores,
            'deployable':False,'interpretation':'Diagnostic gold text with a simple classifier; not a ceiling for stronger models and not a test-time pipeline.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw');p.add_argument('--output',default='artifacts/gold-text-diagnostic')
    run(p.parse_args())
