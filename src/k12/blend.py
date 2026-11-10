"""Select a conservative blend on development folds; report an untouched audit fold."""
import argparse
import json
from pathlib import Path
import numpy as np

from .core import read_csv, label_space, aligned_bundle, map3, save_bundle, write_submission


def run(args):
    data=Path(args.data);train=read_csv(data/'train.csv');test=read_csv(data/'test.csv')
    classes=label_space(read_csv(data/'taxonomy_v1.csv'))
    for path in [args.oof_a,args.oof_b]:
        with np.load(path,allow_pickle=False) as bundle:
            meta=json.loads(str(bundle['meta']))
            if meta.get('deployable') is False:
                raise ValueError('A train-only diagnostic cannot be promoted into a deployable blend')
    folds=read_csv(args.folds).set_index('trace_id').reindex(train.trace_id)['fold'].to_numpy()
    if np.isnan(folds).any():raise ValueError('Missing fold assignment')
    a=aligned_bundle(args.oof_a,train.trace_id.tolist(),classes)
    b=aligned_bundle(args.oof_b,train.trace_id.tolist(),classes)
    tune=folds!=args.audit_fold;audit=~tune
    candidates=[0,.1,.25,.5,.75,.9,1.]
    scores=[map3(train.root_code[tune],(1-w)*a[tune]+w*b[tune],classes) for w in candidates]
    weight=candidates[int(np.argmax(scores))]
    oof=(1-weight)*a+weight*b
    result={'weight_b':weight,'development_map3':max(scores),
            'audit_map3':map3(train.root_code[audit],oof[audit],classes),
            'audit_a':map3(train.root_code[audit],a[audit],classes),
            'audit_b':map3(train.root_code[audit],b[audit],classes),
            'audit_fold':args.audit_fold,
            'warning':'Audit is useful only if held out from all previous model/epoch/prompt selection.'}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    save_bundle(out/'oof.npz',train.trace_id.tolist(),classes,oof,kind='oof_blend')
    if bool(args.test_a)!=bool(args.test_b):raise ValueError('Supply both test bundles')
    if args.test_a:
        ta=aligned_bundle(args.test_a,test.trace_id.tolist(),classes);tb=aligned_bundle(args.test_b,test.trace_id.tolist(),classes)
        pt=(1-weight)*ta+weight*tb
        save_bundle(out/'test.npz',test.trace_id.tolist(),classes,pt,kind='test_blend')
        write_submission(test,pt,classes,out/'submission.csv')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/raw');p.add_argument('--folds',required=True)
    for name in ['oof-a','oof-b']:p.add_argument('--'+name,required=True)
    for name in ['test-a','test-b']:p.add_argument('--'+name)
    p.add_argument('--audit-fold',type=int,default=4);p.add_argument('--output',default='artifacts/blend');run(p.parse_args())
