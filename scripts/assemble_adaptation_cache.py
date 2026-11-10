"""Join separately extracted partitions after checking encoder/checkpoint identity."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def embedded_sources(path):
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in node.targets):
            return json.loads(ast.literal_eval(node.value.args[0]))
    raise ValueError('Bundled source manifest not found')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['fitting','validation','fitting-source','validation-source','output']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args();fit=Path(args.fitting);val=Path(args.validation);out=Path(args.output)
    fm=json.loads((fit/'manifest.json').read_text());vm=json.loads((val/'manifest.json').read_text())
    for key in ['checkpoint_metadata_sha256','base_config_sha256','train_sha256','classes',
                'fold','split_seed','checkpoint_trained_ids','fitting_ids','validation_ids']:
        if fm[key]!=vm[key]:raise ValueError('Partition manifests differ: '+key)
    fs=embedded_sources(args.fitting_source);vs=embedded_sources(args.validation_source)
    for name in ['k12/vision.py','k12/core.py']:
        if fs[name]!=vs[name]:raise ValueError('Encoder or split code differs: '+name)
    with np.load(fit/'base_head.npz',allow_pickle=False) as a,np.load(val/'base_head.npz',allow_pickle=False) as b:
        for key in ['weight','bias','classes']:
            if not np.array_equal(a[key],b[key]):raise ValueError('Frozen classifier differs: '+key)
    fm['cache_sources']={'fitting_manifest_sha256':sha(fit/'manifest.json'),
                         'validation_manifest_sha256':sha(val/'manifest.json'),
                         'fitting_features_sha256':sha(fit/'fitting_features.npz'),
                         'validation_features_sha256':sha(val/'validation_features.npz'),
                         'encoder_helpers_sha256':{n:hashlib.sha256(fs[n].encode()).hexdigest() for n in ['k12/vision.py','k12/core.py']}}
    out.mkdir(parents=True,exist_ok=True)
    for source in [fit/'fitting_features.npz',fit/'base_head.npz',val/'validation_features.npz']:
        shutil.copyfile(source,out/source.name)
    (out/'manifest.json').write_text(json.dumps(fm,indent=2));print(out.resolve())
