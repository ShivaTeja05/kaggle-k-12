"""Private, offline feature extraction from our completed Qwen LoRA checkpoint."""
import argparse
import json
from pathlib import Path


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--username',required=True)
    parser.add_argument('--output',default='kaggle/adaptive-features')
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--fit-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    dst=Path(args.output);dst.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(root/'src')):p.read_text() for p in (root/'src/k12').glob('*.py')}
    arguments=['extract','--output','/kaggle/working/adaptive_features','--batch-size','2']
    if args.test_only:arguments+=['--test-only']
    if args.fit_only:arguments+=['--fit-only']
    code=("import os,sys,json,runpy\nfrom pathlib import Path\n"
          "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
          "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
          f"files=json.loads({json.dumps(json.dumps(files))})\n"
          "root=Path('/kaggle/working/source')\n"
          "for name,source in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
          "sys.path.insert(0,str(root))\n"
          f"sys.argv=['k12.adaptive']+{arguments!r}\nrunpy.run_module('k12.adaptive',run_name='__main__')\n")
    (dst/'extract.py').write_text(code)
    if args.test_only and args.fit_only:raise ValueError('Choose one extraction partition')
    slug='k12-adaptive-features'+('-test' if args.test_only else '-fitting' if args.fit_only else '')
    meta={'id':f'{args.username}/{slug}','title':slug.replace('-',' ').title(),
          'code_file':'extract.py','language':'python','kernel_type':'script','is_private':True,
          'enable_gpu':True,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
          'dataset_sources':[],'competition_sources':['misconception-detection-in-k-12-student-math-workings'],
          'kernel_sources':[f'{args.username}/k12-vision-development'],
          'model_sources':['qwen-lm/qwen-3-vl/transformers/4b-instruct/1']}
    (dst/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(dst.resolve())
