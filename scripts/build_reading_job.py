import argparse
import json
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--username',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--ddp',action='store_true')
    parser.add_argument('--budget-seconds',type=int,default=3500)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(root/'src')):p.read_text() for p in (root/'src/k12').glob('*.py')}
    arguments=['--smoke'] if args.smoke else ['--budget-seconds',str(args.budget_seconds)]
    source=("import os,sys,json,runpy\nfrom pathlib import Path\n"
            "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
            "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
            f"files=json.loads({json.dumps(json.dumps(files))})\n"
            "root=Path('/kaggle/working/source')\n"
            "for name,source in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
            "sys.path.insert(0,str(root))\n"
            +(f"import subprocess\nos.environ['PYTHONPATH']=str(root)\nsubprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','k12.reading_finetune']+{arguments!r},check=True)\n" if args.ddp else
              f"sys.argv=['k12.reading_finetune']+{arguments!r}\nrunpy.run_module('k12.reading_finetune',run_name='__main__')\n"))
    (out/'train.py').write_text(source)
    slug='k12-reading-supervision'+('-ddp' if args.ddp else '')+('-smoke' if args.smoke else '-development')
    metadata={'id':f'{args.username}/{slug}','title':slug.replace('-',' ').title(),'code_file':'train.py',
        'language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':[],
        'competition_sources':['misconception-detection-in-k-12-student-math-workings'],
        'kernel_sources':[f'{args.username}/k12-vision-development'],
        'model_sources':['qwen-lm/qwen-3-vl/transformers/4b-instruct/1']}
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2));print(out.resolve())
