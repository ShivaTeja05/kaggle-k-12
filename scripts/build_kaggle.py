"""Bundle only our source into a private offline Kaggle notebook."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(output, username, smoke=False, recipe='classifier', limit=0, model='qwen3-vl-4b', evidence=False, test=False, fold=0, validate_then_test=False, structured_codes=False, structured_evidence=False, min_validation_map3=.76, budget_seconds=0):
    dst=Path(output);dst.mkdir(parents=True,exist_ok=True)
    files={str(p.relative_to(ROOT/'src')):p.read_text() for p in (ROOT/'src/k12').glob('*.py')}
    if recipe=='classifier':
        arguments=['--smoke','--predict-test'] if smoke else ['--epochs','3','--predict-test']
        module='k12.vision'
    else:
        arguments=['--limit',str(4 if smoke else limit)] if smoke or limit else []
        if evidence:arguments+=['--evidence','--max-new-tokens','192']
        if structured_codes:
            if recipe!='vllm' or evidence:raise ValueError('Structured codes require code-only vLLM')
            arguments+=['--structured-codes']
        if structured_evidence:
            if recipe!='vllm' or not evidence or structured_codes:raise ValueError('Structured evidence requires evidence-mode vLLM')
            arguments+=['--structured-evidence']
        if model=='qwen35-27b-int4' and recipe=='vllm':arguments+=['--eager']
        module='k12.vllm_ranker' if recipe=='vllm' else 'k12.taxonomy_ranker'
        if test:arguments+=['--test']
        arguments+=['--fold',str(fold)]
        if validate_then_test:
            if recipe!='vllm':raise ValueError('Validation gate is implemented for the batched VLM runner')
            arguments+=['--validate-then-test','--min-validation-map3',str(min_validation_map3)]
            if budget_seconds:arguments+=['--budget-seconds',str(budget_seconds)]
    installation=''
    if recipe=='vllm':
        installation=("import subprocess\n"
            "wheel_dirs=sorted({p.parent for p in Path('/kaggle/input').rglob('vllm-*.whl')})\n"
            "if len(wheel_dirs)!=1:raise RuntimeError('Attach the completed private k12-offline-vllm-wheels output')\n"
            "subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(wheel_dirs[0]),'vllm==0.31.0'],check=True)\n"
            "for module_name in ['torchaudio','torchcodec']:\n"
            " probe=subprocess.run([sys.executable,'-c','import '+module_name],capture_output=True,text=True)\n"
            " if probe.returncode:subprocess.run([sys.executable,'-m','pip','uninstall','-y',module_name],check=True)\n")
    bootstrap=("import os, sys, json, runpy\nfrom pathlib import Path\n"
               "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\n"
               "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
               "os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'\n"
               +installation+
               f"files=json.loads({json.dumps(json.dumps(files))})\n"
               "root=Path('/kaggle/working/source')\n"
               "for name,source in files.items():\n p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
               "sys.path.insert(0,str(root))\n"
               f"sys.argv=[{module!r}]+{arguments!r}\n"
               f"runpy.run_module({module!r},run_name='__main__')\n")
    note={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
          'cells':[{'id':'purpose','cell_type':'markdown','metadata':{},'source':[
              '# K12 image classifier development\nPrivate, offline, free GPU experiment. No external APIs.\n',
              'Validation uses held-out images and test-available context only. Test predictions are not submitted automatically.\n']},
              {'id':'run','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[bootstrap]}]}
    (dst/'train.ipynb').write_text(json.dumps(note,indent=1))
    slug=('k12-vision' if recipe=='classifier' else 'k12-taxonomy')+('-smoke' if smoke else '-development')
    if model in {'qwen35-9b','qwen35-27b-int4'}:
        slug=('k12-qwen35' if model=='qwen35-9b' else 'k12-qwen35-27b-int4')+('-pilot' if limit else '-development')
    if recipe=='vllm':slug+='-vllm'
    if validate_then_test:slug+='-gated'
    if test:slug+='-test'
    elif fold:slug+=f'-fold{fold}'
    model_source={'qwen3-vl-4b':'qwen-lm/qwen-3-vl/transformers/4b-instruct/1',
                  'qwen35-9b':'shelterw/qwen3.5/transformers/qwen3.5-9b/1',
                  'qwen35-27b-int4':'boristown/qwen3-5-27b-awq-4bit/transformers/default/1'}[model]
    meta={'id':f'{username}/{slug}','title':slug.replace('-',' ').title(),'code_file':'train.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
          'machine_shape':'NvidiaTeslaT4','dataset_sources':[],
          'competition_sources':['misconception-detection-in-k-12-student-math-workings'],
          'kernel_sources':[f'{username}/k12-offline-vllm-wheels'] if recipe=='vllm' else [],'model_sources':[model_source]}
    (dst/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(dst.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='kaggle/train');p.add_argument('--username',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--recipe',choices=['classifier','taxonomy','vllm'],default='classifier');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--model',choices=['qwen3-vl-4b','qwen35-9b','qwen35-27b-int4'],default='qwen3-vl-4b');p.add_argument('--evidence',action='store_true')
    p.add_argument('--test',action='store_true');p.add_argument('--fold',type=int,default=0)
    p.add_argument('--validate-then-test',action='store_true')
    p.add_argument('--structured-codes',action='store_true')
    p.add_argument('--structured-evidence',action='store_true');p.add_argument('--min-validation-map3',type=float,default=.76)
    p.add_argument('--budget-seconds',type=int,default=0)
    a=p.parse_args();build(a.output,a.username,a.smoke,a.recipe,a.limit,a.model,a.evidence,a.test,a.fold,a.validate_then_test,a.structured_codes,a.structured_evidence,a.min_validation_map3,a.budget_seconds)
