"""Create a CPU-only Kaggle job that prepares versioned offline inference wheels."""
import argparse
import json
from pathlib import Path


SOURCE=r'''import hashlib, json, subprocess, sys, time
from pathlib import Path
out=Path('/kaggle/working/wheels');out.mkdir(exist_ok=True)
wheel='https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
cmd=[sys.executable,'-m','pip','download','--only-binary=:all:','--dest',str(out),wheel,
     '--extra-index-url','https://download.pytorch.org/whl/cu129']
print('CPU-only dependency download; no competition data or credentials included',flush=True)
start=time.monotonic();subprocess.run(cmd,check=True)
files=[]
for p in sorted(out.glob('*.whl')):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 files.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
manifest={'python':sys.version,'seconds':time.monotonic()-start,'files':files,
          'total_bytes':sum(x['bytes'] for x in files),'command':cmd}
Path('/kaggle/working/wheel_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'stage':'WHEELS_READY','count':len(files),'GB':manifest['total_bytes']/1e9}),flush=True)
'''


def build(username,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'prepare.py').write_text(SOURCE)
    meta={'id':f'{username}/k12-offline-vllm-wheels','title':'K12 Offline Vllm Wheels',
          'code_file':'prepare.py','language':'python','kernel_type':'script','is_private':True,
          'enable_gpu':False,'enable_internet':True,'dataset_sources':[],
          'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    print(folder.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--username',required=True);p.add_argument('--output',default='kaggle/wheels')
    a=p.parse_args();build(a.username,a.output)
