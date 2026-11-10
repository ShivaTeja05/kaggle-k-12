import hashlib, json, subprocess, sys, time
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
