import json
from pathlib import Path
root=Path('/kaggle/input')
report=[]
for path in root.rglob('config.json'):
 cfg=json.loads(path.read_text());print(json.dumps({'path':str(path),'config':cfg},indent=2))
 item={'path':str(path),'config':cfg}
 index=path.parent/'model.safetensors.index.json'
 if index.exists():
  m=json.loads(index.read_text())
  item['metadata']=m.get('metadata',{})
  item['vision_weight_count']=sum('visual' in k or 'vision' in k for k in m.get('weight_map',{}))
  item['weight_names_sample']=list(m.get('weight_map',{}))[:12]
 report.append(item)
Path('/kaggle/working/model_inspection.json').write_text(json.dumps(report,indent=2))
