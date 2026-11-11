"""CPU-only: bundle the unlabelled student photos for private pseudo-labelling on RunPod."""
import tarfile
from pathlib import Path

# The canonical copy sits in competition_data/; a legacy duplicate sits at the dataset root.
hits = [p.parent for p in Path('/kaggle/input').rglob('unlabelled.csv') if p.parent.name == 'competition_data']
assert len(hits) == 1, hits
root = hits[0]
folders = [p for p in (root / 'images').iterdir() if p.is_dir()]
print('image folders', [(p.name, sum(1 for _ in p.iterdir())) for p in folders], flush=True)
with tarfile.open('/kaggle/working/k12_unlabelled.tar', 'w') as tar:
    tar.add(root / 'unlabelled.csv', arcname='competition_data/unlabelled.csv')
    for folder in folders:
        if folder.name not in {'train', 'test', 'questions'}:
            tar.add(folder, arcname=f'competition_data/images/{folder.name}')
print('BUNDLE_DONE', Path('/kaggle/working/k12_unlabelled.tar').stat().st_size, flush=True)
