"""CPU-only: bundle train/test photos and question diagrams for the private RunPod run."""
import tarfile
from pathlib import Path

hits = [p.parent for p in Path('/kaggle/input').rglob('train.csv') if (p.parent / 'taxonomy_v1.csv').exists()]
assert len(hits) == 1, hits
root = hits[0]
with tarfile.open('/kaggle/working/k12_images.tar', 'w') as tar:
    for name in ['train.csv', 'test.csv', 'taxonomy_v1.csv']:
        tar.add(root / name, arcname=f'competition_data/{name}')
    for split in ['train', 'test', 'questions']:
        folder = root / 'images' / split
        tar.add(folder, arcname=f'competition_data/images/{split}')
        print(split, sum(1 for _ in folder.iterdir()), flush=True)
print('BUNDLE_DONE', Path('/kaggle/working/k12_images.tar').stat().st_size, flush=True)
