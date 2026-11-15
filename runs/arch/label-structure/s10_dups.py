"""Exact duplicate photos (md5) across train/test/unlabelled."""
import hashlib, json, collections
from pathlib import Path
base = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/data/bundle/x/competition_data/images')
H = collections.defaultdict(list)
for split in sorted(p.name for p in base.iterdir() if p.is_dir()):
    for f in (base/split).iterdir():
        if f.suffix.lower() in ('.jpeg', '.jpg', '.png'):
            H[hashlib.md5(f.read_bytes()).hexdigest()].append((split, f.stem))
counts = collections.Counter(s for v in H.values() for s, _ in v)
dup = [v for v in H.values() if len(v) > 1]
pairs = collections.Counter(tuple(sorted({s for s, _ in v})) for v in dup)
out = dict(files=dict(counts), dup_groups=len(dup), dup_split_combos={'+'.join(k): n for k, n in pairs.items()},
           test_rows_with_train_duplicate=sum(1 for v in dup for s, t in v if s == 'test' and any(x == 'train' for x, _ in v)))
print(json.dumps(out, indent=1))
json.dump(out, open('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure/duplicates.json', 'w'), indent=1)
