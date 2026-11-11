"""Aggregate dataset audit for training-data fixes: duplicates, photo geometry, label structure.

Reads the private image bundle; writes only counts to reports/dataset_fixes_audit.json.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from k12.core import read_csv


def dhash(image, size=16):
    small = ImageOps.exif_transpose(image).convert('L').resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    return np.packbits((pixels[:, 1:] > pixels[:, :-1]).ravel())


def describe(folder, names):
    rows = []
    for name in names:
        path = folder / name
        data = path.read_bytes()
        with Image.open(path) as image:
            orientation = image.getexif().get(274, 1)
            width, height = ImageOps.exif_transpose(image).size
            rows.append({'name': name, 'md5': hashlib.md5(data).hexdigest(), 'hash': dhash(image),
                         'orientation': orientation, 'width': width, 'height': height})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/bundle/x/competition_data')
    args = parser.parse_args()
    root = Path(args.root)
    train, test = read_csv(root / 'train.csv'), read_csv(root / 'test.csv')
    left = describe(root / 'images/train', train.image_file)
    right = describe(root / 'images/test', test.image_file)

    md5_train = Counter(r['md5'] for r in left)
    exact_test_in_train = sum(r['md5'] in md5_train for r in right)
    hashes_train = np.stack([r['hash'] for r in left])
    hashes_test = np.stack([r['hash'] for r in right])
    bits_train = np.unpackbits(hashes_train, axis=1).astype(np.int8)
    bits_test = np.unpackbits(hashes_test, axis=1).astype(np.int8)
    distance = (bits_test[:, None, :] != bits_train[None, :, :]).sum(axis=2)
    nearest = distance.min(axis=1)
    within_train = (bits_train[:, None, :] != bits_train[None, :, :]).sum(axis=2)
    np.fill_diagonal(within_train, 999)

    findings = [json.loads(text.replace('NaN', 'null')) for text in train.misconceptions]
    major = [[f for f in row if f.get('severity') == 'major'] for row in findings]
    secondary = [{f['category'] for f in row if not f.get('root')} - {code}
                 for row, code in zip(major, train.root_code)]
    geometry = left + right
    report = {
        'train_rows': len(train), 'test_rows': len(test),
        'duplicates': {
            'exact_duplicate_train_photos': int(sum(c - 1 for c in md5_train.values() if c > 1)),
            'test_photos_identical_to_a_train_photo': int(exact_test_in_train),
            'test_nearest_train_dhash_distance_quantiles': np.quantile(nearest, [0, .01, .05, .5]).tolist(),
            'test_photos_within_10_bits_of_train': int((nearest <= 10).sum()),
            'train_pairs_within_10_bits': int((within_train <= 10).sum() // 2),
        },
        'geometry': {
            'exif_rotated_photos': int(sum(r['orientation'] not in (1, None) for r in geometry)),
            'landscape_photos': int(sum(r['width'] > r['height'] for r in geometry)),
            'long_side_quantiles': np.quantile([max(r['width'], r['height']) for r in geometry], [0, .1, .5, .9, 1]).tolist(),
            'train_landscape_share': float(np.mean([r['width'] > r['height'] for r in left])),
            'test_landscape_share': float(np.mean([r['width'] > r['height'] for r in right])),
        },
        'labels': {
            'rows_with_secondary_major_codes': int(sum(bool(s) for s in secondary)),
            'secondary_codes_total': int(sum(len(s) for s in secondary)),
            'root_rows_with_secondary': int(sum(bool(s) for s, c in zip(secondary, train.root_code) if c != 'NONE')),
            'none_rows_with_major_findings': int(sum(bool(m) for m, c in zip(major, train.root_code) if c == 'NONE')),
            'codes_with_at_most_2_examples': int(sum(v <= 2 for v in train.root_code.value_counts())),
        },
    }
    Path('reports/dataset_fixes_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
