"""Compare CPU OCR on the same 60 labelled development photos under three views."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from k12.core import make_folds, read_csv


def math_tokens(text):
    text = re.sub(r'(?im)^\s*L\d+\s*\|', '', text)
    text = re.sub(r'\[[^\]\n]*(?:crossed[ -]out|unclear|illegible)[^\]\n]*\]', '', text, flags=re.I)
    text = re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'\1/\2', text)
    for source, target in [('\\times', '×'), ('\\cdot', '×'), ('\\div', '÷'), ('−', '-'), ('\\pm', '±')]:
        text = text.replace(source, target)
    return re.findall(r'\d+(?:[.,]\d+)?|[+\-×÷=/<>±^√%]', text)


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def ocr(image):
    payload = BytesIO()
    image.save(payload, format='PNG')
    start = time.monotonic()
    result = subprocess.run(['tesseract', 'stdin', 'stdout', '-l', 'eng', '--psm', '11'],
                            input=payload.getvalue(), capture_output=True, check=True, timeout=25,
                            env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
    return result.stdout.decode('utf-8', errors='replace'), time.monotonic() - start


def evaluate_row(row, files):
    with Image.open(files[row.image_file]) as raw:
        original = ImageOps.exif_transpose(raw).convert('RGB')
    reduced = original.copy()
    reduced.thumbnail((1024, 1024))
    contrast = ImageOps.autocontrast(ImageOps.grayscale(reduced), cutoff=1)
    gold = math_tokens(row.transcription_steps)
    gold_numbers = Counter(token for token in gold if token[0].isdigit())
    results = {}
    for variant, image in [('long_side_1024', reduced), ('native_resolution', original),
                           ('long_side_1024_autocontrast', contrast)]:
        text, elapsed = ocr(image)
        observed = math_tokens(text)
        found_numbers = Counter(token for token in observed if token[0].isdigit())
        results[variant] = {'edits': edit_distance(gold, observed), 'gold_tokens': len(gold),
                            'gold_numbers': sum(gold_numbers.values()), 'observed_numbers': sum(found_numbers.values()),
                            'matched_numbers': sum((gold_numbers & found_numbers).values()),
                            'empty_math_tokens': int(not observed), 'seconds': elapsed}
    return original.size, results


def main():
    train = read_csv('data/raw/train.csv')
    sample = train.loc[make_folds(train) == 0].sample(n=60, random_state=2026)
    files = {p.name: p for p in Path('data/cpu-image-audit').rglob('*.jpeg')}
    if not set(sample.image_file) <= files.keys():
        raise ValueError('The fixed development photo sample is incomplete')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda row: evaluate_row(row, files), sample.itertuples(index=False)))
    sizes = np.asarray([size for size, _ in results])
    variants = {}
    for name in results[0][1]:
        rows = [r[name] for _, r in results]
        totals = {key: sum(row[key] for row in rows) for key in rows[0]}
        variants[name] = {'math_token_error_rate': totals['edits'] / totals['gold_tokens'],
                          'number_recall': totals['matched_numbers'] / totals['gold_numbers'],
                          'number_precision': totals['matched_numbers'] / max(1, totals['observed_numbers']),
                          'empty_math_token_rows': totals['empty_math_tokens'],
                          'median_ocr_seconds': float(np.median([row['seconds'] for row in rows]))}
    report = {'rows': len(sample), 'fold': 0, 'sample_seed': 2026,
              'ocr_engine': subprocess.check_output(['tesseract', '--version'], text=True).splitlines()[0],
              'language_model': 'eng', 'page_segmentation_mode': 11, 'variants': variants,
              'images_above_1024_long_side': int(sum(sizes.max(axis=1) > 1024)),
              'images_above_qwen_pixel_budget_before_processor': int(sum(sizes.prod(axis=1) > 768 * 768)),
              'median_long_side': float(np.median(sizes.max(axis=1))),
              'median_megapixels': float(np.median(sizes.prod(axis=1)) / 1e6),
              'gold_math_token_count': sum(r['long_side_1024']['gold_tokens'] for _, r in results),
              'interpretation': 'Symbol and number token edit distance includes reading order and limited LaTeX normalization. English Tesseract is a CPU baseline. No Qwen OCR or MAP@3 result is measured here.'}
    Path('reports/cpu_ocr_image_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
