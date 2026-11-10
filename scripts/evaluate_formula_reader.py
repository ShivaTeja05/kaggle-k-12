"""Compare page OCR and two readers on identical automatic crops of 60 development photos."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps


WORK = Path('artifacts/formula-ocr')
ARMS = ('page_tesseract', 'region_tesseract', 'region_formula')
REVISION = '3d46f557e3a1752f4bf81202395af3b5ecfadfd2'


def regions(image):
    gray = ImageOps.grayscale(image)
    width, height = gray.size
    pixels = np.asarray(gray, dtype=np.int16)
    background = np.asarray(gray.filter(ImageFilter.BoxBlur(9)), dtype=np.int16)
    ink = background - pixels > 20
    margin = max(1, width // 40)
    ink[:, :margin] = ink[:, -margin:] = False
    counts = ink.sum(axis=1)
    active = np.flatnonzero((counts >= max(4, width // 80)) & (counts < width * .45))
    spans = []
    for band in np.split(active, np.flatnonzero(np.diff(active) > max(5, height // 100)) + 1):
        if len(band) >= 4:
            spans.append((int(band[0]), int(band[-1]) + 1))
    if not spans:
        return [image]
    while len(spans) > 6:
        i = min(range(len(spans) - 1), key=lambda j: spans[j + 1][0] - spans[j][1])
        spans[i:i + 2] = [(spans[i][0], spans[i + 1][1])]
    crops = []
    for i, (top, bottom) in enumerate(spans):
        pad = max(8, height // 80)
        upper = (spans[i - 1][1] + top) // 2 if i else 0
        lower = (bottom + spans[i + 1][0]) // 2 if i + 1 < len(spans) else height
        top, bottom = max(upper, top - pad), min(lower, bottom + pad)
        columns = np.flatnonzero(ink[top:bottom].sum(axis=0) >= 2)
        left = max(0, int(columns[0]) - pad) if len(columns) else 0
        right = min(width, int(columns[-1]) + 1 + pad) if len(columns) else width
        crops.append(image.crop((left, top, right, bottom)))
    return crops


def cohort():
    from k12.core import make_folds, read_csv

    train = read_csv('data/raw/train.csv')
    folds = make_folds(train)
    development = train.loc[folds == 0].reset_index(drop=True)
    return train.loc[folds != 0], development, development.sample(n=60, random_state=2026)


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def prepare():
    from audit_cpu_ocr import ocr

    _, _, sample = cohort()
    files = {path.name: path for path in Path('data/cpu-image-audit').rglob('*.jpeg')}
    if not set(sample.image_file) <= files.keys():
        raise ValueError('The fixed 60-photo development sample is incomplete')
    (WORK / 'crops').mkdir(parents=True, exist_ok=True)
    # Invalidate any recognition result when rebuilding the crop manifest.
    (WORK / 'recognition.json').unlink(missing_ok=True)

    def prepare_photo(item):
        number, row = item
        with Image.open(files[row.image_file]) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        image.thumbnail((1024, 1024))
        page, page_seconds = ocr(image)
        paths, parts, seconds = [], [], 0.
        for index, crop in enumerate(regions(image)):
            path = WORK / 'crops' / f'{number:02d}-{index}.png'
            crop.save(path)
            paths.append(str(path))
            transcript, elapsed = ocr(crop)
            parts.append(transcript.strip())
            seconds += elapsed
        return {'trace_id': row.trace_id, 'paths': paths,
                'page_tesseract': {'text': page, 'seconds': page_seconds},
                'region_tesseract': {'text': '\n'.join(parts), 'seconds': seconds}}

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(prepare_photo, enumerate(sample.itertuples(index=False))))
    seconds = time.monotonic() - start
    write_json(WORK / 'manifest.json', {'rows': rows, 'prepare_wall_seconds': seconds})
    print(f'Prepared {len(rows)} photos and {sum(len(row["paths"]) for row in rows)} crops in {seconds:.1f}s', flush=True)


def normalize_latex(text):
    # Remove presentation-only markup while preserving numbers, signs and formulas.
    text = re.sub(r'\\(?:left|right|displaystyle|textstyle)\b|\\[,;! ]', '', text)
    text = re.sub(r'\\(?:dfrac|tfrac)\b', lambda _: r'\frac', text)
    return re.sub(r'\s+(?=[}=])|(?<=[{=])\s+', '', text).strip()


def recognize(smoke):
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(WORK / 'paddlex'))
    from paddleocr import FormulaRecognition

    rows = json.loads((WORK / 'manifest.json').read_text())['rows']
    start = time.monotonic()
    reader = FormulaRecognition(model_name='PP-FormulaNet_plus-S', model_dir=str(WORK / 'model'),
                                device='cpu', cpu_threads=4, enable_hpi=False, enable_mkldnn=False)
    load_seconds = time.monotonic() - start
    print(f'Loaded offline CPU reader in {load_seconds:.1f}s', flush=True)
    results = []
    for i, row in enumerate(rows[:1] if smoke else rows, 1):
        photo_start = time.monotonic()
        parts = []
        for path in row['paths']:
            result = next(iter(reader.predict(input=path, batch_size=1)))
            parts.append(normalize_latex(result['rec_formula']))
        results.append({'trace_id': row['trace_id'], 'text': '\n'.join(parts),
                        'seconds': time.monotonic() - photo_start})
        if smoke or i % 10 == 0:
            print(f'Read {i}/{1 if smoke else len(rows)} photos in {time.monotonic() - start:.1f}s', flush=True)
    if not smoke:
        write_json(WORK / 'recognition.json', {'rows': results, 'model_load_seconds': load_seconds,
                                               'recognition_wall_seconds': time.monotonic() - start})


def fidelity(sample, transcripts):
    from audit_cpu_ocr import edit_distance, math_tokens
    from audit_ocr_reading_order import first_cited_equation, token_lines

    totals = Counter()
    for row, prediction in zip(sample.itertuples(index=False), transcripts):
        gold, observed = math_tokens(row.transcription_steps), math_tokens(prediction)
        numbers = Counter(t for t in gold if t[0].isdigit())
        observed_numbers = Counter(t for t in observed if t[0].isdigit())
        cited = first_cited_equation(row)
        totals.update(gold_tokens=len(gold), edits=edit_distance(gold, observed),
                      gold_numbers=sum(numbers.values()), observed_numbers=sum(observed_numbers.values()),
                      matched_numbers=sum((numbers & observed_numbers).values()),
                      matched_tokens=sum((Counter(gold) & Counter(observed)).values()),
                      empty_math_rows=int(not observed), cited_equations=int(cited is not None),
                      exact_cited_equations=int(cited is not None and cited in token_lines(prediction)))
    return {'number_recall': totals['matched_numbers'] / totals['gold_numbers'],
            'number_precision': totals['matched_numbers'] / max(1, totals['observed_numbers']),
            'math_token_error_rate': totals['edits'] / totals['gold_tokens'],
            'order_free_math_token_recall': totals['matched_tokens'] / totals['gold_tokens'],
            'counts': dict(totals)}


def evaluate(controls_only):
    from evaluate_root_examples import combine_model_rankings, example_bank, score_transcripts
    from evaluate_working_verifier import rerank_root_codes, summarize
    from k12.core import aligned_bundle, label_space, read_csv, topk

    fitting, development, sample = cohort()
    ids = sample.trace_id.tolist()
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    dev_ids = development.trace_id.tolist()
    image = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', dev_ids, classes)
    large = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz', dev_ids, classes)
    baselines = {'qwen4b': image[sample.index],
                 'qwen4b_27b_fusion': combine_model_rankings(image, large, classes)[sample.index]}
    manifest = json.loads((WORK / 'manifest.json').read_text())
    if [row['trace_id'] for row in manifest['rows']] != ids:
        raise ValueError('Crop manifest does not align to the fixed development sample')
    arms = {name: [row[name] for row in manifest['rows']] for name in ARMS[:2]}
    report = {'rows': len(sample), 'fold': 0, 'sample_seed': 2026, 'device': 'cpu',
              'root_error_rows': int(sum(sample.root_code != 'NONE')),
              'image_long_side_limit': 1024, 'region_limit_per_photo': 6,
              'tesseract': {'language': 'eng', 'page_segmentation_mode': 11},
              'crops': sum(len(row['paths']) for row in manifest['rows']),
              'prepare_wall_seconds': manifest['prepare_wall_seconds'],
              'scope': 'Reused labeled development sample. Region proposals use image pixels only. '
                       'Frozen root-example reranker uses fitting rows only. 27B predictions are development-only.'}
    if not controls_only:
        results = json.loads((WORK / 'recognition.json').read_text())
        if [row['trace_id'] for row in results['rows']] != ids:
            raise ValueError('Formula predictions do not align to the fixed development sample')
        arms['region_formula'] = results['rows']
        report['formula_reader'] = {'model': 'PaddlePaddle/PP-FormulaNet_plus-S', 'revision': REVISION,
                                    'cpu_threads': 4,
                                    'model_load_seconds': results['model_load_seconds'],
                                    'wall_seconds': results['recognition_wall_seconds']}
    bank = example_bank(fitting, classes)
    truth = sample.root_code.to_numpy()
    root_rows = truth != 'NONE'

    def ranking_metrics(scores, reference):
        metrics = summarize(sample, scores, classes, reference)
        first = topk(scores, classes)[:, 0]
        metrics['root_error_top1_accuracy'] = float(np.mean(first[root_rows] == truth[root_rows]))
        return metrics

    report['baselines'] = {name: ranking_metrics(scores, scores)
                           for name, scores in baselines.items()}
    report['arms'] = {}
    controls = {}
    for name, entries in arms.items():
        transcripts = [entry['text'] for entry in entries]
        recognition = fidelity(sample, transcripts)
        recognition['median_seconds_per_photo'] = float(np.median([entry['seconds'] for entry in entries]))
        rankings, arm_scores = {}, {}
        positive, _ = score_transcripts(transcripts, *bank, len(classes))
        for model, scores in baselines.items():
            reranked = rerank_root_codes(scores, scores * np.exp(2. * positive), classes.index('NONE'))
            metrics = ranking_metrics(reranked, scores)
            for control_name, control in controls.items():
                paired = summarize(sample, reranked, classes, control[model])
                metrics[f'vs_{control_name}'] = {
                    'delta': paired['delta_vs_image'], 'rows_improved': paired['rows_improved'],
                    'rows_regressed': paired['rows_regressed'],
                    'paired_question_bootstrap_delta_95': paired['paired_question_bootstrap_delta_95']}
            rankings[model] = metrics
            arm_scores[model] = reranked
        controls[name] = arm_scores
        report['arms'][name] = {'recognition': recognition, 'rankings': rankings}
    output = WORK / 'controls.json' if controls_only else Path('reports/formula_reader_pilot.json')
    write_json(output, report)
    for name, arm in report['arms'].items():
        print(f'{name} number recall={arm["recognition"]["number_recall"]:.4f} '
              f'4B MAP@3={arm["rankings"]["qwen4b"]["map3"]:.5f} '
              f'fusion MAP@3={arm["rankings"]["qwen4b_27b_fusion"]["map3"]:.5f}')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'smoke', 'recognize', 'controls', 'evaluate'))
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
    elif stage in ('smoke', 'recognize'):
        recognize(smoke=stage == 'smoke')
    else:
        evaluate(controls_only=stage == 'controls')


if __name__ == '__main__':
    main()
