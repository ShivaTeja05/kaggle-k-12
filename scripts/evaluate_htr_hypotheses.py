"""Test alternate formula readings and agreement on the fixed development photos."""
import argparse
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageOps

from evaluate_formula_reader import WORK, cohort, fidelity, normalize_latex, write_json


def recognize(ctc):
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(WORK / 'paddlex'))
    from paddleocr import FormulaRecognition, TextRecognition

    manifest = json.loads((WORK / 'manifest.json').read_text())['rows']
    start = time.monotonic()
    kind, name, directory = ((TextRecognition, 'latin_PP-OCRv5_mobile_rec', 'latin-ctc') if ctc else
                             (FormulaRecognition, 'PP-FormulaNet_plus-S', 'model'))
    reader = kind(model_name=name, model_dir=str(WORK / directory), device='cpu',
                  cpu_threads=3 if ctc else 4, enable_hpi=False, enable_mkldnn=False)
    rows = []
    for i, row in enumerate(manifest, 1):
        readings = []
        for path in row['paths']:
            if ctc:
                result = reader.predict(input=path, batch_size=1)[0]
                readings.append(result['rec_text'])
            else:
                with Image.open(path) as crop:
                    contrast = ImageOps.autocontrast(ImageOps.grayscale(crop), cutoff=1).convert('RGB')
                result = reader.predict(input=np.asarray(contrast), batch_size=1)[0]
                readings.append(normalize_latex(result['rec_formula']))
        rows.append({'trace_id': row['trace_id'], 'readings': readings})
        if i % 10 == 0:
            print(f'{name} read {i}/{len(manifest)} photos', flush=True)
    output = 'independent_ctc.json' if ctc else 'alternate_view.json'
    write_json(WORK / output, {'rows': rows, 'seconds': time.monotonic() - start})


def agreed_equation(left, right):
    from evaluate_working_verifier import arithmetic_value, math_form

    first = re.sub(r'\s*([+\-*/=()])\s*', r'\1', math_form(left))
    second = re.sub(r'\s*([+\-*/=()])\s*', r'\1', math_form(right))
    if first != second or '=' not in first:
        return ''
    if any(arithmetic_value(side) is None for side in first.split('=')):
        return ''
    return first


def cited_window(row):
    """Use teacher root locations for a diagnostic of the current verifier."""
    cited = {n for item in json.loads(row.misconceptions) if item.get('root') is True
             for n in re.findall(r'(?i)\bL(\d+)', str(item.get('step', '')))}
    lines = row.transcription_steps.splitlines()
    hits = [i for i, line in enumerate(lines)
            if (marker := re.match(r'^\s*L(\d+)\s*\|', line, re.I)) and marker.group(1) in cited]
    if not hits:
        return row.transcription_steps, False
    keep = set(hits) | {i - 1 for i in hits if i > 0}
    return '\n'.join(line if i in keep else '' for i, line in enumerate(lines)), True


def evaluate(with_ctc):
    from audit_ocr_reading_order import first_cited_equation, token_lines
    from evaluate_root_examples import combine_model_rankings, example_bank, score_transcripts
    from evaluate_working_verifier import rerank_root_codes, summarize
    from k12.core import aligned_bundle, label_space, read_csv

    fitting, validation, sample = cohort()
    ids = sample.trace_id.tolist()
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    dev_ids = validation.trace_id.tolist()
    original = json.loads((WORK / 'recognition.json').read_text())['rows']
    if [row['trace_id'] for row in original] != ids:
        raise ValueError('Original view does not align to the fixed photo sample')
    native_parts = [row['text'].split('\n') for row in original]
    transcripts = {'original': ['\n'.join(parts) for parts in native_parts]}
    controls = [('contrast', 'alternate_view.json')]
    if with_ctc:
        controls.append(('ctc', 'independent_ctc.json'))
    report = {'rows': len(sample), 'crops': sum(map(len, native_parts)), 'device': 'cpu',
              'scope': 'Multiple hypotheses from image perturbation or an independent CTC reader. '
                       'This reuses the development photos and frozen root-example verifier. '
                       'Agreement can preserve a repeated error. This is post-decoding comparison, '
                       'with no integration of CTC prefix probabilities into beam search.',
              'agreement_coverage': {}, 'either_reader_exact_cited_equations': {},
              'added_reader_wall_seconds': {}, 'rankings': {}}
    cited = [first_cited_equation(row) for row in sample.itertuples(index=False)]
    for name, filename in controls:
        additional = json.loads((WORK / filename).read_text())
        rows = additional['rows']
        if [row['trace_id'] for row in rows] != ids:
            raise ValueError(f'{name} does not align to the fixed photo sample')
        parts = [row['readings'] for row in rows]
        if any(len(a) != len(b) for a, b in zip(native_parts, parts)):
            raise ValueError(f'{name} must contain one hypothesis per original crop')
        # Empty regions break two-line evidence windows, avoiding artificial transitions.
        consensus = [[agreed_equation(a, b) for a, b in zip(left, right)]
                     for left, right in zip(native_parts, parts)]
        transcripts[name] = ['\n'.join(row) for row in parts]
        transcripts[f'{name}_agreement'] = ['\n'.join(row) for row in consensus]
        report['agreement_coverage'][name] = {
            'photos_with_an_agreed_equation': sum(any(row) for row in consensus),
            'agreed_equations': sum(bool(eq) for row in consensus for eq in row)}
        both = [token_lines(a) + token_lines(b) for a, b in zip(transcripts['original'], transcripts[name])]
        report['either_reader_exact_cited_equations'][name] = sum(eq is not None and eq in lines
                                                                for eq, lines in zip(cited, both))
        report['added_reader_wall_seconds'][name] = additional['seconds']
    report['recognition'] = {name: fidelity(sample, text) for name, text in transcripts.items()}
    image = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', dev_ids, classes)
    large = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz', dev_ids, classes)
    bank = example_bank(fitting, classes)
    evidence = {name: score_transcripts(text, *bank, len(classes))[0] for name, text in transcripts.items()}
    for name, _ in controls:
        evidence[f'{name}_or_original'] = np.maximum(evidence['original'], evidence[name])
    full_models = {'qwen4b': image, 'fixed_fusion': combine_model_rankings(image, large, classes)}
    windows, cited_flags = zip(*(cited_window(row) for row in validation.itertuples(index=False)))
    report['teacher_location_diagnostic'] = {'rows': len(validation), 'rows_with_cited_root': sum(cited_flags),
                                             'scope': 'Uses gold root locations at evaluation. Rows without '
                                                      'a cited root retain their full teacher transcript.'}
    teacher_evidence = {name: score_transcripts(text, *bank, len(classes))[0]
                        for name, text in [('full_teacher', validation.transcription_steps), ('cited_window', windows)]}
    for model, full_scores in full_models.items():
        base = full_scores[sample.index]
        ranks = {name: rerank_root_codes(base, base * np.exp(2. * values), classes.index('NONE'))
                 for name, values in evidence.items()}
        report['rankings'][model] = {'image_only': summarize(sample, base, classes, base)}
        for name, scores in ranks.items():
            metrics = summarize(sample, scores, classes, base)
            paired = summarize(sample, scores, classes, ranks['original'])
            metrics['delta_vs_original_view'] = paired['delta_vs_image']
            metrics['paired_delta_vs_original_view_95'] = paired['paired_question_bootstrap_delta_95']
            report['rankings'][model][name] = metrics
            print(f'{model} {name} {metrics["map3"]:.5f}')
        gold_ranks = {name: rerank_root_codes(full_scores, full_scores * np.exp(2. * values), classes.index('NONE'))
                      for name, values in teacher_evidence.items()}
        report['teacher_location_diagnostic'][model] = {
            name: summarize(validation, scores, classes, gold_ranks['full_teacher'])
            for name, scores in gold_ranks.items()}
    write_json(Path('reports/htr_hypothesis_pilot.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('recognize', 'recognize-ctc', 'evaluate'))
    parser.add_argument('--with-ctc', action='store_true')
    args = parser.parse_args()
    if args.stage == 'evaluate':
        evaluate(args.with_ctc)
    else:
        recognize(ctc=args.stage == 'recognize-ctc')
