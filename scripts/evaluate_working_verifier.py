"""Compare literal and structured working evidence on a fixed development split."""
import ast
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from audit_cpu_ocr import ocr
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk


NUMBER = re.compile(r'(?<!\w)\d+(?:[.,]\d+)?')
ARITHMETIC = re.compile(r'[0-9.()+*/\-\s]+')
LINE_ID = re.compile(r'(?im)^[ \t]*L\d+\s*\|[ \t]*')
ANNOTATION = re.compile(r'\[[^\]\n]*\]')


def working_lines(text):
    lines = []
    for line in LINE_ID.sub('', unicodedata.normalize('NFC', str(text))).splitlines():
        if re.search(r'\[[^\]]*(?:crossed[ -]out|unclear|illegible)[^\]]*\]', line, re.I):
            continue
        line = ANNOTATION.sub('', line).strip()
        if line:
            lines.append(line)
    return lines


def math_form(text):
    text = re.sub(r'\\(?:d?frac)\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}', r'(\1/\2)', text)
    for source, target in [('−', '-'), ('×', '*'), ('\\times', '*'), ('\\cdot', '*'),
                           ('÷', '/'), ('\\div', '/'), ('\\left', ''), ('\\right', ''), ('$','')]:
        text = text.replace(source, target)
    return text.strip()


def evaluate_node(node, expression):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(ast.get_source_segment(expression, node))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = evaluate_node(node.operand, expression)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = evaluate_node(node.left, expression)
        right = evaluate_node(node.right, expression)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError('Unsupported expression')


def arithmetic_value(text):
    expression = math_form(text)
    if not expression or len(expression) > 100 or not ARITHMETIC.fullmatch(expression):
        return None
    if '**' in expression or '//' in expression:
        return None
    try:
        return evaluate_node(ast.parse(expression, mode='eval').body, expression)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return None


def evidence_views(transcript, answer):
    lines = working_lines(transcript)
    patterns = []
    checks = []
    verified = 0
    for line in lines:
        form = math_form(line)
        # Keep operators and variables while sharing number patterns across problems.
        patterns.append(NUMBER.sub(' N ', form))
        if '=' in form:
            values = [arithmetic_value(side) for side in form.split('=')]
            if any(v is None for v in values):
                checks.append('eq_unknown')
            else:
                verified += 1
                checks.append('eq_valid' if len(set(values)) == 1 else 'eq_broken')
    final = arithmetic_value(lines[-1].split('=')[-1]) if lines else None
    key = arithmetic_value(str(answer))
    if final is not None and key is not None:
        checks.append('numeric_key_match' if final == key else 'numeric_key_mismatch')
    return ' '.join(lines), ' | '.join(patterns), ' '.join(checks), verified


def fit_verifiers(fit, evaluations, classes):
    inputs = [fit, *evaluations]
    views = [list(zip(*(evidence_views(text, key) for text, key in zip(frame.work, frame.correct_answer))))
             for frame in inputs]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=15000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=24000, sublinear_tf=True)
    patterns = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=3, max_features=16000, sublinear_tf=True)
    checks = TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r'\S+', sublinear_tf=True)
    word.fit(views[0][0])
    char.fit(views[0][0])
    patterns.fit(views[0][1])
    checks.fit(views[0][2])
    literal = [hstack([word.transform(v[0]), char.transform(v[0])], format='csr') for v in views]
    structured = [hstack([x, patterns.transform(v[1]), checks.transform(v[2])], format='csr')
                  for x, v in zip(literal, views)]
    outputs = {}
    all_rows = np.ones(len(fit), dtype=bool)
    errors = fit.root_code.ne('NONE').to_numpy()
    for name, matrices, selected in [('literal', literal, all_rows), ('structured', structured, all_rows),
                                     ('root_only', structured, errors)]:
        model = LogisticRegression(C=4, max_iter=500).fit(matrices[0][selected], fit.root_code.iloc[selected])
        indices = [classes.index(code) for code in model.classes_]
        outputs[name] = []
        for matrix in matrices[1:]:
            p = np.full((matrix.shape[0], len(classes)), 1e-12)
            p[:, indices] = model.predict_proba(matrix)
            outputs[name].append(normalize(p))
    coverage = [{'nonempty_work': sum(bool(s) for s in v[0]),
                 'rows_with_checked_numeric_equality': sum(n > 0 for n in v[3])} for v in views]
    return outputs, coverage


def shortlist_mask(reference, shortlist):
    selected = np.argsort(-reference, axis=1, kind='stable')[:, :shortlist]
    mask = np.zeros_like(reference, dtype=bool)
    np.put_along_axis(mask, selected, True, axis=1)
    return mask


def rerank(reference, evidence, weight=.35, shortlist=8):
    mask = shortlist_mask(reference, shortlist)
    return (1 - weight) * normalize(reference * mask) + weight * normalize(evidence * mask)


def rerank_root_codes(reference, evidence, none_index, weight=.35, shortlist=8):
    mask = shortlist_mask(reference, shortlist)
    mask[:, none_index] = False
    error_mass = (reference * mask).sum(axis=1, keepdims=True)
    proposed = normalize(evidence * mask) * error_mass
    return np.where(mask, (1 - weight) * reference + weight * proposed, reference)


def summarize(frame, scores, classes, reference):
    truth = frame.root_code.to_numpy()
    ranked = topk(scores, classes)
    baseline = topk(reference, classes)
    points = row_scores(truth, ranked)
    delta = points - row_scores(truth, baseline)
    by_question = pd.DataFrame({'question': frame.question_id, 'delta': delta})
    groups = by_question.groupby('question').delta.agg(['sum', 'size']).to_numpy()
    samples = np.random.default_rng(2026).integers(0, len(groups), (2000, len(groups)))
    boot = groups[samples, 0].sum(axis=1) / groups[samples, 1].sum(axis=1)
    errors = truth != 'NONE'
    return {'map3': float(points.mean()), 'top1_accuracy': float(np.mean(ranked[:, 0] == truth)),
            'recall_at_3': float(np.mean(np.any(ranked == truth[:, None], axis=1))),
            'root_error_map3': float(points[errors].mean()), 'none_map3': float(points[~errors].mean()),
            'root_error_rows_predicted_none_first': int(sum(ranked[errors, 0] == 'NONE')),
            'delta_vs_image': float(delta.mean()), 'rows_improved': int(sum(delta > 0)),
            'rows_regressed': int(sum(delta < 0)),
            'paired_question_bootstrap_delta_95': np.quantile(boot, [.025, .975]).tolist()}


def read_ocr_sample(sample):
    files = {path.name: path for path in Path('data/cpu-image-audit').rglob('*.jpeg')}
    if not set(sample.image_file) <= files.keys():
        raise ValueError('The fixed 60-photo CPU audit sample is incomplete')

    def read_one(filename):
        with Image.open(files[filename]) as raw:
            image = ImageOps.exif_transpose(raw).convert('RGB')
        image.thumbnail((1024, 1024))
        return ocr(image)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(read_one, sample.image_file))


def main():
    train = read_csv('data/raw/train.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    folds = make_folds(train)
    fit = train.loc[folds != 0].assign(work=lambda df: df.transcription_steps)
    development = train.loc[folds == 0].reset_index(drop=True).assign(work=lambda df: df.transcription_steps)
    sample = development.sample(n=60, random_state=2026)
    ocr_sample = sample.assign(work=read_ocr_sample(sample))
    predictions, coverage = fit_verifiers(fit, [development, sample, ocr_sample], classes)
    reference = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz',
                               development.trace_id.tolist(), classes)
    report = {'fitting_rows': len(fit), 'development_rows': len(development), 'paired_photo_rows': len(sample),
              'fold': 0, 'shortlist': 8, 'verifier_weight': .35, 'ocr': 'Tesseract eng PSM 11 at 1024-pixel long side',
              'evidence_coverage': dict(zip(['fit_gold', 'development_gold', 'sample_gold', 'sample_ocr'], coverage)),
              'scope': 'Exploratory validation on a reused development fold. All verifiers were fitted only on gold fitting transcripts. The paired OCR comparison includes a train-to-inference text shift.',
              'development_gold': {}, 'paired_sample': {}}
    error_reranker = partial(rerank_root_codes, none_index=classes.index('NONE'))
    variants = [('literal', predictions['literal'], rerank), ('structured', predictions['structured'], rerank),
                ('structured_fixed_none', predictions['structured'], error_reranker),
                ('root_only', predictions['root_only'], error_reranker)]
    for group, frame, image, index in [('development_gold', development, reference, 0),
                                       ('paired_sample', sample, reference[sample.index], 1)]:
        report[group]['image_baseline'] = summarize(frame, image, classes, image)
        for name, evidence, combine in variants:
            combined = combine(image, evidence[index])
            report[group][name + '_gold_rerank'] = summarize(frame, combined, classes, image)
            if group == 'paired_sample':
                combined = combine(image, evidence[2])
                report[group][name + '_ocr_rerank'] = summarize(frame, combined, classes, image)
    Path('reports/working_verifier_progress.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
