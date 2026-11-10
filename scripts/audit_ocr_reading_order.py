"""Separate line-order penalties from symbol errors in the CPU OCR pilot."""
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from audit_cpu_ocr import edit_distance, math_tokens
from evaluate_working_verifier import read_ocr_sample
from k12.core import make_folds, read_csv


def token_lines(text):
    return [tokens for line in text.splitlines() if (tokens := math_tokens(line))]


def line_costs(reference, prediction):
    """Edit costs with ordered line alignment and free line assignment."""
    n, m = len(reference), len(prediction)
    size = max(n, m)
    if size == 0:
        return 0, 0
    cost = np.empty((size, size), dtype=int)
    ref_padded = reference + [[] for _ in range(size - n)]
    pred_padded = prediction + [[] for _ in range(size - m)]
    for i, ref in enumerate(ref_padded):
        for j, pred in enumerate(pred_padded):
            cost[i, j] = edit_distance(ref, pred)
    row_indices, column_indices = linear_sum_assignment(cost)
    unordered = int(cost[row_indices, column_indices].sum())

    previous = [0, *np.cumsum([len(line) for line in prediction]).tolist()]
    for i, ref in enumerate(reference):
        current = [previous[0] + len(ref)]
        for j, pred in enumerate(prediction):
            current.append(min(previous[j + 1] + len(ref), current[j] + len(pred),
                               previous[j] + int(cost[i, j])))
        previous = current
    return previous[-1], unordered


def first_cited_equation(row):
    references = {match for finding in json.loads(row.misconceptions) if finding.get('root') is True
                  for match in re.findall(r'(?i)\bL(\d+)', str(finding.get('step', '')))}
    for line in row.transcription_steps.splitlines():
        marker = re.match(r'^\s*L(\d+)\s*\|', line, re.I)
        if marker and marker.group(1) in references:
            tokens = math_tokens(line)
            if '=' in tokens and len(tokens) >= 3:
                return tokens
    return None


def main():
    train = read_csv('data/raw/train.csv')
    sample = train.loc[make_folds(train) == 0].sample(n=60, random_state=2026)
    predictions = read_ocr_sample(sample)
    totals = Counter()
    for row, transcript in zip(sample.itertuples(index=False), predictions):
        reference = token_lines(row.transcription_steps)
        prediction = token_lines(transcript)
        ref_tokens = [token for line in reference for token in line]
        pred_tokens = [token for line in prediction for token in line]
        ordered, unordered = line_costs(reference, prediction)
        overlap = sum((Counter(ref_tokens) & Counter(pred_tokens)).values())
        totals.update({'reference_tokens': len(ref_tokens), 'predicted_tokens': len(pred_tokens),
                       'line_ordered_edits': ordered, 'line_unordered_edits': unordered,
                       'math_token_overlap': overlap,
                       'minimum_edits_with_free_token_reordering': max(len(ref_tokens), len(pred_tokens)) - overlap,
                       'rows_with_order_gain': int(ordered > unordered)})
        cited = first_cited_equation(row)
        if cited is not None:
            totals['rows_with_cited_equation'] += 1
            totals['cited_equation_exact_ocr_line_matches'] += int(cited in prediction)

    count = totals['reference_tokens']
    report = {'rows': len(sample), 'fold': 0, 'sample_seed': 2026,
              'reader': 'English Tesseract, PSM 11, 1024-pixel long side',
              'ordered_line_math_token_error': totals['line_ordered_edits'] / count,
              'unordered_line_math_token_error': totals['line_unordered_edits'] / count,
              'optimistic_token_error_with_free_reordering': totals['minimum_edits_with_free_token_reordering'] / count,
              'order_free_math_token_recall': totals['math_token_overlap'] / count,
              'counts': dict(totals),
              'scope': 'The two line scores use the same gold and OCR line fragments. Unordered assignment is a diagnostic, with no OCR changes. Token overlap ignores order, grouping and semantic equivalence. Exact cited-equation matching requires all normalized math tokens in one detected line.'}
    Path('reports/ocr_reading_order_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__=='__main__':
    main()
