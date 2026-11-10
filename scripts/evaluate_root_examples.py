"""Screen a CPU reranker that contrasts root and downstream working examples."""
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate_working_verifier import (NUMBER, math_form, read_ocr_sample, rerank_root_codes,
                                       summarize, working_lines)
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, topk


def local_windows(text):
    previous = ''
    for index, line in enumerate(str(text).splitlines(), 1):
        match = re.match(r'^\s*L(\d+)\s*\|', line, flags=re.I)
        line_number = int(match.group(1)) if match else index
        active = working_lines(line)
        if not active:
            previous = ''
            continue
        current = math_form(active[0])
        yield line_number, ' | '.join(part for part in (previous, current) if part)
        previous = current


def example_bank(fitting, classes):
    texts, codes, is_root = [], [], []
    class_index = {code: i for i, code in enumerate(classes)}
    for row in fitting.itertuples(index=False):
        windows = dict(local_windows(row.transcription_steps))
        for finding in json.loads(row.misconceptions):
            code = finding.get('category')
            if code == 'NONE' or code not in class_index or finding.get('severity') != 'major':
                continue
            root = finding.get('root') is True
            if root != (code == row.root_code):
                continue
            references = re.findall(r'(?i)\bL(\d+)', str(finding.get('step', '')))
            example = next((windows[int(n)] for n in references if int(n) in windows), None)
            if example:
                texts.append(example)
                codes.append(class_index[code])
                is_root.append(root)
    return texts, np.asarray(codes), np.asarray(is_root, dtype=bool)


def score_transcripts(transcripts, texts, codes, is_root, class_count):
    literal = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2).fit(texts)
    patterns = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2)
    pattern_bank = patterns.fit_transform([NUMBER.sub(' N ', text) for text in texts])
    literal_bank = literal.transform(texts)
    positive_indices = {code: np.flatnonzero((codes == code) & is_root) for code in np.unique(codes[is_root])}
    negative_indices = {code: np.flatnonzero((codes == code) & ~is_root) for code in positive_indices}
    positive = np.zeros((len(transcripts), class_count))
    contrasted = np.zeros_like(positive)
    for row, transcript in enumerate(transcripts):
        windows = [window for _, window in local_windows(transcript)]
        if not windows:
            continue
        literal_sim = literal.transform(windows) @ literal_bank.T
        pattern_sim = patterns.transform([NUMBER.sub(' N ', window) for window in windows]) @ pattern_bank.T
        similarity = (.35 * literal_sim + .65 * pattern_sim).toarray()
        for code, indices in positive_indices.items():
            root_match = similarity[:, indices].max(axis=1)
            negatives = negative_indices[code]
            downstream_match = similarity[:, negatives].max(axis=1) if len(negatives) else 0
            positive[row, code] = root_match.max()
            contrasted[row, code] = np.maximum(0, root_match - .5 * downstream_match).max()
    return positive, contrasted


def combine_model_rankings(image, second_model, classes):
    scores = 1e-9 * image.copy()
    class_index = {code: i for i, code in enumerate(classes)}
    for ranked in (topk(image, classes, 8), topk(second_model, classes, 3)):
        for row, labels in enumerate(ranked):
            for rank, label in enumerate(labels, 1):
                scores[row, class_index[label]] += 1 / rank
    return normalize(scores)


def main():
    train = read_csv('data/raw/train.csv')
    classes = label_space(read_csv('data/raw/taxonomy_v1.csv'))
    folds = make_folds(train)
    fitting = train.loc[folds != 0]
    validation = train.loc[folds == 0].reset_index(drop=True)
    sample = validation.sample(n=60, random_state=2026)
    reference = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz',
                               validation.trace_id.tolist(), classes)
    second_model = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz',
                                  validation.trace_id.tolist(), classes)
    fused = combine_model_rankings(reference, second_model, classes)
    texts, codes, is_root = example_bank(fitting, classes)
    if not np.any(is_root) or not np.any(~is_root):
        raise ValueError('Both root and downstream training examples are required')
    report = {'fitting_rows': len(fitting), 'development_rows': len(validation),
              'paired_photo_rows': len(sample), 'root_examples': int(is_root.sum()),
              'downstream_examples': int((~is_root).sum()),
              'root_code_coverage': int(len(np.unique(codes[is_root]))),
              'shortlist': 8, 'mix_weight': .35, 'evidence_log_weight': 2.,
              'scope': 'Reused development fold. All examples are drawn from the fitting partition. Teacher transcripts are diagnostic inputs. OCR uses the same 60 training photos as the earlier CPU pilot.',
              'multi_model_scope': 'Cached 4B and 27B development predictions. Each verifier compares with fixed rank fusion. The 27B has only development predictions.',
              'evaluations': {}, 'multi_model_evaluations': {}}
    groups = [('development_teacher', validation, validation.transcription_steps, reference, fused),
              ('sample_teacher', sample, sample.transcription_steps, reference[sample.index], fused[sample.index]),
              ('sample_tesseract', sample, read_ocr_sample(sample), reference[sample.index], fused[sample.index])]
    none_index = classes.index('NONE')
    for name, frame, transcripts, image, fusion in groups:
        positive, contrast = score_transcripts(transcripts, texts, codes, is_root, len(classes))
        results = {'image_baseline': summarize(frame, image, classes, image)}
        multimodel = {'fixed_rank_fusion': summarize(frame, fusion, classes, image)}
        for variant, evidence in [('root_examples', positive), ('root_vs_downstream', contrast)]:
            boost = np.exp(2. * evidence)
            scores = rerank_root_codes(image, image * boost, none_index)
            results[variant] = summarize(frame, scores, classes, image)
            scores = rerank_root_codes(fusion, fusion * boost, none_index)
            multimodel[variant] = summarize(frame, scores, classes, fusion)
        report['evaluations'][name] = results
        report['multi_model_evaluations'][name] = multimodel
    Path('reports/root_example_screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
