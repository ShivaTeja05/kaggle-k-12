"""Audit which errors the saved Qwen rankings make on the same development rows."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_root_examples import combine_model_rankings
from evaluate_working_verifier import summarize
from k12.core import aligned_bundle, label_space, make_folds, read_csv, row_scores, topk


def family(code):
    return code.split('-', 1)[0].upper()


def slice_metrics(truth, codes, mask):
    selected = codes[mask, :3]
    targets = truth[mask]
    points = row_scores(targets, selected)
    return {'rows': len(points), 'map3': float(points.mean()) if len(points) else None,
            'first_correct': int(sum(selected[:, 0] == targets)),
            'top3_hits': int(sum(points > 0)), 'lost_points': float(np.sum(1 - points))}


def screen_router(validation, image, fused, left, right, classes):
    """Predict the paired benefit of fusion with each question held out of the router fit."""
    truth = validation.root_code.to_numpy()
    target = row_scores(truth, topk(fused, classes)) - row_scores(truth, left)
    class_index = {code: i for i, code in enumerate(classes)}
    order = np.sort(image, axis=1)
    features = np.column_stack([
        image[:, class_index['NONE']], order[:, -1], order[:, -1] - order[:, -2],
        left[:, 0] == 'NONE', right[:, 0] == 'NONE', np.any(right == 'NONE', axis=1),
        left[:, 0] == right[:, 0], np.any(left[:, :3] == right[:, :1], axis=1),
        [len(set(a[:3]) & set(b)) for a, b in zip(left, right)],
        image[np.arange(len(image)), [class_index[code] for code in right[:, 0]]],
    ])
    gate = make_pipeline(StandardScaler(), Ridge(alpha=20))
    predicted = cross_val_predict(gate, features, target, groups=validation.question_id,
                                  cv=GroupKFold(n_splits=5))
    use_fusion = predicted > 0
    routed = np.where(use_fusion[:, None], fused, image)
    against_fusion = summarize(validation, routed, classes, fused)
    return {'scope': 'Question-group cross-fitting of the router only. The input rankings are fixed '
                     'development predictions, and the 27B has no test predictions.',
            'routed_to_fusion': int(sum(use_fusion)),
            'versus_image': summarize(validation, routed, classes, image),
            'delta_vs_fixed_fusion': against_fusion['delta_vs_image'],
            'paired_question_bootstrap_delta_vs_fixed_fusion_95': against_fusion['paired_question_bootstrap_delta_95']}


def screen_candidate_reranker(validation, image, fused, left, right, classes):
    """Reorder the candidate union using only model scores and taxonomy families."""
    class_index = {code: i for i, code in enumerate(classes)}
    class_families = np.asarray([family(code) for code in classes])
    features, targets, groups, positions = [], [], [], []
    for i, row in enumerate(validation.itertuples(index=False)):
        rank_4b = {code: rank for rank, code in enumerate(left[i], 1)}
        rank_27b = {code: rank for rank, code in enumerate(right[i], 1)}
        none_probability = image[i, class_index['NONE']]
        candidates = dict.fromkeys([*left[i, :8], *right[i]])
        for code in candidates:
            index = class_index[code]
            family_match = class_families == class_families[index]
            features.append([
                np.log(max(image[i, index], 1e-12)), 1 / rank_4b[code],
                1 / rank_27b[code] if code in rank_27b else 0.,
                image[i, family_match].sum(),
                family(code) == family(right[i, 0]), code == 'NONE',
                none_probability * (code == 'NONE'),
                image[i, class_index[left[i, 0]]] - image[i, class_index[left[i, 1]]],
            ])
            targets.append(code == row.root_code)
            groups.append(row.question_id)
            positions.append((i, index))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=1000))
    confidence = cross_val_predict(model, features, targets, groups=groups,
                                   cv=GroupKFold(n_splits=5), method='predict_proba')[:, 1]
    reranked = np.zeros_like(image)
    row_index, col_index = np.asarray(positions).T
    reranked[row_index, col_index] = confidence
    against_fusion = summarize(validation, reranked, classes, fused)
    return {'scope': 'Question-group cross-fitting of the candidate reranker only. Features are fixed '
                     'development rankings, probabilities and taxonomy families. No OCR features.',
            'versus_image': summarize(validation, reranked, classes, image),
            'delta_vs_fixed_fusion': against_fusion['delta_vs_image'],
            'paired_question_bootstrap_delta_vs_fixed_fusion_95': against_fusion['paired_question_bootstrap_delta_95']}


def main():
    train = read_csv('data/raw/train.csv')
    definitions = read_csv('data/raw/taxonomy_v1.csv')
    classes = label_space(definitions)
    taxonomy = definitions.set_index('id')['name'].to_dict()
    taxonomy['NONE'] = 'No major misconception'
    folds = make_folds(train)
    fit, validation = train.loc[folds != 0], train.loc[folds == 0].reset_index(drop=True)
    ids = validation.trace_id.tolist()
    truth = validation.root_code.to_numpy()
    roots = truth != 'NONE'
    target_family = np.asarray([family(code) for code in truth])
    major = [[finding for finding in json.loads(text) if finding.get('severity') == 'major']
             for text in validation.misconceptions]
    downstream = [{finding['category'] for finding in row if finding.get('root') is not True}
                  for row in major]
    support = validation.root_code.map(fit.root_code.value_counts()).fillna(0).to_numpy()
    seen = validation.question_id.isin(fit.question_id).to_numpy()
    final_correct = validation.final_answer_correct.astype(str).str.lower().eq('true').to_numpy()
    masks = {'none': ~roots, 'root_error': roots,
             'root_error_with_correct_final_answer': roots & final_correct,
             'root_error_with_other_final_answer': roots & ~final_correct,
             'root_error_with_multiple_major_findings': roots & np.asarray([len(row) > 1 for row in major]),
             'root_error_with_at_most_one_major_finding': roots & np.asarray([len(row) <= 1 for row in major]),
             'root_error_0_to_5_fit_examples': roots & (support <= 5),
             'root_error_6_to_20_fit_examples': roots & (support > 5) & (support <= 20),
             'root_error_above_20_fit_examples': roots & (support > 20),
             'root_error_with_question_diagram': roots & validation.question_image_file.str.strip().ne('').to_numpy(),
             'root_error_without_question_diagram': roots & validation.question_image_file.str.strip().eq('').to_numpy(),
             'root_error_seen_question': roots & seen,
             'root_error_unseen_question': roots & ~seen}
    image = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', ids, classes)
    large = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz', ids, classes)
    fused = combine_model_rankings(image, large, classes)
    models = {'qwen3_vl_4b_calibrated': image, 'qwen3_5_27b_frozen': large, 'fixed_rank_fusion': fused}
    rankings = {name: topk(scores, classes, 3 if name == 'qwen3_5_27b_frozen' else len(classes))
                for name, scores in models.items()}
    left, right = rankings['qwen3_vl_4b_calibrated'], rankings['qwen3_5_27b_frozen']
    left_points, right_points = row_scores(truth, left), row_scores(truth, right)
    in_left8 = np.any(left[:, :8] == truth[:, None], axis=1)
    in_right3 = right_points > 0
    both_miss = (left_points == 0) & ~in_right3
    union = in_left8 | in_right3
    missing = roots & ~union

    def count_lost(mask):
        return {'rows': int(sum(mask)), 'lost_points': float(np.sum(1 - left_points[mask]))}

    report = {'rows': len(validation), 'fitting_rows': len(fit), 'fold': 0,
              'scope': 'Reused development fold with aligned prediction IDs. 4B is task-fitted and calibrated. '
                       '27B is frozen and its saved artifact provides three ranks only. '
                       'This compares complete model recipes. Gold labels and annotation flags are audit inputs.',
              'confidence_routing': screen_router(validation, image, fused, left, right, classes),
              'candidate_reranking': screen_candidate_reranker(validation, image, fused, left, right, classes),
              'models': {},
              'complementarity': {'27b_better_rows': int(sum(right_points > left_points)),
                                  '4b_better_rows': int(sum(left_points > right_points)),
                                  'top3_rescues_27b_of_4b': int(sum((left_points == 0) & in_right3)),
                                  'top3_rescues_4b_of_27b': int(sum((left_points > 0) & ~in_right3)),
                                  'both_miss_top3': int(sum(both_miss)),
                                  'both_miss_top3_root_errors': int(sum(roots & both_miss)),
                                  'both_miss_top3_but_in_4b_top8': int(sum(both_miss & in_left8)),
                                  '4b_top8_plus_27b_top3_coverage': float(np.mean(union)),
                                  'root_errors_outside_combined_candidates': int(sum(missing)),
                                  'missing_codes': dict(Counter(truth[missing]).most_common()),
                                  'oracle_choose_better_map3': float(np.maximum(left_points, right_points).mean())},
              '4b_loss_by_correct_label_rank': {
                  'rank_2_or_3': count_lost((left_points > 0) & (left_points < 1)),
                  'rank_4_to_8': count_lost((left_points == 0) & in_left8),
                  'outside_top8': count_lost(~in_left8)}}
    for name, codes in rankings.items():
        first = codes[:, 0]
        wrong = first != truth
        first_family = np.asarray([family(code) for code in first])
        points = row_scores(truth, codes)
        modes = {'false_error_on_none': ~roots & wrong,
                 'root_missed_as_none': roots & (first == 'NONE'),
                 'wrong_root_family': roots & wrong & (first != 'NONE') & (first_family != target_family),
                 'wrong_code_in_correct_family': roots & wrong & (first_family == target_family)}
        confusions = Counter(zip(truth[wrong], first[wrong]))
        report['models'][name] = {
            'metrics': summarize(validation, models[name], classes, image),
            'slices': {key: slice_metrics(truth, codes, mask) for key, mask in masks.items()},
            'rank_one_failures': {key: {'rows': int(sum(mask)), 'lost_points': float(np.sum(1 - points[mask]))}
                                  for key, mask in modes.items()},
            'wrong_first_matches_annotated_downstream_major': int(sum(
                is_wrong and guess in findings for is_wrong, guess, findings in zip(wrong, first, downstream))),
            'largest_repeated_confusions': [
                {'gold_code': gold, 'gold_name': taxonomy[gold], 'first_code': guess,
                 'first_name': taxonomy[guess], 'rows': count}
                for (gold, guess), count in confusions.most_common(10)]}

    output = Path('reports/model_error_audit.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
