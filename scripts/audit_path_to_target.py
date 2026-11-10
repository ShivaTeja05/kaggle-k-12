"""Audit canonical data and the ranking headroom of saved development predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from k12.core import TRAIN_ONLY, aligned_bundle, fingerprint, label_space, make_folds, read_csv, row_scores, topk


def normalized_text(value):
    return ' '.join(unicodedata.normalize('NFC', str(value)).split())


def audit_dataset(train, test, taxonomy):
    context = ['question_text', 'correct_answer', 'question_image_file', 'grade', 'topic_name']
    questions = pd.concat([train, test], ignore_index=True)[['question_id', *context]].copy()
    for field in context:
        questions[field] = questions[field].map(normalized_text)
    conflicts = questions.groupby('question_id')[context].nunique().gt(1).sum()
    supports = train.root_code.value_counts()
    seen_question = test.question_id.isin(train.question_id)
    seen_pair = train.groupby('question_id').root_code.agg(set)
    root_count_errors = root_code_errors = missing_root_line_refs = nonfinite_annotations = 0
    crosses = uncertain = missing_line_prefix = multi_major = 0
    for row in train.itertuples(index=False):
        bad_constants = []

        def replace_constant(value):
            bad_constants.append(value)
            return None

        findings = json.loads(row.misconceptions, parse_constant=replace_constant)
        nonfinite_annotations += bool(bad_constants)
        roots = [f for f in findings if f.get('root') is True]
        root_count_errors += len(roots) != int(row.root_code != 'NONE')
        root_code_errors += any(f.get('category') != row.root_code for f in roots)
        line_ids = set(re.findall(r'(?im)^\s*L(\d+)\s*\|', row.transcription_steps))
        for finding in roots:
            refs = set(re.findall(r'(?i)\bL(\d+)', str(finding.get('step', ''))))
            missing_root_line_refs += bool(refs - line_ids)
        multi_major += sum(f.get('severity') == 'major' for f in findings) > 1
        crosses += bool(re.search(r'\bcrossed[ -]out\b|\bstruck[ -]out\b', row.transcription_steps, re.I))
        uncertain += bool(re.search(r'\b(?:unclear|illegible|unreadable)\b', row.transcription_steps, re.I))
        missing_line_prefix += not bool(line_ids)

    held = make_folds(train) == 0
    fitted = train.loc[~held]
    val = train.loc[held]
    fitted_codes = fitted.groupby('question_id').root_code.agg(set)
    same_question = val.question_id.isin(fitted_codes.index)
    matching_code = np.asarray([
        code in fitted_codes.get(question, set())
        for question, code in zip(val.question_id, val.root_code)
    ])
    return {
        'train_rows': len(train), 'test_rows': len(test), 'classes': len(label_space(taxonomy)),
        'duplicate_trace_ids': {'train': int(train.trace_id.duplicated().sum()),
                                'test': int(test.trace_id.duplicated().sum())},
        'train_test_trace_id_overlap': len(set(train.trace_id) & set(test.trace_id)),
        'train_test_image_filename_overlap': len(set(train.image_file) & set(test.image_file)),
        'test_train_only_fields': sorted(set(test.columns) & TRAIN_ONLY),
        'shared_question_context_conflicts': {key: int(n) for key, n in conflicts.items()},
        'test_seen_question_rows': int(seen_question.sum()),
        'test_rows_with_diagram': int(test.question_image_file.str.strip().ne('').sum()),
        'question_prior_hard_filter_diagnostic': {
            'heldout_seen_question_rows': int(same_question.sum()),
            'heldout_seen_question_with_unseen_code_rows': int((same_question & ~matching_code).sum()),
            'labelled_questions_with_only_one_seen_code': int(sum(len(codes) == 1 for codes in seen_pair)),
        },
        'support': {str(k): int(v) for k, v in supports.items()},
        'taxonomy_codes_without_train_examples': sorted(set(taxonomy.id) - set(supports.index)),
        'train_none_rows': int(train.root_code.eq('NONE').sum()),
        'train_none_with_clean_false': int((train.root_code.eq('NONE') & ~train.clean).sum()),
        'train_correct_final_but_root_error': int((train.final_answer_correct & train.root_code.ne('NONE')).sum()),
        'train_wrong_final_but_none': int((~train.final_answer_correct & train.root_code.eq('NONE')).sum()),
        'root_count_mismatch_rows': root_count_errors, 'root_code_mismatch_rows': root_code_errors,
        'root_findings_referencing_missing_transcript_line': missing_root_line_refs,
        'rows_with_nonfinite_annotation_json_constants': nonfinite_annotations,
        'rows_with_multiple_major_findings': multi_major,
        'rows_with_crossed_out_markers': crosses, 'rows_with_uncertain_transcription_markers': uncertain,
        'rows_without_any_recognized_line_marker': missing_line_prefix,
    }


def audit_candidates(train, classes):
    folds = make_folds(train)
    validation = train.loc[folds == 0].reset_index(drop=True)
    ids = validation.trace_id.tolist()
    truth = validation.root_code.to_numpy()
    reference = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', ids, classes)
    direct = aligned_bundle('runs/vision-development/k12/validation_epoch_3.npz', ids, classes)
    larger = aligned_bundle('runs/qwen35-27b/vllm_ranker/validation.npz', ids, classes)
    gold = aligned_bundle('artifacts/gold-text-diagnostic/gold_text_diagnostic.npz', train.trace_id.tolist(), classes)[folds == 0]
    scores = row_scores(truth, topk(reference, classes))
    class_codes = np.asarray(classes)
    families = np.asarray([code.split('-', 1)[0].upper() for code in classes])
    true_families = np.asarray([code.split('-', 1)[0].upper() for code in truth])
    correct_presence = (truth[:, None] == 'NONE') == (class_codes[None, :] == 'NONE')
    correct_family = true_families[:, None] == families[None, :]
    ranking = topk(reference, classes, len(classes))
    rank = np.argmax(ranking == truth[:, None], axis=1) + 1
    other = topk(larger, classes)
    # The 27B artifact contains three ranked codes, so compare ranks instead
    # of interpreting its artificial scores as probabilities.
    class_indices = {code: i for i, code in enumerate(classes)}
    fused = np.zeros_like(reference)
    for i, (left, right) in enumerate(zip(ranking, other)):
        for codes in (left[:8], right):
            for position, code in enumerate(codes, 1):
                fused[i, class_indices[code]] += 1 / position
    # Deterministic tie breaker preserves the calibrated model's order.
    fused += 1e-9 * reference
    fused_scores = row_scores(truth, topk(fused, classes))
    by_question = pd.DataFrame({'question': validation.question_id, 'delta': fused_scores - scores})
    groups = by_question.groupby('question').delta.agg(['sum', 'size']).to_numpy()
    samples = np.random.default_rng(2026).integers(0, len(groups), size=(2000, len(groups)))
    bootstrap = groups[samples, 0].sum(axis=1) / groups[samples, 1].sum(axis=1)
    error_confusions = Counter(zip(truth[(truth != 'NONE') & (rank > 1)], ranking[(truth != 'NONE') & (rank > 1), 0]))
    budget = float(.82 - scores.mean())
    shortlist = {}
    for k in (3, 5, 8, 10, 12, 20):
        combined = [set(a[:k]) | set(b) for a, b in zip(ranking, other)]
        union_recall = np.mean([target in candidates for target, candidates in zip(truth, combined)])
        shortlist[str(k)] = {'calibrated_recall': float(np.mean(rank <= k)),
                             'direct_image_recall': float(np.mean(np.any(topk(direct, classes, k) == truth[:, None], axis=1))),
                             'union_with_27b_top3_recall': float(union_recall),
                             'mean_union_size': float(np.mean([len(c) for c in combined])),
                             'remaining_gain_if_perfectly_reranked': float(np.mean(rank <= k) - scores.mean())}
    slices = {}
    support = validation.root_code.map(train.root_code.value_counts())
    for name, mask in [('none', truth == 'NONE'), ('has_root_error', truth != 'NONE'),
                       ('label_support_at_most_5', support <= 5), ('label_support_above_20', support > 20),
                       ('has_diagram', validation.question_image_file.str.strip().ne(''))]:
        slices[name] = {'rows': int(mask.sum()), 'map3': float(scores[mask].mean()),
                        'recall_at_8': float(np.mean(rank[mask] <= 8)),
                        'lost_score_sum': float(np.sum(1 - scores[mask]))}
    return {
        'rows': len(ids), 'fold': 0, 'previously_used_for_model_selection': True,
        'calibrated_map3': float(scores.mean()), 'top1': float(np.mean(rank == 1)),
        'diagnostic_with_gold_error_presence': float(row_scores(truth, topk(reference * correct_presence, classes)).mean()),
        'diagnostic_with_gold_code_family': float(row_scores(truth, topk(reference * correct_family, classes)).mean()),
        'rank_counts': {'1': int(sum(rank == 1)), '2': int(sum(rank == 2)), '3': int(sum(rank == 3)),
                        '4_to_8': int(sum((rank > 3) & (rank <= 8))), 'outside_8': int(sum(rank > 8))},
        'target': .82, 'target_gap': budget, 'target_gap_as_sum_of_row_scores': budget * len(ids),
        'fixed_reciprocal_rank_fusion_map3': float(fused_scores.mean()),
        'fusion_delta_vs_calibrated_question_bootstrap_95': np.quantile(bootstrap, [.025, .975]).tolist(),
        'error_only_map3_needed_at_current_none_score': float((.82 * len(ids) - scores[truth == 'NONE'].sum()) / sum(truth != 'NONE')),
        'most_common_error_confusions': [
            {'gold': gold_code, 'predicted_first': guessed_code, 'rows': count}
            for (gold_code, guessed_code), count in error_confusions.most_common(10)
        ],
        'minimum_top1_at_current_recall_at_8_for_target': float(2 * .82 - np.mean(rank <= 8)),
        'gold_transcript_tfidf_map3_same_rows': float(row_scores(truth, topk(gold, classes)).mean()),
        'shortlist': shortlist, 'slices': slices,
    }


def audit_teacher_targets(train, tokenizer_path):
    from tokenizers import Tokenizer
    from k12.reading_finetune import teacher_response

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cap = 128
    lengths = {'reading': [], 'cause': []}
    cited_rows = line_starts_lost = line_ends_lost = 0
    for row in train.itertuples(index=False):
        reading = teacher_response(row, 0)
        encoding = tokenizer.encode(reading, add_special_tokens=False)
        lengths['reading'].append(len(encoding.ids))
        cause = tokenizer.encode(teacher_response(row, 1), add_special_tokens=False)
        lengths['cause'].append(len(cause.ids))
        findings = json.loads(row.misconceptions)
        refs = {n for f in findings if f.get('root') for n in re.findall(r'(?i)\bL(\d+)', f['step'])}
        transcript = row.transcription_steps
        prefix_length = len(reading) - len(transcript)
        lines = list(re.finditer(r'(?im)^[ \t]*L(\d+)\s*\|', transcript))
        for i, line in enumerate(lines):
            if line.group(1) not in refs:
                continue
            cited_rows += 1
            if len(encoding.ids) > cap:
                cutoff = encoding.offsets[cap - 1][1] - prefix_length
                line_end = lines[i + 1].start() if i + 1 < len(lines) else len(transcript)
                line_starts_lost += line.start() >= cutoff
                line_ends_lost += len(transcript[:line_end].rstrip()) > cutoff
            break
    return {'rows': len(train), 'teacher_token_cap': cap, 'tokenizer_sha256': fingerprint(tokenizer_path),
            'root_rows_with_identifiable_cited_line': cited_rows,
            'first_cited_line_start_at_or_after_cutoff': line_starts_lost,
            'first_cited_line_extends_beyond_cutoff': line_ends_lost,
            'target_lengths': {kind: {'median': float(np.median(values)), 'p90': float(np.percentile(values, 90)),
                                      'max': max(values), 'rows_over_cap': sum(v > cap for v in values)}
                               for kind, values in lengths.items()}}


def audit_unlabelled(train, test, path):
    unlabelled = read_csv(path)
    labelled_questions = set(train.question_id)
    unlabelled_questions = set(unlabelled.question_id)
    low_support_codes = set(train.root_code.value_counts().loc[lambda counts: counts <= 5].index)
    low_support_questions = set(train.loc[train.root_code.isin(low_support_codes), 'question_id'])
    return {'sha256': fingerprint(path), 'rows': len(unlabelled), 'questions': len(unlabelled_questions),
            'duplicate_trace_ids': int(unlabelled.trace_id.duplicated().sum()),
            'unlabelled_train_trace_id_overlap': len(set(unlabelled.trace_id) & set(train.trace_id)),
            'unlabelled_test_trace_id_overlap': len(set(unlabelled.trace_id) & set(test.trace_id)),
            'unlabelled_rows_on_labelled_questions': int(unlabelled.question_id.isin(labelled_questions).sum()),
            'unlabelled_rows_on_questions_with_any_rare_train_code': int(unlabelled.question_id.isin(low_support_questions).sum()),
            'test_rows_with_question_in_pool': int(test.question_id.isin(unlabelled_questions).sum()),
            'test_rows_on_unseen_train_questions_with_question_in_pool': int(
                (test.question_id.isin(unlabelled_questions) & ~test.question_id.isin(labelled_questions)).sum()),
            'note': 'Question overlap measures sampling opportunity. It does not establish any unlabelled row’s root code.'}


def audit_family_head(train, classes, cache):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['train_sha256'] != fingerprint('data/raw/train.csv') or manifest['classes'] != classes:
        raise ValueError('Frozen features do not match the canonical data and classes')
    folds = make_folds(train)
    fitting_ids = set(train.loc[folds != 0, 'trace_id'])
    validation_ids = set(train.loc[folds == 0, 'trace_id'])
    if set(manifest['checkpoint_trained_ids']) != fitting_ids:
        raise ValueError('Feature extractor was trained on a different partition')
    with np.load(cache / 'fitting_features.npz', allow_pickle=False) as fitting:
        fit_ids = fitting['ids'].tolist()
        fit_hidden = normalize(fitting['hidden'])
    with np.load(cache / 'validation_features.npz', allow_pickle=False) as heldout:
        val_ids = heldout['ids'].tolist()
        val_hidden = normalize(heldout['hidden'])
    if set(fit_ids) != fitting_ids or set(val_ids) != validation_ids or len(fit_ids) != len(fitting_ids) or len(val_ids) != len(validation_ids):
        raise ValueError('Frozen feature IDs do not match the fitting and development partitions')

    labels = train.set_index('trace_id').root_code
    families = np.asarray([code.split('-', 1)[0].upper() for code in classes])
    fit_family = labels.loc[fit_ids].str.split('-').str[0].str.upper()
    val_family = labels.loc[val_ids].str.split('-').str[0].str.upper().to_numpy()
    model = LogisticRegression(C=4, max_iter=400).fit(fit_hidden, fit_family)
    predicted_family = model.predict_proba(val_hidden)
    reference = aligned_bundle('artifacts/holdout-calibration/tuned_development.npz', val_ids, classes)
    family_order = sorted(set(families))
    family_mass = np.column_stack([reference[:, families == f].sum(axis=1) for f in family_order])
    index = {family: i for i, family in enumerate(family_order)}
    class_family = np.asarray([index[f] for f in families])
    new_mass = np.zeros_like(family_mass)
    new_mass[:, [index[f] for f in model.classes_]] = predicted_family
    blend = reference * (.5 + .5 * new_mass[:, class_family] / family_mass[:, class_family])
    truth = labels.loc[val_ids]
    return {'rows': len(val_ids), 'families': len(family_order),
            'reference_family_top1_accuracy': float(np.mean(np.asarray(family_order)[family_mass.argmax(axis=1)] == val_family)),
            'trained_family_top1_accuracy': float(np.mean(model.classes_[predicted_family.argmax(axis=1)] == val_family)),
            'fixed_half_family_blend_map3': float(row_scores(truth, topk(blend, classes)).mean()),
            'reference_map3': float(row_scores(truth, topk(reference, classes)).mean()),
            'scope': 'Exploratory held-out image features. One regularized family classifier on fitting rows. No test features.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=Path, help='Optional saved Qwen tokenizer.json for supervision-length audit')
    parser.add_argument('--unlabelled', type=Path, help='Optional canonical unlabelled.csv for pool coverage')
    parser.add_argument('--feature-cache', type=Path, help='Optional fitting/validation features for a family classifier')
    args = parser.parse_args()
    data = Path('data/raw')
    train, test, taxonomy = [read_csv(data / name) for name in ('train.csv', 'test.csv', 'taxonomy_v1.csv')]
    report = {'data_sha256': {p.name: fingerprint(p) for p in data.glob('*.csv')},
              'dataset': audit_dataset(train, test, taxonomy),
              'candidates': audit_candidates(train, label_space(taxonomy))}
    if args.unlabelled:
        report['unlabelled'] = audit_unlabelled(train, test, args.unlabelled)
    if args.feature_cache:
        report['family_head'] = audit_family_head(train, label_space(taxonomy), args.feature_cache)
    Path('reports/path_to_target_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report['candidates'], indent=2))
    if args.tokenizer:
        targets = audit_teacher_targets(train, args.tokenizer)
        Path('reports/reading_target_audit.json').write_text(json.dumps(targets, indent=2) + '\n')
        print(json.dumps(targets, indent=2))


if __name__ == '__main__':
    main()
