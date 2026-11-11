"""Candidate reranker using only signals that exist for every test row.

Inputs are the 4B image probabilities and question/topic priors, so a fitted
recipe can score the full test set on CPU. The reranker is cross-fitted by
question inside the reused development fold; it is not an untouched audit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_working_verifier import summarize
from k12.baseline import HierarchicalPrior
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, save_bundle, write_submission
from k12.holdout_calibration import calibrated


def family(code):
    return code.split('-', 1)[0].upper()


class CandidateFeatures:
    """Pairwise (row, code) features from image scores and fitting-row priors."""

    def __init__(self, classes, taxonomy, image_k, prior_k):
        self.classes = list(classes)
        self.index = {code: i for i, code in enumerate(self.classes)}
        self.families = np.asarray([family(code) for code in self.classes])
        self.band = np.asarray([float(taxonomy.get(code, np.nan)) for code in self.classes])
        self.image_k, self.prior_k = image_k, prior_k

    def fit(self, fit_rows):
        self.prior = HierarchicalPrior(self.classes).fit(fit_rows)
        counts = fit_rows.groupby('question_id').root_code.value_counts()
        self.question_counts = {key: count for key, count in counts.items()}
        self.question_sizes = fit_rows.question_id.value_counts().to_dict()
        frequency = fit_rows.root_code.value_counts()
        self.global_count = np.asarray([frequency.get(code, 0) for code in self.classes], dtype=float)
        return self

    def build(self, rows, image):
        prior = self.prior.predict_proba(rows)
        image_order = np.argsort(-image, axis=1, kind='stable')
        prior_order = np.argsort(-prior, axis=1, kind='stable')
        features, positions = [], []
        for i, row in enumerate(rows.itertuples(index=False)):
            image_rank = np.empty(len(self.classes), dtype=int)
            image_rank[image_order[i]] = np.arange(1, len(self.classes) + 1)
            prior_rank = np.empty(len(self.classes), dtype=int)
            prior_rank[prior_order[i]] = np.arange(1, len(self.classes) + 1)
            candidates = dict.fromkeys([*image_order[i, :self.image_k], *prior_order[i, :self.prior_k]])
            size = self.question_sizes.get(row.question_id, 0)
            top = image_order[i, :2]
            family_mass = {name: image[i, self.families == name].sum() for name in set(self.families[list(candidates)])}
            prior_family = {name: prior[i, self.families == name].sum() for name in family_mass}
            for index in candidates:
                code = self.classes[index]
                seen = self.question_counts.get((row.question_id, code), 0)
                is_none = code == 'NONE'
                grade_gap = 0. if np.isnan(self.band[index]) else float(row.grade) - self.band[index]
                features.append([
                    np.log(max(image[i, index], 1e-9)), 1 / image_rank[index],
                    np.log(max(prior[i, index], 1e-9)), 1 / prior_rank[index],
                    np.log1p(seen), seen / size if size else 0., size > 0,
                    family_mass[self.families[index]], prior_family[self.families[index]],
                    is_none, is_none * image[i, self.index['NONE']], is_none * prior[i, self.index['NONE']],
                    image[i, top[0]] - image[i, top[1]], np.log1p(self.global_count[index]),
                    grade_gap, grade_gap < 0,
                ])
                positions.append((i, index))
        return np.asarray(features, dtype=float), np.asarray(positions)


def score_matrix(shape, positions, confidence, fallback):
    """Candidates take reranker confidence; other codes keep image order far below."""
    scores = 1e-6 * normalize(fallback)
    scores[positions[:, 0], positions[:, 1]] = confidence + 1e-3
    return scores


def make_model(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def crossfit(validation, image, builder, c, splits=5):
    features, positions = builder.build(validation, image)
    truth = validation.root_code.to_numpy()
    targets = np.asarray(builder.classes)[positions[:, 1]] == truth[positions[:, 0]]
    groups = validation.question_id.to_numpy()[positions[:, 0]]
    confidence = np.empty(len(features))
    for fitting, held in GroupKFold(n_splits=splits).split(features, targets, groups):
        model = make_model(c).fit(features[fitting], targets[fitting])
        confidence[held] = model.predict_proba(features[held])[:, 1]
    return score_matrix(image.shape, positions, confidence, image), features, targets, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw')
    parser.add_argument('--validation', default='runs/vision-development/k12/validation_epoch_3.npz')
    parser.add_argument('--test', default='runs/vision-development/k12/test.npz')
    parser.add_argument('--output', default='artifacts/deployable-reranker')
    parser.add_argument('--image-k', type=int, default=10)
    parser.add_argument('--prior-k', type=int, default=3)
    parser.add_argument('--c', type=float, default=.1)
    parser.add_argument('--write-test', action='store_true')
    args = parser.parse_args()

    data = Path(args.data)
    train, test = read_csv(data / 'train.csv'), read_csv(data / 'test.csv')
    definitions = read_csv(data / 'taxonomy_v1.csv')
    classes = label_space(definitions)
    # Bands look like '4' or '4-6'; the lower bound marks when a skill is first taught.
    bands = (definitions.set_index('id').grade_band.str.extract(r'(\d+)')[0].astype(float).to_dict())
    folds = make_folds(train)
    fit, validation = train.loc[folds != 0], train.loc[folds == 0].reset_index(drop=True)
    image = aligned_bundle(args.validation, validation.trace_id.tolist(), classes)
    prior = HierarchicalPrior(classes).fit(fit).predict_proba(validation)
    submitted = calibrated(image, prior, .25, 2.)
    crossfit_calibration = aligned_bundle('artifacts/holdout-calibration/meta_crossfit.npz',
                                          validation.trace_id.tolist(), classes)

    builder = CandidateFeatures(classes, bands, args.image_k, args.prior_k).fit(fit)
    sweep = {}
    for c in [.01, .03, .1, .3, 1.]:
        scores, *_ = crossfit(validation, image, builder, c)
        sweep[str(c)] = summarize(validation, scores, classes, submitted)['map3']
    reranked, features, targets, positions = crossfit(validation, image, builder, args.c)
    truth = validation.root_code.to_numpy()
    coverage = np.mean([truth[i] in {classes[j] for j in positions[positions[:, 0] == i, 1]}
                        for i in range(len(validation))])
    report = {
        'rows': len(validation), 'fold': 0, 'candidate_pairs': len(features),
        'image_k': args.image_k, 'prior_k': args.prior_k, 'c': args.c,
        'candidate_coverage': float(coverage),
        'scope': 'Question-group cross-fitting of the reranker inside the reused development fold. '
                 'Features exist for every test row: 4B probabilities and fitting-row priors only.',
        'submitted_fixed_blend': summarize(validation, submitted, classes, submitted),
        'crossfit_calibration': summarize(validation, crossfit_calibration, classes, submitted),
        'reranker_versus_submitted_blend': summarize(validation, reranked, classes, submitted),
        'reranker_versus_crossfit_calibration': summarize(validation, reranked, classes, crossfit_calibration),
        'c_sweep_map3': sweep,
        'warning': 'C was swept on the same rows; the selected setting is optimistic. Not a public score.'}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_bundle(out / 'crossfit.npz', validation.trace_id.tolist(), classes, reranked,
                kind='question_crossfit_reranker_within_development')
    if args.write_test:
        # Fit the reranker on all development pairs; test priors use every labelled row.
        model = make_model(args.c).fit(features, targets)
        test_image = aligned_bundle(args.test, test.trace_id.tolist(), classes)
        test_builder = CandidateFeatures(classes, bands, args.image_k, args.prior_k).fit(train)
        test_features, test_positions = test_builder.build(test, test_image)
        test_scores = score_matrix(test_image.shape, test_positions,
                                   model.predict_proba(test_features)[:, 1], test_image)
        save_bundle(out / 'test.npz', test.trace_id.tolist(), classes, test_scores, kind='reranked_test')
        write_submission(test, test_scores, classes, out / 'submission.csv')
        report['test_written'] = True
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'scope'}, indent=1))


if __name__ == '__main__':
    main()
