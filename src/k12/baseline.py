"""Question/topic priors using only inputs available in test.csv."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .core import (read_csv, label_space, make_folds, map3, save_bundle,
                   write_submission, fingerprint)


class HierarchicalPrior:
    def __init__(self, classes, question_strength=2.0, topic_strength=8.0):
        self.classes = list(classes)
        self.a, self.b = question_strength, topic_strength

    def fit(self, df):
        ci = {c: i for i, c in enumerate(self.classes)}
        k = len(ci)
        self.global_p = np.bincount(df.root_code.map(ci), minlength=k).astype(float) + 0.5
        self.global_p /= self.global_p.sum()
        self.topic, self.question = {}, {}
        for col, target in [('topic_name', self.topic), ('question_id', self.question)]:
            for key, g in df.groupby(col, sort=False):
                target[str(key)] = np.bincount(g.root_code.map(ci), minlength=k).astype(float)
        return self

    def predict_proba(self, df):
        result = []
        for r in df.itertuples(index=False):
            t = self.topic.get(str(r.topic_name), np.zeros(len(self.classes)))
            pt = (t + self.b * self.global_p) / (t.sum() + self.b)
            q = self.question.get(str(r.question_id), np.zeros(len(self.classes)))
            result.append((q + self.a * pt) / (q.sum() + self.a))
        return np.asarray(result)


def run(data, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train, test, tax = [read_csv(Path(data) / p) for p in ['train.csv', 'test.csv', 'taxonomy_v1.csv']]
    classes = label_space(tax)
    if not set(train.root_code) <= set(classes):
        raise ValueError('Unknown labels in train')
    folds = make_folds(train)
    import pandas as pd
    pd.DataFrame({'trace_id': train.trace_id, 'fold': folds}).to_csv(output/'folds.csv', index=False)
    p = np.zeros((len(train), len(classes)))
    fold_scores = []
    for f in sorted(set(folds)):
        a, v = train.loc[folds != f], train.loc[folds == f]
        model = HierarchicalPrior(classes).fit(a)
        p[folds == f] = model.predict_proba(v)
        fold_scores.append(map3(v.root_code, p[folds == f], classes))
    model = HierarchicalPrior(classes).fit(train)
    test_p = model.predict_proba(test)
    joblib.dump(model, output/'model.joblib')
    save_bundle(output/'oof.npz', train.trace_id.tolist(), classes, p, kind='oof', folds=folds.tolist())
    save_bundle(output/'test.npz', test.trace_id.tolist(), classes, test_p, kind='test')
    write_submission(test, test_p, classes, output/'submission.csv')
    result = {'model': 'hierarchical_prior', 'map3_oof': map3(train.root_code, p, classes),
              'fold_map3': fold_scores, 'train_sha256': fingerprint(Path(data)/'train.csv'),
              'status': 'Real CPU baseline; no image signal; not a top-three model'}
    (output/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', default='data/raw'); p.add_argument('--output', default='artifacts/baseline')
    a = p.parse_args(); run(a.data, a.output)
