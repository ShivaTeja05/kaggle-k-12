import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import read_csv, label_space, aligned_bundle, topk, row_scores


def analyze(data, predictions, output, folds=None, allow_holdout=False):
    full_train = read_csv(Path(data)/'train.csv');train=full_train
    if allow_holdout:
        with np.load(predictions,allow_pickle=False) as b:ids=b['ids'].tolist()
        if not set(ids)<=set(train.trace_id):raise ValueError('Predictions include non-training IDs')
        train=train.set_index('trace_id').loc[ids].reset_index()
    classes = label_space(read_csv(Path(data)/'taxonomy_v1.csv'))
    p = aligned_bundle(predictions, train.trace_id.tolist(), classes)
    ranks = topk(p, classes)
    out = train[['trace_id', 'question_id', 'grade', 'topic_name', 'root_code']].copy()
    out['prediction'] = [' '.join(r) for r in ranks]
    out['score'] = row_scores(train.root_code, ranks)
    out['top1_correct'] = ranks[:, 0] == train.root_code
    out['top3_hit'] = [c in r for c, r in zip(train.root_code, ranks)]
    out['has_question_diagram']=train.question_image_file.astype(str).str.strip().ne('')
    out['final_answer_correct']=train.final_answer_correct.astype(str).str.lower().isin(['true','1'])
    out['clean_annotation']=train.clean.astype(str).str.lower().isin(['true','1'])
    out['gold_has_unclear_span']=train.transcription_steps.str.contains('unclear',case=False,regex=False)
    downstream=[]
    for raw,predicted in zip(train.misconceptions,ranks[:,0]):
        findings=json.loads(raw)
        nonroots={x.get('category') for x in findings if not x.get('root')}
        downstream.append(predicted in nonroots)
    out['top1_matches_nonroot_finding']=downstream
    counts = full_train.root_code.value_counts()
    out['training_support'] = out.root_code.map(counts)
    out['support_band'] = pd.cut(out.training_support, [0, 1, 5, 20, 10000], labels=['singleton', '2-5', '6-20', '21+'])
    if folds:
        fm = read_csv(folds).set_index('trace_id')['fold']
        out['fold'] = out.trace_id.map(fm)
        if out['fold'].isna().any(): raise ValueError('Missing fold IDs')
        out['question_seen'] = False
        for f in sorted(out.fold.unique()):
            full_folds=full_train.trace_id.map(fm)
            seen = set(full_train.loc[full_folds != f, 'question_id'])
            out.loc[out.fold == f, 'question_seen'] = out.loc[out.fold == f, 'question_id'].isin(seen)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root/'oof_errors.csv', index=False)
    slices = {}
    for col in ['root_code', 'grade', 'support_band','has_question_diagram','final_answer_correct',
                'clean_annotation','gold_has_unclear_span'] + (['question_seen'] if folds else []):
        table = out.groupby(col, observed=True).agg(rows=('score','size'), map3=('score','mean'), accuracy=('top1_correct','mean'), recall3=('top3_hit','mean'))
        table.to_csv(root/f'by_{col}.csv')
        slices[col] = table.reset_index().astype(object).to_dict(orient='records')
    true_none = train.root_code.to_numpy() == 'NONE'
    predicted_none = ranks[:, 0] == 'NONE'
    # Resample whole question groups: less optimistic than treating similar rows as independent.
    agg = out.groupby('question_id').score.agg(['sum','size']).to_numpy()
    rng = np.random.default_rng(2026)
    sample = rng.integers(0, len(agg), (2000, len(agg)))
    boot = agg[sample, 0].sum(axis=1) / agg[sample, 1].sum(axis=1)
    summary = {'map3': float(out.score.mean()), 'accuracy': float(out.top1_correct.mean()),
               'recall3': float(out.top3_hit.mean()), 'question_bootstrap_95': np.quantile(boot,[.025,.975]).tolist(),
               'false_alarm_on_NONE': float((~predicted_none[true_none]).mean()) if true_none.any() else None,
               'error_missed_as_NONE': float(predicted_none[~true_none].mean()) if (~true_none).any() else None,
               'top1_wrong_and_matches_nonroot_finding':int(((~out.top1_correct)&out.top1_matches_nonroot_finding).sum()),
               'correct_final_answer_root_error_predicted_NONE':int((out.final_answer_correct&~true_none&predicted_none).sum()),
               'incorrect_final_answer_NONE_predicted_error':int((~out.final_answer_correct&true_none&~predicted_none).sum()),
               'rows':len(train),'prediction_scope':'holdout' if allow_holdout else 'full_oof',
               'note': 'Local scores are not leaderboard scores; intervals exclude model-selection uncertainty.'}
    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    pd.crosstab(out.root_code, ranks[:,0]).to_csv(root/'confusion.csv')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/raw');p.add_argument('--predictions',required=True);p.add_argument('--output',default='artifacts/analysis');p.add_argument('--folds');p.add_argument('--allow-holdout',action='store_true')
    a=p.parse_args(); analyze(a.data,a.predictions,a.output,a.folds,a.allow_holdout)
