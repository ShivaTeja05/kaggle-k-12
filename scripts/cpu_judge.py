"""A judge that runs on this Mac: gradient-boosted trees over simple evidence features, no GPU.

For every photo that Qwen3.8 has described and every code, it scores "would the teachers give this code?" from:
  what Qwen3.8 said        its three ranked codes, the code family, whether it found an error line at all
  the question's history   how often teachers gave this code to other students on the same question (never the row itself)
  similar workings         how close Qwen's transcription is to the teacher transcriptions of same-question rows with this code
  similar teacher notes    how close Qwen's description is to teacher notes with this code (same question / other questions)
  the code itself          how close Qwen's description is to the code's definition; how common the code is in the topic
Trained on the fit rows (their evidence is leave-one-out), checked on dev, applied to the test photos Qwen described.
Qwen3.8 never saw a label, so the same features exist for training, dev and test rows.

Usage: python scripts/cpu_judge.py --out runs/cpu_judge   (writes dev.npz, test.npz, report.json; never submits)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from build_ranker_v3 import KEYS, QWEN, qwen_answers, root_note
from k12.core import label_space, make_folds, read_csv, row_scores, save_bundle, topk

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['q_first', 'q_second', 'q_third', 'q_rank_share', 'same_family_as_first', 'q_says_none', 'is_none',
         'question_share', 'question_count', 'question_rows', 'tx_sim_same_question', 'tx_sim_rank', 'note_sim_same_question',
         'note_sim_other_question', 'definition_sim', 'topic_share', 'global_share', 'tx_sim_best_any_code']


class Pool:
    """Teacher-labelled rows used as evidence."""

    def __init__(self, frame, classes, tx_vec, note_vec):
        self.f = frame.reset_index(drop=True); self.classes = classes; self.ci = {c: i for i, c in enumerate(classes)}
        self.code = np.array([self.ci[c] for c in self.f.root_code]); self.q = self.f.question_id.to_numpy()
        self.trace = self.f.trace_id.to_numpy(); self.topic = self.f.topic_name.to_numpy()
        self.T = tx_vec.transform(self.f.transcription_steps.astype(str))
        notes = [root_note(r) for r in self.f.itertuples(index=False)]
        self.has_note = np.array([bool(n) for n in notes]); self.N = note_vec.transform(notes)
        self.onehot = np.zeros((len(self.f), len(classes))); self.onehot[np.arange(len(self.f)), self.code] = 1


def features(rows, answers, pool, tx_vec, note_vec, def_matrix, def_vec):
    """One feature vector per (row, code): array [rows, classes, features]. Rows must have a Qwen answer."""
    C = len(pool.classes); X = np.zeros((len(rows), C, len(NAMES)), dtype=np.float32)
    fam = np.array([c.split('-')[0] for c in pool.classes])
    QT = tx_vec.transform([str(answers[t].get('transcription', '')) for t in rows.trace_id])
    text = [f"{answers[t].get('root_line', '')} {answers[t].get('reasoning', '')}" for t in rows.trace_id]
    QN = note_vec.transform(text); QD = def_vec.transform(text)
    S_tx = np.asarray((QT @ pool.T.T).todense()); S_note = np.asarray((QN @ pool.N.T).todense()); S_def = np.asarray((QD @ def_matrix.T).todense())
    glob = pool.onehot.mean(0)
    for i, row in enumerate(rows.itertuples(index=False)):
        a = answers[row.trace_id]; other = pool.trace != row.trace_id
        same = other & (pool.q == row.question_id); far = other & (pool.q != row.question_id) & pool.has_note
        for r, k in enumerate(KEYS):
            if a.get(k) in pool.ci:
                X[i, pool.ci[a[k]], r] = 1; X[i, pool.ci[a[k]], 3] = (.6, .25, .15)[r]
        if a.get('first_code') in pool.ci:
            X[i, :, 4] = fam == a['first_code'].split('-')[0]
        X[i, :, 5] = a.get('first_code') == 'NONE'; X[i, pool.ci['NONE'], 6] = 1
        n = same.sum(); counts = pool.onehot[same].sum(0)
        X[i, :, 7] = counts / n if n else 0; X[i, :, 8] = np.log1p(counts); X[i, :, 9] = np.log1p(n)
        if n:
            best = np.where(pool.onehot[same] > 0, S_tx[i, same][:, None], 0).max(0)
            X[i, :, 10] = best; X[i, :, 11] = best / best.max() if best.max() > 0 else 0
            noted = same & pool.has_note
            if noted.any():
                X[i, :, 12] = np.where(pool.onehot[noted] > 0, S_note[i, noted][:, None], 0).max(0)
            X[i, :, 17] = S_tx[i, same].max()
        if far.any():
            X[i, :, 13] = np.where(pool.onehot[far] > 0, S_note[i, far][:, None], 0).max(0)
        X[i, :, 14] = S_def[i]
        topic = other & (pool.topic == row.topic_name)
        X[i, :, 15] = pool.onehot[topic].mean(0) if topic.any() else 0
        X[i, :, 16] = glob
    return X


def scores(model, X):
    p = model.predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(X.shape[:2])
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='runs/cpu_judge'); ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    taxonomy = read_csv(ROOT / 'data/raw/taxonomy_v1.csv'); classes = label_space(taxonomy)
    train = read_csv(ROOT / 'data/raw/train.csv'); test = read_csv(ROOT / 'data/raw/test.csv'); folds = make_folds(train)
    fit = train.loc[folds != 0].reset_index(drop=True); dev = train.loc[folds == 0].reset_index(drop=True)
    q_train, q_test = qwen_answers(QWEN['train']), qwen_answers(QWEN['test'])
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    defs = {r.id: f'{r.name}. {r.description}' for r in taxonomy.itertuples(index=False)}
    def_text = [defs.get(c, 'No major error. The working is correct or has only minor slips.') for c in classes]

    def run(pool_frame, train_frame, apply):
        tx_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=2).fit(pool_frame.transcription_steps.astype(str))
        note_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([n for n in map(root_note, pool_frame.itertuples(index=False)) if n])
        def_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(def_text); D = def_vec.transform(def_text)
        pool = Pool(pool_frame, classes, tx_vec, note_vec)
        rows = train_frame[train_frame.trace_id.isin(q_train)].reset_index(drop=True)
        X = features(rows, q_train, pool, tx_vec, note_vec, D, def_vec)
        y = np.zeros(X.shape[:2], dtype=int); y[np.arange(len(rows)), [pool.ci[c] for c in rows.root_code]] = 1
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_leaf_nodes=15, l2_regularization=1., min_samples_leaf=20,
                                               class_weight={0: 1., 1: 8.}, random_state=a.seed)
        model.fit(X.reshape(-1, X.shape[-1]), y.reshape(-1))
        return [(frame, scores(model, features(frame, ans, pool, tx_vec, note_vec, D, def_vec))) for frame, ans in apply], model, (X, y)

    # dev check: evidence and training from fit rows only
    devq = dev[dev.trace_id.isin(q_train)].reset_index(drop=True)
    res, model, _ = run(fit, fit, [(devq, q_train)])
    P = res[0][1]
    pts = row_scores(devq.root_code.to_numpy(), topk(P, classes))
    slices = json.loads((ROOT / 'data/v7/dev_slices.json').read_text()); sl = np.array([slices[t] for t in devq.trace_id])
    qfirst = np.array([[q_train[t].get(k) for k in KEYS] for t in devq.trace_id])
    qpts = np.array([next((1 / (j + 1) for j, c in enumerate(r) if c == t), 0.) for r, t in zip(qfirst, devq.root_code)])
    part = lambda s: {'all': float(s.mean()), 'none': float(s[sl == 'none'].mean()), 'seen_pair': float(s[sl == 'seen_pair'].mean()),
                      'hard': float(s[np.isin(sl, ['new_pair', 'unseen_q'])].mean())}
    report = {'dev_rows': len(devq), 'cpu_judge_alone': part(pts), 'qwen_alone_same_rows': part(qpts), 'features': NAMES}
    save_bundle(out / 'dev.npz', devq.trace_id.tolist(), classes, P, kind='cpu_judge_dev')
    # test: evidence and training from all 1,500 rows
    testq = test[test.trace_id.isin(q_test)].reset_index(drop=True)
    res, _, _ = run(train, train, [(testq, q_test)])
    save_bundle(out / 'test.npz', testq.trace_id.tolist(), classes, res[0][1], kind='cpu_judge_test')
    report['test_rows'] = len(testq)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
