"""Score a zero-shot vision model's ranked codes on the development fold, alone and fused with the submitted system.

Inputs are jsonl files in the runs/claude format ({"trace_id", "answer": {first_code, second_code, third_code}},
optionally "samples": [answers]). Several files are pooled by adding their rank shares.

The fusion is submitted_system x model^v, then Sinkhorn(lambda); (floor, v, lambda) are cross-fitted over 5 question
folds, the same way the ranker weight was chosen. Usage:
  python scripts/score_vlm_dev.py name=runs/deep/q38/pod_out/q38_dev_think.jsonl [name2=...] --report reports/q38_dev.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals, sinkhorn, unseen_pairs
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk

ROOT = F.ROOT
SHARES = (.6, .25, .15)
RANKER = 'runs/deep/pointwise/pod_out/pw_full/predict_pw_dev/predictions.npz'


def shares(path, ids, classes, first_only=False):
    """Rows x classes rank shares (no floor), the share of rows with an answer, and the samples used per row."""
    index = {c: i for i, c in enumerate(classes)}
    answers = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        got = [a for a in (record.get('samples') or [record.get('answer') or {}]) if a.get('first_code') in index]
        answers[record['trace_id']] = got[:1] if first_only else got
    S = np.zeros((len(ids), len(classes))); used = np.zeros(len(ids), int)
    for row, trace in enumerate(ids):
        got = answers.get(trace, [])
        used[row] = len(got)
        for answer in got:
            seen = set()
            for share, key in zip(SHARES, ('first_code', 'second_code', 'third_code')):
                code = answer.get(key)
                if code in index and code not in seen:
                    S[row, index[code]] += share / len(got); seen.add(code)
    return S, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='name=path.jsonl')
    ap.add_argument('--report')
    ap.add_argument('--first-only', action='store_true', help='use only the first sample of each row')
    ap.add_argument('--prefix', action='store_true', help='partial run: score only the leading block of rows that are all answered '
                    '(rows are sent in order, so later finished rows are the quick ones and would bias the score)')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    pts = lambda P: row_scores(y, topk(P, classes))
    newc, unq = unseen_pairs(fit, y, qid); hard = newc | unq; none = y == 'NONE'; seen = ~none & ~hard
    grp = np.array(json.load(open(ROOT / 'runs/deep/catalog/catalog_meta.json'))['grp'])
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER, ids, classes)
    qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])

    def crossfit(cache, rows=None):
        """Each fold's rows get the setting that scored best on the other folds (only `rows` count when given)."""
        rows = np.ones(len(ids), bool) if rows is None else rows
        keys = list(cache); S = {k: pts(v) for k, v in cache.items()}
        R = P0.copy(); chosen = []
        for f in range(5):
            m = mf != f; k = keys[int(np.argmax([S[k][m & rows].mean() for k in keys]))]; chosen.append(k); R[~m] = cache[k][~m]
        return pts(R), chosen

    def parts(s, mask=None):
        m = np.ones(len(s), bool) if mask is None else mask
        out = {'rows': int(m.sum()), 'map3': float(s[m].mean()), 'none': float(s[m & none].mean()), 'seen_pair': float(s[m & seen].mean()),
               'hard': float(s[m & hard].mean())}
        out.update({f'hard_{x}': float(s[m & hard & (grp == x)].mean()) for x in 'ABC' if (m & hard & (grp == x)).any()})
        return out

    base, base_choice = crossfit({(w, l): fuse(P0, P1, w, l, pi, 30) for w in F.WS for l in F.LAMS})
    report = {'submitted_system_crossfit': parts(base), 'submitted_choice_w_lam': [list(k) for k in base_choice]}
    named = {}
    for spec in a.runs:
        name, path = spec.split('=', 1)
        named[name] = shares(ROOT / path, ids, classes, a.first_only)
    if len(named) > 1:
        named['pooled'] = (sum(S for S, _ in named.values()) / len(named), np.minimum.reduce([u for _, u in named.values()]))
    for name, (S, used) in named.items():
        have = used > 0
        if a.prefix and not have.all():
            have[int(np.argmin(have)):] = False
        alone = pts(normalize(S + 1e-6))
        entry = {'answered_rows': int(have.sum()), 'samples_per_row_mean': float(used[have].mean()) if have.any() else 0.,
                 'alone_on_answered': parts(alone, have), 'top1_on_answered': float((topk(normalize(S + 1e-6), classes)[:, 0] == y)[have].mean()),
                 'submitted_on_same_rows': parts(base, have)}
        if have.sum() >= 150:  # enough rows for a cross-fitted fusion; everything below is on the answered rows only
            cache = {}
            for floor in (.01, .03, .08):
                Q = normalize(S + floor)
                for v in (0, .25, .5, .75, 1, 1.5, 2):
                    for lam in F.LAMS:
                        cache[(floor, v, lam)] = sinkhorn(normalize(P0 * P1 ** .5 * Q ** v), pi, lam, 30) if lam else normalize(P0 * P1 ** .5 * Q ** v)
            fused, chosen = crossfit(cache, have)
            both, both_choice = crossfit({(floor, v, lam): (sinkhorn(normalize(P0 * normalize(S + floor) ** v), pi, lam, 30) if lam else normalize(P0 * normalize(S + floor) ** v))
                                          for floor in (.01, .03, .08) for v in (0, .25, .5, .75, 1, 1.5, 2) for lam in F.LAMS}, have)
            entry.update({'fused_with_submitted_crossfit': parts(fused, have), 'fused_choice_floor_v_lam': [list(k) for k in chosen],
                          'fused_minus_submitted': boot_ci(fused[have], base[have], qid[have]),
                          'alone_minus_submitted': boot_ci(alone[have], base[have], qid[have]),
                          'fused_minus_alone': boot_ci(fused[have], alone[have], qid[have]),
                          'fused_with_blend_only_crossfit': parts(both, have), 'blend_only_choice': [list(k) for k in both_choice]})
        report[name] = entry
    for label, path in (('claude_opus_photo', 'runs/claude/dev_opus.jsonl'), ('claude_opus_transcript', 'runs/claude/dev_opus_text_tx.jsonl')):
        if (ROOT / path).exists():
            S, used = shares(ROOT / path, ids, classes)
            report[label] = parts(pts(normalize(S + 1e-6)), used > 0)
            for name, (_, used_model) in named.items():  # the same reference on each run's answered rows
                if not (used_model > 0).all():
                    report[f'{label}_on_{name}_rows'] = parts(pts(normalize(S + 1e-6)), (used > 0) & (used_model > 0))
    if a.report:
        (ROOT / a.report).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
