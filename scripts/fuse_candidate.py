"""Stage B fusion: raw dev blend x candidate-student bundle^w, then Sinkhorn; cross-fitted gates and test staging (CPU).

Never submits. --stage-test writes a CSV and metrics.json with "submitted": false.
--subset scores a bundle that covers only some dev rows (PLAN step 4 interim abort): student vs blend
recall@3 on the unseen-pair rows it covers; no fusion, gates or staging.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_claude import claude_scores
from k12.core import aligned_bundle, fingerprint, label_space, make_folds, normalize, read_csv, row_scores, save_bundle, topk, write_submission

ROOT = Path(__file__).resolve().parents[1]
DEV = [('runs/pod/text32/validation_tx_dev_epoch_1.npz', .35), ('runs/pod/tx/text_tx_dev_e3.npz', .30),
       ('runs/pod/dev-distill/validation_epoch_2.npz', .35)]
TEST = [('runs/pod/text32-full/test.npz', .35), ('runs/pod/text-full/test.npz', .30), ('runs/pod/full-distill/test.npz', .35)]
WS, LAMS, REF_LAM = [0, .25, .5, .75, 1, 1.5], [0, .25, .5, .75], .5


def geo(paths, ids, classes):
    lp = sum(w * np.log(aligned_bundle(ROOT / p, ids, classes)) for p, w in paths)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


def marginals(codes, classes):
    ci = {c: i for i, c in enumerate(classes)}
    pi = np.bincount(codes.map(ci), minlength=len(classes)) + .5
    return pi / pi.sum()


def sinkhorn(P, target, lam, iters):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))


def fuse(P0, PB, w, lam, target, iters):
    F = normalize(P0 * np.clip(PB, 1e-12, None) ** w) if w else P0
    return sinkhorn(F, target, lam, iters) if lam else F


def boot_ci(new_pts, ref_pts, groups, B=4000, seed=2026):
    d = np.asarray(new_pts) - np.asarray(ref_pts)
    g = pd.DataFrame({'g': groups, 'd': d}).groupby('g').d.agg(['sum', 'size']).to_numpy()
    s = np.random.default_rng(seed).integers(0, len(g), (B, len(g)))
    b = g[s, 0].sum(1) / g[s, 1].sum(1)
    return float(d.mean()), [float(x) for x in np.quantile(b, [.025, .975])]


def parsed(g):  # candidate_student appends "ANSWER:" itself when the model never wrote it, so prefer its flag
    return bool(g['parsed']) if 'parsed' in g else 'ANSWER:' in g['text']


def unseen_pairs(fit, y, qid):
    """Error rows whose code is new for a fit-seen question, and error rows of unseen questions (ensemble-anatomy)."""
    qc = fit.groupby('question_id').root_code.apply(set).to_dict(); err = y != 'NONE'
    seen = np.array([q in qc for q in qid]); inq = np.array([t in qc.get(q, set()) for t, q in zip(y, qid)])
    return seen & ~inq & err, ~seen & err


def interim(a, dev, fit, classes, sids):
    m = dev.trace_id.isin(sids).to_numpy()
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    P0 = geo(DEV, ids, classes); ref = sinkhorn(P0, marginals(fit.root_code, classes), REF_LAM, 30)
    PB = aligned_bundle(ROOT / a.student, [t for t, k in zip(ids, m) if k], classes)
    newc, unq = unseen_pairs(fit, y, qid); up = (newc | unq)[m]
    rec = lambda P: np.array([t in r for t, r in zip(y[m], topk(P, classes))])
    pts = lambda P: row_scores(y[m], topk(P, classes))
    rep = {'student': a.student, 'rows': int(m.sum()), 'unseen_pair_rows': int(up.sum()),
           'unseen_pair_recall3': {'student': float(rec(PB)[up].mean()), 'raw_blend': float(rec(P0[m])[up].mean()),
                                   'blend_sinkhorn': float(rec(ref[m])[up].mean())},
           'map3': {'student': float(pts(PB).mean()), 'raw_blend': float(pts(P0[m]).mean()), 'blend_sinkhorn': float(pts(ref[m]).mean())},
           'unseen_pair_map3': {'student': float(pts(PB)[up].mean()), 'raw_blend': float(pts(P0[m])[up].mean()),
                                'blend_sinkhorn': float(pts(ref[m])[up].mean())}}
    r = rep['unseen_pair_recall3']
    rep['interim_gate_student_beats_blends'] = bool(r['student'] > max(r['raw_blend'], r['blend_sinkhorn']))
    if a.generations:
        gens = [json.loads(x) for x in Path(ROOT / a.generations).read_text().splitlines() if x.strip()]
        rep['parse_rate'] = float(np.mean([parsed(g) for g in gens]))
    return rep


def order_scores(generations, cand_dev, ids, y, classes):
    """Per-order MAP@3 and parse rate from the student's generations.jsonl and the dev candidate file."""
    cands = {r['trace_id']: r['codes_per_prompt'] for r in map(json.loads, Path(cand_dev).read_text().splitlines())}
    gens = [json.loads(x) for x in Path(generations).read_text().splitlines() if x.strip()]
    base = min(int(g['order']) for g in gens)
    ci, ri = {c: i for i, c in enumerate(classes)}, {t: i for i, t in enumerate(ids)}
    out = {'parse_rate': float(np.mean([parsed(g) for g in gens])), 'generations': len(gens)}
    for o in sorted({int(g['order']) for g in gens}):
        P, seen = np.full((len(ids), len(classes)), 1e-4), np.zeros(len(ids), bool)
        for g in gens:
            if int(g['order']) != o or g['trace_id'] not in ri:
                continue
            i, codes = ri[g['trace_id']], cands[g['trace_id']][o - base]
            lp = g['letter_probs']
            for k, c in enumerate(codes):
                P[i, ci[c]] = lp.get(chr(65 + k), 1e-4) if isinstance(lp, dict) else lp[k]
            seen[i] = True
        s = row_scores(y, topk(normalize(P), classes))
        out[f'order{o}'] = {'map3': float(s[seen].mean()), 'rows': int(seen.sum()),
                            'parse_rate': float(np.mean([parsed(g) for g in gens if int(g['order']) == o]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', required=True, help='dev bundle of the candidate student (P_B)')
    ap.add_argument('--generations', help='student generations.jsonl on dev, for parse rate and per-order scores')
    ap.add_argument('--cand-dev', default='data/v5/cand_dev.jsonl')
    ap.add_argument('--claude', default='runs/claude/dev_opus.jsonl')
    ap.add_argument('--report', default='runs/arch/student/fuse_dev.json')
    ap.add_argument('--stage-test', help='test bundle of the candidate student; stages the fused submission')
    ap.add_argument('--out', default='artifacts/submission-3way-student')
    ap.add_argument('--allow-failed-gates', action='store_true', help='stage even if a gate fails (for pipeline checks only)')
    ap.add_argument('--subset', action='store_true', help='the student bundle covers only some dev rows: interim report only')
    a = ap.parse_args()
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv')); C = np.asarray(classes)
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    with np.load(ROOT / a.student, allow_pickle=False) as b:
        sids = set(b['ids'].tolist())
    if sids != set(ids):
        if not a.subset or not sids <= set(ids):
            raise SystemExit(f'student bundle covers {len(sids & set(ids))} of {len(ids)} dev ids ({len(sids - set(ids))} foreign); '
                             'use --subset for an interim check on a dev subset')
        rep = interim(a, dev, fit, classes, sids)
        Path(ROOT / a.report).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps(rep, indent=1))
        return
    P0, PB, pi_fit = geo(DEV, ids, classes), aligned_bundle(ROOT / a.student, ids, classes), marginals(fit.root_code, classes)
    pts = lambda P: row_scores(y, topk(P, classes))
    ref = pts(sinkhorn(P0, pi_fit, REF_LAM, 30))
    grid = [(w, l) for w in WS for l in LAMS]
    cache = {k: fuse(P0, PB, *k, pi_fit, 30) for k in grid}
    S = {k: pts(v) for k, v in cache.items()}
    qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
    R, choices = np.empty_like(P0), []
    for f in range(5):
        m = mf != f
        k = grid[int(np.argmax([S[k][m].mean() for k in grid]))]; choices.append(k); R[~m] = cache[k][~m]
    R0 = np.empty_like(P0)
    for f in range(5):
        m = mf != f; R0[~m] = cache[(0, LAMS[int(np.argmax([S[(0, l)][m].mean() for l in LAMS]))])][~m]
    cf = pts(R); delta, ci95 = boot_ci(cf, ref, qid)
    best = grid[int(np.argmax([S[k].mean() for k in grid]))]
    # unseen pairs: error rows whose code is new for a fit-seen question, plus error rows of unseen questions
    newc, unq = unseen_pairs(fit, y, qid); up = newc | unq
    ct = C[claude_scores(ROOT / a.claude, ids, classes)[0].argmax(1)]
    bt, st = C[P0.argmax(1)], C[PB.argmax(1)]; dis = bt != ct
    agree = lambda t, m: float((t == ct)[m].mean())
    sb = pts(PB)
    gates = {'a_delta_ge_0.015_ci_lo_gt_0': bool(delta >= .015 and ci95[0] > 0),
             'b_unseen_pair_map3_ge_0.50': bool(cf[up].mean() >= .50),
             'c_student_claude_top1_agree_unseen_pair_ge_0.65': bool(agree(st, up) >= .65),
             'd_student_claude_top1_agree_blend_claude_disagree_ge_0.45': bool(agree(st, dis) >= .45)}
    rep = {'student': a.student, 'student_sha256': fingerprint(ROOT / a.student),
           'reference_blend_sinkhorn_lam0.5': float(ref.mean()), 'raw_blend': float(pts(P0).mean()),
           'crossfit_map3': float(cf.mean()), 'delta_vs_reference': delta, 'ci95': ci95,
           'reference_crossfit_lam_only': float(pts(R0).mean()), 'delta_vs_crossfit_reference': boot_ci(cf, pts(R0), qid),
           'fold_choices_w_lam': [list(map(float, k)) for k in choices], 'chosen_w_lam_full_dev': list(map(float, best)),
           'in_sample_chosen': float(S[best].mean()),
           'in_sample_by_w_at_lam0.5': {str(w): float(S[(w, .5)].mean()) for w in WS},
           'student_alone': {'map3': float(sb.mean()), 'unseen_pair_map3': float(sb[up].mean()),
                             'claude_agree_unseen_pair_106': agree(st, up), 'claude_agree_new_code_84': agree(st, newc),
                             'claude_agree_blend_claude_disagree': agree(st, dis)},
           'rows': {'unseen_pair': int(up.sum()), 'new_code_seen_question': int(newc.sum()),
                    'unseen_question': int(unq.sum()), 'blend_claude_disagree': int(dis.sum())},
           'unseen_pair_map3': {'raw_blend': float(pts(P0)[up].mean()), 'reference': float(ref[up].mean()),
                                'crossfit_fused': float(cf[up].mean())},
           'fused_crossfit_claude_agree': {'unseen_pair': agree(C[R.argmax(1)], up), 'blend_claude_disagree': agree(C[R.argmax(1)], dis)},
           'gates': gates, 'gates_passed': all(gates.values())}
    if a.generations:
        rep['student_generations'] = order_scores(ROOT / a.generations, ROOT / a.cand_dev, ids, y, classes)
    if a.stage_test:
        if not rep['gates_passed'] and not a.allow_failed_gates:
            raise SystemExit('gates failed; not staging (use --allow-failed-gates only for pipeline checks)')
        te = read_csv(ROOT / 'data/raw/test.csv'); tids = te.trace_id.tolist()
        PT, PBT = geo(TEST, tids, classes), aligned_bundle(ROOT / a.stage_test, tids, classes)
        w, lam = best
        Q = fuse(PT, PBT, w, lam, marginals(tr.root_code, classes), 50)
        out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
        write_submission(te, Q, classes, out / 'submission.csv')
        save_bundle(out / 'test.npz', tids, classes, Q, kind='blend_x_student_sinkhorn_test', w=w, lam=lam)
        a0, a1 = topk(PT, classes), topk(Q, classes)
        m = {'rows': len(te), 'base': 'text32-full .35, text-full .30, full-distill .35 geometric (artifacts/submission-3way-blend)',
             'student_test_bundle': a.stage_test, 'student_test_sha256': fingerprint(ROOT / a.stage_test),
             'w': w, 'lambda': lam, 'post_processing': 'Sinkhorn toward the 1,500-row train root_code marginals (+0.5), 50 iterations, label-free on test',
             'development': {k: rep[k] for k in ('crossfit_map3', 'delta_vs_reference', 'ci95', 'gates', 'gates_passed')},
             'test_top1_changed_vs_blend': float((a0[:, 0] != a1[:, 0]).mean()),
             'test_none_top1_share': float((a1[:, 0] == 'NONE').mean()),
             'script': 'scripts/fuse_candidate.py', 'submission_sha256': fingerprint(out / 'submission.csv'), 'submitted': False}
        (out / 'metrics.json').write_text(json.dumps(m, indent=2) + '\n')
        rep['staged'] = {'dir': str(out), 'submission_sha256': m['submission_sha256'], 'w': w, 'lambda': lam,
                         'test_top1_changed_vs_blend': m['test_top1_changed_vs_blend']}
    Path(ROOT / a.report).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
