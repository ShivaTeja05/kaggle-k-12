"""Router benchmark on the development fold: product fusion, hard router and soft fusion, side by side (CPU).

Never submits and never stages a file for submission. It uses the 301 development rows only (training photos held
out of every model here); no test row, no pod, no API and no network are touched. The only file written is --report.

Inputs are exactly those scripts/fuse_vlm.py uses for dev: the three blend bundles (fuse_candidate.DEV), the ranker's
dev bundle, and the reasoner's zero-shot answers read with score_vlm_dev.shares(first_only=True); rows are sent when
the submitted system's top probability is below --threshold (0.817).

Systems (ARCHITECTURE_REFORM_2026.md sections 3, 4.2 and 6), each with the four-slice report on all 301 rows (rows
without a reasoner answer keep the memoriser) and on the 270 answered rows, and a paired bootstrap over questions
against the cross-fitted product fusion:
  a   the submitted system at its fixed setting (w=0.5, lambda=0.75, 30 iterations, fit-row marginals);
  b   the reasoner alone;
  c   product fusion, (eps, v, lambda) cross-fitted over 5 question folds exactly as fuse_vlm.py does; it must
      reproduce reports/q38_fuse.json (0.8037037 cross-fitted, 0.8141975 in-sample, choice (0.01, 1, 0.5));
  d   hard router gated by the TRUE slice (needs the label: an upper reference, not deployable), on sent rows as
      section 4.2.1 defines it, and once more ignoring the send flag;
  e   hard router gated by the learned, cross-fitted gate (label "row is hard", sent rows, threshold 0.5);
  f   hard router with three hand gates built from test-time facts;
  g   soft fusion with the learned gate: beta fixed, and (eps, v, lambda, beta) chosen jointly inside the folds.
In d, e and f the rows that are not routed keep the submitted system.

What is and is not cross-fitted. The gate is always out-of-fold by question. In the g lines except the last, the
settings for a fold are chosen on the other four folds, but the gate scores of those four folds come from models that
saw the scored fold's gate labels, so the nesting is not strict; the last g line re-fits the gate without the scored
fold and is the one to quote. The hand gates of f are fixed thresholds, not fitted here, but 'confidence < 0.4' was
the best of twelve gates in the earlier hybrid probe on these same rows (design, section 4.2.3), so its line is not
an unworn estimate. d uses the label.
The design's 301-row line (0.780 / 0.924 / 0.860 / 0.571) picked settings on all rows of the other folds; the
reference here picks them on the answered rows, as fuse_vlm.py does; both are printed.

Usage:
  PYTHONPATH=src:scripts python scripts/eval_router.py [--report reports/router_benchmark.json]
"""
import argparse
import json
import warnings
from types import SimpleNamespace

import numpy as np

import fuse_candidate as F
from fuse_candidate import boot_ci, fuse, geo, marginals
from fuse_vlm import GRID, RANKER_DEV
from k12 import router
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk
from k12.slices import print_reports, slice_masks, slice_report
from score_vlm_dev import shares

ROOT = F.ROOT
ANSWERS = 'runs/deep/q38/pod_out/q38_dev_think.jsonl'
THRESHOLD = .817
BETAS = (0, .25, .5, .75, 1)
EXPECTED = {'crossfit': 0.8037037037037037, 'in_sample': 0.8141975308641974, 'choice': (.01, 1, .5)}  # reports/q38_fuse.json
LOOSE = ('settings are chosen on the other four folds, but those folds\' gate scores come from models that saw the scored '
         'fold\'s gate labels; the strictly nested line is "g soft, nested, gate re-fitted per fold"')


def load_dev(answers=ANSWERS, threshold=THRESHOLD):
    """Everything fuse_vlm.py builds for dev, plus the memoriser M, the slices and the question folds."""
    tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
    folds = make_folds(tr); dev = tr.loc[folds == 0].reset_index(drop=True); fit = tr.loc[folds != 0].reset_index(drop=True)
    ids, y, qid = dev.trace_id.tolist(), dev.root_code.to_numpy(), dev.question_id.to_numpy()
    P0, pi = geo(F.DEV, ids, classes), marginals(fit.root_code, classes)
    P1 = aligned_bundle(ROOT / RANKER_DEV, ids, classes)
    SUB = fuse(P0, P1, .5, .75, pi, 30)
    S, used = shares(ROOT / answers, ids, classes, first_only=True)
    have = used > 0; sent = have & (SUB.max(1) < threshold)
    with warnings.catch_warnings():  # the folds of fuse_vlm.py, line for line; pandas' string array only triggers a warning
        warnings.simplefilter('ignore', UserWarning)
        qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
    g = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([g[q] for q in qid])
    pts = lambda P: row_scores(y, topk(P, classes))
    return SimpleNamespace(classes=classes, ids=ids, y=y, qid=qid, fit=fit, P0=P0, P1=P1, pi=pi, M=normalize(P0 * P1 ** .5),
                           SUB=SUB, sub=pts(SUB), S=S, have=have, sent=sent, mf=mf, pts=pts, threshold=threshold,
                           masks=slice_masks(fit, y, qid), fit_codes=router.code_sets(fit.question_id, fit.root_code))


def balanced(D, P, lam):
    return router.sinkhorn(P, D.pi, lam, 30) if lam else P


def crossfit(D, score, keys):
    """fuse_vlm.py's loop: each fold's rows take the setting that scored best on the other folds' answered rows."""
    cf, chosen = D.sub.copy(), []
    for f in range(5):
        m = D.mf != f; k = keys[int(np.argmax([score[k][m & D.have].mean() for k in keys]))]; chosen.append(k); cf[~m] = score[k][~m]
    return cf, chosen


def product_crossfit(D):
    """Product fusion through k12.router, cross-fitted over fuse_vlm.GRID; also the setting chosen on all dev rows."""
    score = {(eps, v, lam): D.pts(balanced(D, router.product_fusion(D.M, router.reasoner_probs(D.S, eps), D.sent, v), lam))
             for eps, v, lam in GRID}
    cf, chosen = crossfit(D, score, GRID)
    best = GRID[int(np.argmax([score[k][D.have].mean() for k in GRID]))]
    return {'scores': cf, 'fold_choices': chosen, 'best': best, 'in_sample': float(score[best][D.have].mean()),
            'in_sample_scores': score[best], 'grid': score}


def formula_check(D):
    """Largest gap between k12.router's product fusion and the expression written in fuse_vlm.py, over the grid."""
    gap, same_scores = 0., True
    for eps, v, lam in GRID:
        Q = np.where(D.sent[:, None], normalize(D.S + eps), 1.)
        old = normalize(D.P0 * D.P1 ** .5 * Q ** v); old = F.sinkhorn(old, D.pi, lam, 30) if lam else old
        new = balanced(D, router.product_fusion(D.M, router.reasoner_probs(D.S, eps), D.sent, v), lam)
        gap = max(gap, float(np.abs(old - new).max())); same_scores &= bool((D.pts(old) == D.pts(new)).all())
    return {'max_abs_probability_gap': gap, 'row_scores_identical_on_every_grid_setting': same_scores}


def soft_grid(D, g, betas=BETAS):
    """Row scores of the balanced soft fusion for every (eps, v, lambda, beta); smaller beta comes first, so a tie
    in the selection goes to the setting nearer the product fusion."""
    return {(eps, v, lam, beta): D.pts(balanced(D, router.soft_fusion(D.M, router.reasoner_probs(D.S, eps), D.sent, v, g, beta), lam))
            for beta in betas for eps, v, lam in GRID}


def gate_inputs(D):
    Z = router.gate_features(D.SUB, D.M, D.S, D.sent, D.qid, D.fit_codes, D.classes)
    return Z, D.masks['hard'][D.sent].astype(float), D.qid[D.sent], D.mf[D.sent]


def learned_gate(D, l2):
    """Cross-fitted gate on sent rows over the fusion's question folds; 0 on every row that was not sent."""
    Z, label, groups, fold = gate_inputs(D)
    g = np.zeros(len(D.y)); g[D.sent] = router.crossfit_gate(Z, label, groups, l2=l2, folds=fold)
    return g


def strict_gate(D, l2, f):
    """Gate scores for scoring fold f with nothing fitted on fold f: its own rows are scored by one model fitted on
    the other four folds, and those four folds' rows are cross-fitted among themselves. Fold f's gate labels are
    never read (tests/test_router.py changes them and checks that no score moves)."""
    Z, label, groups, fold = gate_inputs(D)
    inner = fold != f; gs = np.zeros(len(Z))
    gs[inner] = router.crossfit_gate(Z[inner], label[inner], groups[inner], l2=l2, folds=fold[inner])
    gs[~inner] = router.predict_logistic(router.fit_logistic(Z[inner], label[inner], l2), Z[~inner])
    g = np.zeros(len(D.y)); g[D.sent] = gs
    return g


def soft_strict(D, l2, betas=BETAS):
    """Soft fusion with nothing fitted on the scored fold: for each outer fold the gate is re-fitted without it
    (strict_gate) and (eps, v, lambda, beta) are chosen on the other four folds."""
    cf, chosen = D.sub.copy(), []
    for f in range(5):
        score = soft_grid(D, strict_gate(D, l2, f), betas); keys = list(score)
        m = D.mf != f; k = keys[int(np.argmax([score[k][m & D.have].mean() for k in keys]))]; chosen.append(k); cf[~m] = score[k][~m]
    return cf, chosen


def crossfit_on_all_rows(D, score, keys):
    """The variant behind the design's 301-row line: settings chosen on all rows of the other folds, not only the
    answered ones. Reported for reconciliation; the reference stays fuse_vlm.py's loop."""
    cf = D.sub.copy()
    for f in range(5):
        m = D.mf != f; cf[~m] = score[keys[int(np.argmax([score[k][m].mean() for k in keys]))]][~m]
    return cf


def auc(label, score):
    """Probability that a positive row outscores a negative one (ties count half)."""
    pos, neg = score[label > .5], score[label < .5]
    return float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())


def gate_quality(D, gate):
    """Rows routed, and precision and recall of the gate for 'hard', on sent rows."""
    hard, routed = D.masks['hard'][D.sent], gate[D.sent] >= .5
    hit = int((routed & hard).sum())
    return {'sent_rows': int(D.sent.sum()), 'hard_sent_rows': int(hard.sum()), 'routed_rows': int(routed.sum()), 'routed_hard_rows': hit,
            'precision_hard': float(hit / routed.sum()) if routed.any() else None, 'recall_hard': float(hit / hard.sum()) if hard.any() else None}


def main():
    ap = argparse.ArgumentParser(description='Benchmark the product fusion, the hard router and the soft fusion on the 301 '
                                 'development rows (CPU). Uses dev rows only; never submits or stages anything.')
    ap.add_argument('--answers', default=ANSWERS, help='reasoner answers on dev (jsonl in the runs/claude format)')
    ap.add_argument('--threshold', type=float, default=THRESHOLD, help='rows are sent when the submitted top probability is below this')
    ap.add_argument('--report', default='reports/router_benchmark.json', help='where the JSON report is written')
    ap.add_argument('--route-eps', type=float, default=.01, help='floor of the reasoner probabilities on hard-routed rows (order is unaffected)')
    ap.add_argument('--l2', type=float, default=1., help='L2 strength of the learned gate')
    a = ap.parse_args()
    D = load_dev(a.answers, a.threshold)
    n, have, sent, hard = len(D.y), D.have, D.sent, D.masks['hard']
    prod = product_crossfit(D); ref = prod['scores']

    def entry(scores, **extra):
        e = {'all_rows': slice_report(scores, D.masks), 'answered_rows': slice_report(scores, D.masks, have),
             'minus_product_crossfit': {'all_rows': boot_ci(scores, ref, D.qid), 'answered_rows': boot_ci(scores[have], ref[have], D.qid[have])}}
        e.update(extra)
        return e

    def routed(gate):
        """Hard router: gated sent rows take the reasoner alone, every other row keeps the submitted system."""
        return D.pts(router.hard_route(D.SUB, router.reasoner_probs(D.S, a.route_eps), sent, gate))

    systems = {}
    systems['a submitted, fixed setting'] = entry(D.sub)
    alone = D.pts(normalize(D.S + 1e-6))
    systems['b reasoner alone (unanswered: submitted)'] = entry(np.where(have, alone, D.sub), note=f'the all-rows line gives the {int((~have).sum())} unanswered rows to the submitted system')
    systems['c product, cross-fitted (reference)'] = entry(ref, fold_choices_eps_v_lam=[list(k) for k in prod['fold_choices']])
    systems['c product, in-sample (0.01, 1, 0.5)'] = entry(prod['in_sample_scores'], note='in-sample: setting chosen on the rows it is scored on',
                                                           chosen_eps_v_lam_all_dev=list(prod['best']))
    truth = (hard & sent).astype(float)
    systems['d hard, TRUE slice gate (not deployable)'] = entry(routed(truth), gate=gate_quality(D, truth))
    every = D.pts(router.hard_route(D.SUB, router.reasoner_probs(D.S, a.route_eps), have, hard.astype(float)))
    systems['d same, ignoring the send flag'] = entry(every, note='every answered hard row goes to the reasoner, sent or not; '
                                                      'this is the 0.807 of the design, section 4.2.3', routed_rows=int((have & hard).sum()))
    g = learned_gate(D, a.l2)
    systems['e hard, learned cross-fitted gate'] = entry(routed(g), gate=gate_quality(D, g))
    conf = D.SUB.max(1)
    Z, label, _, _ = gate_inputs(D)
    new_code = np.zeros(n, bool); new_code[sent] = Z[:, router.FEATURES.index('reasoner_top_new_error_code')] > .5
    hand = {'f hard, confidence < 0.4': sent & (conf < .4), 'f hard, confidence < 0.6': sent & (conf < .6),
            'f hard, reasoner new code & conf < 0.6': sent & new_code & (conf < .6)}
    for name, gate in hand.items():
        systems[name] = entry(routed(gate.astype(float)), gate=gate_quality(D, gate.astype(float)),
                              note='fixed hand threshold, not fitted here; confidence < 0.4 was the best of twelve gates in the earlier hybrid probe on these same rows')
    soft = soft_grid(D, g)
    for beta in BETAS:
        keys = [k for k in soft if k[3] == beta]
        cf, chosen = crossfit(D, soft, keys)
        systems[f'g soft, beta={beta:g}, (eps,v,lam) cross-fit'] = entry(cf, fold_choices_eps_v_lam_beta=[list(k) for k in chosen], note=LOOSE)
    beta0_equals_product = bool((systems['g soft, beta=0, (eps,v,lam) cross-fit']['all_rows']['map3'] == systems['c product, cross-fitted (reference)']['all_rows']['map3'])
                                and all((soft[k + (0,)] == prod['grid'][k]).all() for k in GRID))
    keys = list(soft); cf, chosen = crossfit(D, soft, keys)
    best = keys[int(np.argmax([soft[k][have].mean() for k in keys]))]
    systems['g soft, (eps,v,lam,beta) nested'] = entry(cf, fold_choices_eps_v_lam_beta=[list(k) for k in chosen], note=LOOSE,
                                                      in_sample_choice=list(best), in_sample_on_answered=float(soft[best][have].mean()))
    cf, chosen = soft_strict(D, a.l2)
    systems['g soft, nested, gate re-fitted per fold'] = entry(cf, fold_choices_eps_v_lam_beta=[list(k) for k in chosen],
                                                              note='strict nesting: gate and settings fitted without the scored fold')

    top_sub, top_r = D.SUB[sent].argmax(1), D.S[sent].argmax(1); agree = np.zeros(n, bool); agree[sent] = top_sub == top_r
    split = lambda m: {'rows': int(m.sum()), 'submitted': float(D.sub[m].mean()), 'product_crossfit': float(ref[m].mean()),
                       'product_in_sample': float(prod['in_sample_scores'][m].mean()), 'gain_crossfit': float((ref - D.sub)[m].mean()),
                       'gain_crossfit_share_of_total': float((ref - D.sub)[m].sum() / (ref - D.sub)[have].sum())}
    monitor = {'sent_rows': int(sent.sum()), 'top1_agreement_submitted_vs_reasoner': float(agree[sent].mean()),
               'agree': split(sent & agree), 'disagree': split(sent & ~agree), 'unsent_answered': split(have & ~sent)}
    mu, sd, theta = router.fit_logistic(Z, label, a.l2)
    gate = {'features': list(router.FEATURES), 'l2': a.l2, 'folds': 'the fusion question folds (rng seed 7), restricted to sent rows',
            'crossfit_auc_hard': auc(label, g[sent]), 'threshold_0.5': gate_quality(D, g), 'hard_share_of_sent': float(label.mean()),
            'mean_gate_on_hard': float(g[sent][label > .5].mean()), 'mean_gate_on_other': float(g[sent][label < .5].mean()),
            'all_sent_rows_fit_standardised_coefficients': {'intercept': float(theta[0]), **{k: float(t) for k, t in zip(router.FEATURES, theta[1:])}},
            'note': 'coefficients are from one fit on all sent rows, for reading only; every score above uses cross-fitted gates'}
    cfr = slice_report(ref, D.masks, have)['map3']
    defaults = a.answers == ANSWERS and a.threshold == THRESHOLD  # the recorded numbers belong to these inputs only
    on_all = crossfit_on_all_rows(D, prod['grid'], GRID)
    repro = {'expected': {'crossfit': EXPECTED['crossfit'], 'in_sample': EXPECTED['in_sample'], 'choice': list(EXPECTED['choice'])},
             'crossfit_on_answered': cfr, 'in_sample_chosen': prod['in_sample'], 'chosen_eps_v_lam_all_dev': list(prod['best']),
             'reproduces_q38_fuse': bool(abs(cfr - EXPECTED['crossfit']) < 1e-6 and abs(prod['in_sample'] - EXPECTED['in_sample']) < 1e-6
                                         and tuple(prod['best']) == EXPECTED['choice']),
             'checked': defaults, 'soft_beta0_equals_product': beta0_equals_product, **formula_check(D),
             'design_301_row_line': {'design': [.780, .924, .860, .571], 'settings_chosen_on_all_rows': slice_report(on_all, D.masks),
                                     'settings_chosen_on_answered_rows_reference': slice_report(ref, D.masks),
                                     'note': 'the design table picked settings on all rows of the other folds; fuse_vlm.py and the reference here use the answered rows'}}
    saved = ROOT / 'reports/q38_fuse.json'
    if saved.exists():
        old = json.loads(saved.read_text())
        repro['q38_fuse_json'] = {'crossfit': old['fused_crossfit_on_answered']['map3'], 'in_sample': old['in_sample_chosen'],
                                  'choice': old['chosen_floor_v_lam_all_dev'],
                                  'fold_choices_match': old['fold_choices_floor_v_lam'] == [list(k) for k in prod['fold_choices']]}
    rep = {'script': 'scripts/eval_router.py', 'data': 'development fold only (301 training rows held out of every model); nothing submitted or staged',
           'answers': a.answers, 'threshold': a.threshold, 'reference': 'c product, cross-fitted (reference)',
           'bootstrap': 'paired over questions, 4000 resamples, seed 2026 (fuse_candidate.boot_ci): [difference, [2.5%, 97.5%]]',
           'rows': {'dev': n, 'answered': int(have.sum()), 'sent': int(sent.sum()),
                    'slices_all': {k: int(v.sum()) for k, v in D.masks.items()}, 'slices_answered': {k: int((v & have).sum()) for k, v in D.masks.items()},
                    'slices_sent': {k: int((v & sent).sum()) for k, v in D.masks.items()}},
           'reproduction': repro, 'systems': systems, 'gate': gate, 'complementarity': monitor}
    (ROOT / a.report).write_text(json.dumps(rep, indent=1) + '\n')

    for key, title in (('all_rows', f'All {n} dev rows (rows without a reasoner answer keep the memoriser)'),
                       ('answered_rows', f'The {int(have.sum())} rows the reasoner answered')):
        extra = {'header': 'minus product cross-fit [95% range]'}
        for name, e in systems.items():
            d, (lo, hi) = e['minus_product_crossfit'][key]; extra[name] = f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
        print_reports(title, [(name, e[key]) for name, e in systems.items()], extra)
    print('\nGates on the %d sent rows (%d hard): rows routed to the reasoner, precision and recall for "hard"' % (sent.sum(), label.sum()))
    for name, e in systems.items():
        if 'gate' in e:
            q = e['gate']; pr = '   n/a' if q['precision_hard'] is None else f"{q['precision_hard']:.3f}"
            print(f"  {name:<44} routed {q['routed_rows']:>3}  precision {pr}  recall {q['recall_hard']:.3f}")
    print(f"  learned gate, cross-fitted AUC for hard: {gate['crossfit_auc_hard']:.3f} (mean gate {gate['mean_gate_on_hard']:.3f} on hard, "
          f"{gate['mean_gate_on_other']:.3f} on the rest)")
    print(f"\nComplementarity on sent rows: top-1 agreement submitted vs reasoner {monitor['top1_agreement_submitted_vs_reasoner']:.3f}")
    for k in ('agree', 'disagree', 'unsent_answered'):
        s = monitor[k]
        print(f"  {k:<16} rows {s['rows']:>3}  submitted {s['submitted']:.4f}  product cross-fit {s['product_crossfit']:.4f}  "
              f"gain {s['gain_crossfit']:+.4f}  share of total gain {s['gain_crossfit_share_of_total']:+.2f}")
    print('\nNesting: every g line but the last chooses settings on folds whose gate scores saw the scored fold\'s gate labels; quote the last.'
          '\n  f uses fixed hand thresholds (confidence < 0.4 was the best of twelve in the earlier hybrid probe on these rows); d uses the true label.')
    print(f"\nReproduction of reports/q38_fuse.json: cross-fitted {cfr:.7f} (expected 0.8037037), in-sample {prod['in_sample']:.7f} "
          f"(expected 0.8141975), choice {tuple(prod['best'])} -> "
          f"{('OK' if repro['reproduces_q38_fuse'] else 'MISMATCH') if defaults else 'not checked (other answers or threshold)'}")
    r = repro['design_301_row_line']['settings_chosen_on_all_rows']
    print(f"  design's 301-row line 0.780 / 0.924 / 0.860 / 0.571 is the variant that picks settings on all rows of the other folds: "
          f"{r['map3']:.4f} / {r['none']:.4f} / {r['seen_pair']:.4f} / {r['hard']:.4f}; the reference above picks them on answered rows, as fuse_vlm.py does")
    print(f"  router vs fuse_vlm formula: max probability gap {repro['max_abs_probability_gap']:.2e}, row scores identical on every grid "
          f"setting: {repro['row_scores_identical_on_every_grid_setting']}; soft beta=0 equals product: {beta0_equals_product}")
    print(f'\nWrote {a.report}. Dev rows only; nothing submitted or staged.')
    if defaults and not repro['reproduces_q38_fuse']:
        raise SystemExit('product fusion did not reproduce reports/q38_fuse.json')


if __name__ == '__main__':
    main()
