"""Stage 2 of the CPU test (the proposal's cpu_test, made adversarial).
Fusion is identical to design-structured/s6: error-conditional geometric mix of the 0.718 blend and the
mechanism-mapper distribution, NONE mass kept from the blend, weight chosen on an 11-point grid by
question-grouped 5-fold cross-fit (10 repeats), paired question-cluster bootstrap vs the blend.

E0  reproduce s6: Claude's dev mechanism through the mapper.
E1  is the 'mechanism' more than Claude's decision? Fuse Claude's ranked codes directly instead.
E2  degradation curves: replace a fraction r of rows' mechanisms with a wrong one, 4 corruption models:
      swap_any        another dev row's mechanism (random)
      swap_section    another dev row whose mechanism maps into the same taxonomy section
      q_shortcut      an out-of-fold Claude mechanism from an UNLABELLED photo of the same question
                      (a writer that regresses to the question's typical mechanism)
      blend_echo      a mechanism whose code is the blend's own top-1 error code (errors correlated with
                      the ensemble we already have)
    each with random row choice and with 'hard-first' row choice (corruptions land first on rows where
    the blend is wrong -- the realistic case, since those rows are hard for any 8B).
E3  realistic student proxies: existing Claude-distilled students (8B trained on 100% Claude labels, etc.).
    Keep Claude's mechanism where the student's top-1 agrees with Claude's first code, otherwise
    substitute a mechanism for the student's own top-1 code. Also fuse the student's decision directly.
E4  shortcut premise: on code-unseen rows, how often is each model's top-1 a code already seen for that
    question in fit?"""
import json, numpy as np
from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv, row_scores, topk

OUT = 'runs/arch/test-structured/'
train = read_csv('data/raw/train.csv'); tax = read_csv('data/raw/taxonomy_v1.csv'); classes = label_space(tax)
folds = make_folds(train); fit = train[folds != 0].reset_index(drop=True); dev = train[folds == 0].reset_index(drop=True)
y = dev.root_code.values; n = len(y); err = y != 'NONE'
seen = {}
for r in fit.itertuples(): seen.setdefault(r.question_id, set()).add(r.root_code)
bin_ = np.array(['NONE' if c == 'NONE' else ('code_seen' if c in seen.get(q, set()) else ('code_unseen' if q in seen else 'q_unseen'))
                 for c, q in zip(y, dev.question_id)])
C = np.load(OUT + 't1_mapper_cache.npz', allow_pickle=True)  # own file, written by t1
assert list(C['dev_ids']) == dev.trace_id.tolist()
err_codes = list(C['err_codes']); E = np.array(err_codes)
P_dev, P_pool, pool_q, pool_code = C['P_dev'], C['P_pool'], C['pool_q'], C['pool_code']
section = dict(zip(tax.id, tax.section))
ci = {c: i for i, c in enumerate(classes)}; ei = [ci[c] for c in err_codes]

L = lambda p: aligned_bundle(p, dev.trace_id.tolist(), classes)
blend = normalize(np.exp(.35*np.log(L('runs/pod/text32/validation_tx_dev_epoch_1.npz')) + .30*np.log(L('runs/pod/tx/text_tx_dev_e3.npz'))
                         + .35*np.log(L('runs/pod/dev-distill/validation_epoch_2.npz'))))
ref = row_scores(y, topk(blend, classes)); qs = dev.question_id.values; uq = np.unique(qs)
be = blend[:, ei] / blend[:, ei].sum(1, keepdims=True)
blend_wrong = ref < 1.0  # rows where blend top-1 is not the gold code

def fuse(P, w):
    e = normalize(np.exp((1 - w) * np.log(be) + w * np.log(np.clip(P, 1e-9, None)))); f = blend.copy()
    f[:, ei] = (1 - blend[:, [ci['NONE']]]) * e; return f
grid = np.round(np.arange(0, 1.01, .1), 2)
FOLDS = []
for rep in range(10):
    perm = np.random.default_rng(rep).permutation(uq); fq = {q: i % 5 for i, q in enumerate(perm)}
    FOLDS.append(np.array([fq[q] for q in qs]))
def crossfit(P):
    cache = {w: row_scores(y, topk(fuse(P, w), classes)) for w in grid}; res, ws = [], []
    for f in FOLDS:
        sc = np.zeros(n)
        for k in range(5):
            wb = max(grid, key=lambda w: (cache[w][f != k].mean(), -w)); sc[f == k] = cache[wb][f == k]; ws.append(wb)
        res.append(sc)
    return np.mean(res, 0), float(np.mean(ws)), {float(w): round(float(cache[w].mean()), 4) for w in grid}
G = {q: np.flatnonzero(qs == q) for q in uq}
def boot(d, B=4000):
    r = np.random.default_rng(1); m = [d[np.concatenate([G[q] for q in r.choice(uq, len(uq))])].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]
def mech_err_map(P):  # mapper quality on gold error rows
    return round(float(row_scores(y[err], topk(P[err], err_codes)).mean()), 4)
def summary(P, with_ci=True):
    sc, wbar, ins = crossfit(P); d = sc - ref
    o = {'crossfit': round(float(sc.mean()), 4), 'delta': round(float(d.mean()), 4), 'mean_w': round(wbar, 3),
         'mech_err_map3': mech_err_map(P), 'delta_by_bin': {b: round(float(d[bin_ == b].sum() / n), 4) for b in sorted(set(bin_))}}
    if with_ci: o['ci'] = boot(d)
    return o, d

out = {'reference_blend_dev': round(float(ref.mean()), 4), 'n_dev': n, 'bins': {b: int((bin_ == b).sum()) for b in sorted(set(bin_))}}
cl = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open('runs/claude/dev_opus.jsonl')}
claude_first = np.array([(cl.get(t) or {}).get('first_code', 'NONE') or 'NONE' for t in dev.trace_id])

# ---------- E0 / E1
out['E0_claude_mechanism'], d0 = summary(P_dev)
pc = np.full((n, len(classes)), 1e-4)
for i, t in enumerate(dev.trace_id):
    a = cl.get(t) or {}
    for s, c in zip((.6, .25, .15), [a.get(k) for k in ('first_code', 'second_code', 'third_code')]):
        if c in ci and pc[i, ci[c]] < s: pc[i, ci[c]] = s
pc = normalize(pc); Pc_err = pc[:, ei] / pc[:, ei].sum(1, keepdims=True)
out['E1_claude_codes_direct'], d1 = summary(Pc_err)
out['E1_claude_alone_dev'] = round(float(row_scores(y, topk(pc, classes)).mean()), 4)
d01 = d0 - d1; out['E1_mechanism_minus_codes'] = {'delta': round(float(d01.mean()), 4), 'ci': boot(d01)}
mt = E[P_dev.argmax(1)]; ce = claude_first != 'NONE'
out['E1_mapper_top1_equals_claude_first_code_on_claude_error_rows'] = round(float(np.mean(mt[ce] == claude_first[ce])), 4)
print(json.dumps({k: out[k] for k in out if k.startswith('E')}, indent=1), flush=True)

# ---------- replacement helpers
mapped_top = E[P_dev.argmax(1)]; mapped_sec = np.array([section[c] for c in mapped_top])
pool_by_q, pool_by_code, pool_by_qc = {}, {}, {}
for k, (q, c) in enumerate(zip(pool_q, pool_code)):
    pool_by_q.setdefault(q, []).append(k); pool_by_code.setdefault(c, []).append(k); pool_by_qc.setdefault((q, c), []).append(k)
dev_by_code = {}
for i, c in enumerate(mapped_top): dev_by_code.setdefault(c, []).append(i)
blend_top_err = E[be.argmax(1)]
UNIF = np.full(len(err_codes), 1 / len(err_codes))

def mech_for_code(i, code, rng, fallback):
    """An out-of-fold mechanism whose (Claude) code is `code`: same question first, then any question,
    then another dev row whose mechanism maps to it; else `fallback`."""
    for lst in (pool_by_qc.get((qs[i], code)), pool_by_code.get(code)):
        if lst: return P_pool[rng.choice(lst)]
    lst = [j for j in dev_by_code.get(code, []) if j != i]
    return P_dev[rng.choice(lst)] if lst else fallback

def replace(i, scheme, rng):
    if scheme == 'swap_any':
        return P_dev[rng.choice([j for j in range(n) if j != i])]
    if scheme == 'swap_section':
        lst = [j for j in np.flatnonzero(mapped_sec == mapped_sec[i]) if j != i]
        return P_dev[rng.choice(lst)] if lst else replace(i, 'swap_any', rng)
    if scheme == 'q_shortcut':
        lst = pool_by_q.get(qs[i])
        return P_pool[rng.choice(lst)] if lst else replace(i, 'swap_section', rng)
    if scheme == 'blend_echo':
        return mech_for_code(i, blend_top_err[i], rng, be[i])
    raise ValueError(scheme)

def corrupt(r, scheme, order, seed):
    rng = np.random.default_rng(1000 + seed); k = int(round(r * n))
    if order == 'random':
        S = rng.permutation(n)[:k]
    else:  # hard-first: blend-wrong rows first (random within), then the rest
        pri = np.where(blend_wrong, 0, 1) + rng.random(n) * .5; S = np.argsort(pri)[:k]
    P = P_dev.copy()
    for i in S: P[i] = replace(i, scheme, rng)
    return P

rates = np.round(np.arange(0, 1.01, .1), 2); SEEDS = 10
curves = {}
for order in ('random', 'hard_first'):
    for scheme in ('swap_any', 'swap_section', 'q_shortcut', 'blend_echo'):
        rows = []
        for r in rates:
            ds, ms, ws = [], [], []
            for s in range(SEEDS if 0 < r else 1):
                P = corrupt(r, scheme, order, s); sc, wbar, _ = crossfit(P)
                ds.append(sc - ref); ms.append(mech_err_map(P)); ws.append(wbar)
            dm = np.mean(ds, 0); per_seed = [float(d.mean()) for d in ds]
            rows.append({'rate': float(r), 'mech_err_map3': round(float(np.mean(ms)), 4),
                         'mech_quality_vs_claude': round(float(np.mean(ms)) / out['E0_claude_mechanism']['mech_err_map3'], 3),
                         'delta': round(float(dm.mean()), 4), 'delta_seed_range': [round(min(per_seed), 4), round(max(per_seed), 4)],
                         'ci_seedavg': boot(dm, 2000), 'mean_w': round(float(np.mean(ws)), 3)})
        curves[f'{order}/{scheme}'] = rows
        be_ = [x for x in rows if x['delta'] >= .02]
        print(order, scheme, [(x['rate'], x['mech_quality_vs_claude'], x['delta']) for x in rows], flush=True)
out['E2_curves'] = curves

def breakeven(rows, thr=.02, key='delta'):
    """Largest corruption rate that still has delta >= thr (and the mech quality at that point),
    linear interpolation between grid rates."""
    for a, b in zip(rows, rows[1:]):
        if a[key] >= thr > b[key]:
            t = (a[key] - thr) / (a[key] - b[key])
            return {'rate': round(a['rate'] + t * (b['rate'] - a['rate']), 3),
                    'mech_quality_vs_claude': round(a['mech_quality_vs_claude'] + t * (b['mech_quality_vs_claude'] - a['mech_quality_vs_claude']), 3)}
    return {'rate': None if rows[0][key] < thr else 1.0, 'mech_quality_vs_claude': None}
out['E2_breakeven_delta_0.02'] = {k: breakeven(v) for k, v in curves.items()}
def ci_breakeven(rows):  # largest rate whose seed-averaged CI lower bound is still > 0
    ok = [x for x in rows if x['ci_seedavg'][0] > 0]
    return max((x['rate'] for x in ok), default=None)
out['E2_last_rate_with_ci_lower_above_0'] = {k: ci_breakeven(v) for k, v in curves.items()}

# ---------- E3 realistic student proxies
students = {
    '8B_scale100_tx': 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz',
    '8B_scale100_goldtx': 'runs/pod/scaling/scale_100/validation_gold_epoch_1.npz',
    '8B_scale25_tx': 'runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz',
    '8B_blend_member_tx': 'runs/pod/tx/text_tx_dev_e3.npz',
    '32B_blend_member_tx': 'runs/pod/text32/validation_tx_dev_epoch_1.npz',
    'VL8B_distill': 'runs/pod/dev-distill/validation_epoch_2.npz',
    'blend_itself': None,
}
E3 = {}
for name, path in students.items():
    Ps = blend if path is None else L(path)
    top = np.array(classes)[Ps.argmax(1)]; Ps_err = Ps[:, ei] / Ps[:, ei].sum(1, keepdims=True)
    agree = top == claude_first
    o = {'student_dev_map3': round(float(row_scores(y, topk(Ps, classes)).mean()), 4),
         'top1_agree_with_claude_all': round(float(agree.mean()), 4),
         'top1_agree_with_claude_on_claude_error_rows': round(float(agree[ce].mean()), 4),
         'top1_agree_with_claude_on_code_unseen': round(float(agree[bin_ == 'code_unseen'].mean()), 4)}
    ds, ms = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(500 + s); P = P_dev.copy()
        for i in np.flatnonzero(~agree):
            P[i] = UNIF if top[i] == 'NONE' else mech_for_code(i, top[i], rng, Ps_err[i])
        sc, _, _ = crossfit(P); ds.append(sc - ref); ms.append(mech_err_map(P))
    dm = np.mean(ds, 0)
    o['simulated_writer'] = {'mech_err_map3': round(float(np.mean(ms)), 4),
                             'mech_quality_vs_claude': round(float(np.mean(ms)) / out['E0_claude_mechanism']['mech_err_map3'], 3),
                             'delta': round(float(dm.mean()), 4), 'ci': boot(dm),
                             'delta_by_bin': {b: round(float(dm[bin_ == b].sum() / n), 4) for b in sorted(set(bin_))}}
    if path is not None:
        o['direct_decision_fusion'], _ = summary(Ps_err)
    E3[name] = o; print(name, json.dumps(o), flush=True)
out['E3_student_proxies'] = E3

# ---------- E4 shortcut premise on code-unseen rows
cu = bin_ == 'code_unseen'
def shortcut_rate(top):
    return round(float(np.mean([t in seen.get(q, set()) for t, q in zip(top[cu], qs[cu])])), 4)
E4 = {'n_code_unseen': int(cu.sum()), 'claude_first': shortcut_rate(claude_first), 'claude_mechanism_mapper': shortcut_rate(mapped_top),
      'blend': shortcut_rate(np.array(classes)[blend.argmax(1)])}
for name, path in students.items():
    if path: E4[name] = shortcut_rate(np.array(classes)[L(path).argmax(1)])
out['E4_top1_is_question_seen_code_on_code_unseen_rows'] = E4
print(json.dumps(E4, indent=1))
json.dump(out, open(OUT + 't2_degrade.json', 'w'), indent=1)
