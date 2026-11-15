"""W3 decisive CPU test: what is the transcriber-improvement (A1/A2) CEILING after blend+Sinkhorn, and how much of it
is (a) the verdict-phrase label leak in gold transcripts and (b) on rows a Claude-fidelity reader would actually fix?

Method: substitute the 32B member's predictions on GOLD transcripts (runs/pod/text32/validation_gold_epoch_1.npz) for its
machine-transcript predictions on a row subset S, keep the 8B and VL members fixed, re-blend (.35/.30/.35 geometric) and
re-apply Sinkhorn (lambda .5, target = fit label prior + .5). Optional 'all text members' variant also moves the 8B e3
member by the s100 (8B scaling run) gold/tx log-ratio -- an ESTIMATE, as in runs/arch/reading-impact.
Nothing is fitted on dev (weights/lambda were fixed before this test), so no cross-fitting is needed for the point
estimates; a 5-fold question-grouped cross-fitted lambda is also reported as a robustness check.
CIs: question-grouped paired bootstrap (boot_ci) vs the reference blend+Sinkhorn (0.737)."""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/label-structure')
from common import load, ROOT, boot_ci
from k12.core import aligned_bundle, topk, row_scores, normalize

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}
pi_fit = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi_fit /= pi_fit.sum()
def sinkhorn(P, lam=.5, iters=30):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = normalize(P * np.exp(b)); b += np.log(pi_fit / Q.mean(0))
    return normalize(P * np.exp(lam * b))
LP = lambda p: np.log(np.clip(aligned_bundle(ROOT/p, ids, classes), 1e-12, None))
L = dict(m32=LP('runs/pod/text32/validation_tx_dev_epoch_1.npz'), g32=LP('runs/pod/text32/validation_gold_epoch_1.npz'),
         m8=LP('runs/pod/tx/text_tx_dev_e3.npz'), vl=LP('runs/pod/dev-distill/validation_epoch_2.npz'),
         s100m=LP('runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'), s100g=LP('runs/pod/scaling/scale_100/validation_gold_epoch_1.npz'),
         s25m=LP('runs/pod/scaling/scale_25/validation_tx_dev_epoch_1.npz'), s25g=LP('runs/pod/scaling/scale_25/validation_gold_epoch_1.npz'))
def blendP(l32, l8):
    lp = .35 * l32 + .30 * l8 + .35 * L['vl']; return normalize(np.exp(lp - lp.max(1, keepdims=True)))
def score(P): return row_scores(y, topk(P, classes))
REF_RAW = blendP(L['m32'], L['m8']); REF = sinkhorn(REF_RAW); ref = score(REF)
print('reference blend', score(REF_RAW).mean().round(4), '-> +Sinkhorn', ref.mean().round(4))

# ---- row descriptors -------------------------------------------------------------------------------------------------
PH = re.compile(r'\[[^\]]*student[^\]]*answer[^\]]*correct[^\]]*\]', re.I)
g_ph = dev.transcription_steps.astype(str).str.contains(PH).to_numpy()
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json'))
m_ph = np.array([bool(PH.search(tx.get(t, ''))) for t in ids])
R = pd.read_csv(ROOT/'runs/arch/reading-impact/per_row_reading.csv', keep_default_na=False, dtype={'trace_id': str}).set_index('trace_id').reindex(ids)
w2 = pd.read_csv(ROOT/'runs/deep/approach-W3/w2_rows.csv').set_index('trace_id').reindex(ids)
w1 = pd.read_csv(ROOT/'runs/deep/approach-W3/w1_rows.csv', keep_default_na=False).set_index('trace_id').reindex(ids)
num = lambda c: pd.to_numeric(R[c], errors='coerce').to_numpy()
md, cd = num('m_span_dist'), num('c_span_dist')
err = y != 'NONE'; none = ~err
m_in = w2.m_span_in.to_numpy() == 1; c_in = w2.c_span_in.to_numpy() == 1
has_c = ~np.isnan(cd)
claude_better = err & has_c & (cd < md - 1e-9)          # Claude strictly closer to gold root span than machine
claude_recovers = err & ~m_in & c_in                     # exact span: machine lost, Claude kept
lost_both = err & ~m_in & ~c_in
m_exact = err & m_in
grp = w1.grp.to_numpy()

def variant(S, all_text=False, ratio='s100', lam=.5):
    l32 = np.where(S[:, None], L['g32'], L['m32'])
    l8 = L['m8']
    if all_text:
        l8 = np.where(S[:, None], L['m8'] + L[ratio + 'g'] - L[ratio + 'm'], L['m8'])
    P = blendP(l32, l8)
    return score(sinkhorn(P, lam) if lam else P)

subsets = {
 'ALL rows (full gold oracle)': np.ones(len(y), bool),
 'ALL rows except gold-verdict rows (leak removed)': ~g_ph,
 'gold-verdict NONE rows only (the leak)': g_ph,
 'NONE rows without verdict': none & ~g_ph,
 'ERROR rows only': err,
 'ERROR rows, Claude span strictly closer than machine (A1-reachable, by edit distance)': claude_better,
 'ERROR rows, exact span lost by machine & kept by Claude (A1-reachable, exact)': claude_recovers,
 'ERROR rows, exact span lost by both readers (A1-unreachable)': lost_both,
 'ERROR rows, machine already has exact span (reading not the issue for span)': m_exact,
 'M6 rows (w1 group)': grp == 'M6',
}
out = dict(reference=dict(blend=float(score(REF_RAW).mean()), blend_sinkhorn=float(ref.mean())), n_dev=len(y), variants={})
for name, S in subsets.items():
    rec = dict(n_rows_substituted=int(S.sum()), n_error=int((S & err).sum()))
    for tag, kw in [('32B_only', {}), ('all_text_s100_ESTIMATE', dict(all_text=True)), ('all_text_s25_ESTIMATE', dict(all_text=True, ratio='s25'))]:
        s = variant(S, **kw); d, c = boot_ci(s, ref, qid)
        rec[tag] = dict(map3=round(float(s.mean()), 4), delta=round(d, 4), ci95=[round(x, 4) for x in c],
                        rows_up=int((s > ref).sum()), rows_down=int((s < ref).sum()))
    # no-Sinkhorn comparison (vs raw blend) to see whether balancing absorbs the reading gain
    s0 = variant(S, lam=0); d0, c0 = boot_ci(s0, score(REF_RAW), qid)
    rec['32B_only_no_sinkhorn_vs_raw_blend'] = dict(delta=round(d0, 4), ci95=[round(x, 4) for x in c0])
    # single-model 32B gold-minus-machine on this subset (row-level)
    a, b = score(normalize(np.exp(L['g32']))), score(normalize(np.exp(L['m32'])))
    if S.sum(): rec['32B_single_gold_minus_machine_on_subset'] = round(float((a - b)[S].mean()), 4)
    out['variants'][name] = rec
    r = rec['32B_only']; e = rec['all_text_s100_ESTIMATE']
    print(f"{name[:88]:88s} n={rec['n_rows_substituted']:3d} 32B: {r['delta']:+.4f} {r['ci95']}  allText(est): {e['delta']:+.4f} {e['ci95']}")

# ---- robustness: question-grouped 5-fold cross-fitted lambda for full oracle and leak-free oracle --------------------
qs = dev.question_id.unique().copy(); np.random.default_rng(7).shuffle(qs)
gq = {q: i % 5 for i, q in enumerate(qs)}; mf = np.array([gq[q] for q in dev.question_id])
grid = [0, .25, .5, .75, 1.]
def crossfit(Praw):
    cache = {l: sinkhorn(Praw, l) if l else Praw for l in grid}; pts = np.empty(len(y))
    for f in range(5):
        m = mf != f; l = grid[int(np.argmax([score(cache[l])[m].mean() for l in grid]))]; pts[~m] = score(cache[l])[~m]
    return pts
ref_cf = crossfit(REF_RAW)
cf = {}
for name in ['ALL rows (full gold oracle)', 'ALL rows except gold-verdict rows (leak removed)', 'ERROR rows, Claude span strictly closer than machine (A1-reachable, by edit distance)']:
    S = subsets[name]; Praw = blendP(np.where(S[:, None], L['g32'], L['m32']), L['m8'])
    s = crossfit(Praw); d, c = boot_ci(s, ref_cf, qid); cf[name] = dict(ref_cf=round(float(ref_cf.mean()), 4), map3=round(float(s.mean()), 4), delta=round(d, 4), ci95=[round(x, 4) for x in c])
out['crossfit_lambda'] = cf
print('cross-fitted lambda:', json.dumps(cf, indent=0))

# ---- verdict-leak rows: what does the 32B do on them with gold vs machine reading -------------------------------------
P32g, P32m = normalize(np.exp(L['g32'])), normalize(np.exp(L['m32']))
ni = ci['NONE']
out['verdict_rows'] = dict(n=int(g_ph.sum()), p_none_32B_gold=round(float(P32g[g_ph, ni].mean()), 3), p_none_32B_machine=round(float(P32m[g_ph, ni].mean()), 3),
                           p_none_32B_gold_other_NONE=round(float(P32g[none & ~g_ph, ni].mean()), 3), p_none_32B_machine_other_NONE=round(float(P32m[none & ~g_ph, ni].mean()), 3),
                           ref_map_on_verdict_rows=round(float(ref[g_ph].mean()), 3), machine_tx_with_phrase=int(m_ph.sum()))
out['row_counts'] = dict(error=int(err.sum()), claude_better=int(claude_better.sum()), claude_recovers=int(claude_recovers.sum()), lost_both=int(lost_both.sum()),
                         machine_exact=int(m_exact.sum()), claude_transcript_missing_on_error=int((err & ~has_c).sum()),
                         claude_worse_than_machine=int((err & has_c & (cd > md + 1e-9)).sum()))
print(json.dumps(out['verdict_rows']), json.dumps(out['row_counts']))
json.dump(out, open(ROOT/'runs/deep/test-W3/t1_reader_ceiling.json', 'w'), indent=1)
