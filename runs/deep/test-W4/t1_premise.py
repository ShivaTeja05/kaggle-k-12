"""W4 test T1 (CPU, adversarial): is the Claude pool's late ROOT_LINE prior a labelling artifact, a pool-photo property,
or a property that TEST shares? Approach A's positional reweight (pool -> gold last-line shares) is only right if
(1) the same labeller (Claude Opus zero-shot, same prompt family) on train-distribution photos is NOT late-biased, and
(2) the test photos look like train, not like the pool.
Outputs runs/deep/test-W4/t1_premise.json. No API, no GPU, no data leaves the machine."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT / 'runs/deep/test-W4'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(ROOT / 'runs/deep/symbolic')]
from common import load
from symutil import root_lines
from k12.core import read_csv, label_space, aligned_bundle, normalize

tr, tax, classes, folds, dev, fit = load()
ci = {c: i for i, c in enumerate(classes)}
ids_of = lambda tx: [int(m.group(1)) for m in (re.match(r'\s*L(\d+)', l) for l in str(tx).splitlines() if l.strip()) if m]
first = lambda s: (int(re.findall(r'L(\d+)', str(s or ''))[0]) if re.findall(r'L(\d+)', str(s or '')) else None)
BUCKETS = ((1, 3), (4, 6), (7, 10), (11, 999))
bname = lambda n: next(f'{lo}-{hi if hi < 999 else "+"}' for lo, hi in BUCKETS if lo <= n <= hi)
rng = np.random.default_rng(2026)


def rows_from(pairs, extra=None):
    """pairs: list of (root_line, line_ids, group, code) -> DataFrame with is_last, bucket."""
    out = []
    for r, L, g, c in pairs:
        if r is None or r not in L or not L:
            continue
        out.append(dict(last=int(r == L[-1]), b=bname(len(L)), n=len(L), g=g, code=c))
    return pd.DataFrame(out)


def share(df):
    return round(float(df['last'].mean()), 3) if len(df) else None


def by_bucket(df):
    return {b: [int((df.b == b).sum()), share(df[df.b == b])] for b in [bname(lo) for lo, _ in BUCKETS]}


def std_share(df, ref):
    """last-line share standardised to ref's bucket mix."""
    w = ref.b.value_counts(normalize=True)
    s = df.groupby('b')['last'].mean()
    return float(sum(w[b] * s.get(b, np.nan) for b in w.index if b in s.index) / sum(w[b] for b in w.index if b in s.index))


def boot_diff(a, b, ref, B=2000):
    """bootstrap (grouped by g) of std_share(a)-std_share(b); a and b resampled independently by group."""
    ga = {k: v for k, v in a.groupby('g')}; gb = {k: v for k, v in b.groupby('g')}
    ka, kb = list(ga), list(gb); d = []
    for _ in range(B):
        sa = pd.concat([ga[ka[i]] for i in rng.integers(0, len(ka), len(ka))])
        sb = pd.concat([gb[kb[i]] for i in rng.integers(0, len(kb), len(kb))])
        d.append(std_share(sa, ref) - std_share(sb, ref))
    return [round(float(x), 3) for x in np.quantile(d, [.025, .975])]


res = {}
# ---- gold
def gold_pairs(df):
    out = []
    for r in df.itertuples():
        if r.root_code == 'NONE':
            continue
        s, _ = root_lines(r.misconceptions)
        out.append((min(s) if s else None, ids_of(r.transcription_steps), r.question_id, r.root_code))
    return out
G_tr = rows_from(gold_pairs(tr)); G_fit = rows_from(gold_pairs(fit)); G_dev = rows_from(gold_pairs(dev))

# ---- Claude zero-shot on dev (same labeller, same system prompt family as the pool) and on the pool
def claude_rows(path, qmap, keep=None):
    out = []; allrows = []
    for l in open(path):
        x = json.loads(l); a = x.get('answer') or {}
        t = x['trace_id']
        if keep is not None and t not in keep:
            continue
        if not a.get('first_code'):
            continue
        L = ids_of(a.get('transcription', ''))
        allrows.append(dict(t=t, n=len(L), none=a['first_code'] == 'NONE', code=a['first_code'], g=qmap.get(t)))
        if a['first_code'] != 'NONE':
            out.append((first(a.get('root_line')), L, qmap.get(t), a['first_code']))
    return rows_from(out), pd.DataFrame(allrows)
qmap = dict(zip(tr.trace_id, tr.question_id))
unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv')
qmap.update(dict(zip(unl.trace_id, unl.question_id)))
C_dev, C_dev_all = claude_rows(ROOT / 'runs/claude/dev_opus.jsonl', qmap, set(dev.trace_id))
C_pool, C_pool_all = claude_rows(ROOT / 'runs/claude/unl_all_v4.jsonl', qmap)

res['last_line_share'] = {
    'gold_train_err': [len(G_tr), share(G_tr)], 'gold_dev_err': [len(G_dev), share(G_dev)],
    'claude_zeroshot_dev_called_err': [len(C_dev), share(C_dev)], 'claude_pool_called_err': [len(C_pool), share(C_pool)]}
res['by_bucket'] = {'gold_train': by_bucket(G_tr), 'gold_dev': by_bucket(G_dev),
                    'claude_zeroshot_dev': by_bucket(C_dev), 'claude_pool': by_bucket(C_pool)}
res['std_to_gold_train_bucket_mix'] = {k: round(std_share(v, G_tr), 3) for k, v in
                                       [('gold_train', G_tr), ('gold_dev', G_dev), ('claude_zeroshot_dev', C_dev), ('claude_pool', C_pool)]}
res['diff_ci_question_grouped'] = {
    'claude_dev_minus_gold_dev': [round(std_share(C_dev, G_tr) - std_share(G_dev, G_tr), 3), boot_diff(C_dev, G_dev, G_tr)],
    'pool_minus_claude_dev': [round(std_share(C_pool, G_tr) - std_share(C_dev, G_tr), 3), boot_diff(C_pool, C_dev, G_tr, B=500)]}
print(json.dumps({k: res[k] for k in ('last_line_share', 'std_to_gold_train_bucket_mix', 'diff_ci_question_grouped')}, indent=1))

# ---- same-question, same-bucket control: pool error rows re-weighted to gold train's (question, bucket) mix
gcell = G_tr.groupby(['g', 'b']).size(); pcell = C_pool.groupby(['g', 'b'])
common_cells = [k for k in gcell.index if k in pcell.groups]
num = sum(gcell[k] * C_pool.loc[pcell.groups[k], 'last'].mean() for k in common_cells)
den = sum(gcell[k] for k in common_cells)
gnum = sum(gcell[k] * G_tr[(G_tr.g == k[0]) & (G_tr.b == k[1])]['last'].mean() for k in common_cells)
res['question_bucket_matched'] = {'cells': len(common_cells), 'gold_rows_covered': int(den),
                                  'gold_share': round(gnum / den, 3), 'pool_share_reweighted': round(num / den, 3)}
# also code-matched within question+bucket
gcell3 = G_tr.groupby(['g', 'b', 'code']).size(); pcell3 = C_pool.groupby(['g', 'b', 'code'])
cc3 = [k for k in gcell3.index if k in pcell3.groups]
d3 = sum(gcell3[k] for k in cc3)
res['question_bucket_code_matched'] = {
    'cells': len(cc3), 'gold_rows_covered': int(d3),
    'gold_share': round(sum(gcell3[k] * G_tr[(G_tr.g == k[0]) & (G_tr.b == k[1]) & (G_tr.code == k[2])]['last'].mean() for k in cc3) / d3, 3),
    'pool_share_reweighted': round(sum(gcell3[k] * C_pool.loc[pcell3.groups[k], 'last'].mean() for k in cc3) / d3, 3)}
print(res['question_bucket_matched'], res['question_bucket_code_matched'])

# same comparison with Claude zero-shot dev vs pool on the same (question,bucket) cells (same labeller both sides)
dcell = C_dev.groupby(['g', 'b']).size()
cc = [k for k in dcell.index if k in pcell.groups]
dd = sum(dcell[k] for k in cc)
res['claude_dev_vs_pool_question_bucket_matched'] = {
    'cells': len(cc), 'dev_rows_covered': int(dd),
    'claude_dev_share': round(sum(dcell[k] * C_dev[(C_dev.g == k[0]) & (C_dev.b == k[1])]['last'].mean() for k in cc) / dd, 3),
    'pool_share_reweighted': round(sum(dcell[k] * C_pool.loc[pcell.groups[k], 'last'].mean() for k in cc) / dd, 3)}
print(res['claude_dev_vs_pool_question_bucket_matched'])

# ---- does TEST look like train or like the pool? (label-free)
test = read_csv(ROOT / 'data/raw/test.csv')
txd = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json')); txt = json.load(open(ROOT / 'runs/pod/tx/tx_test.json'))
nl = lambda d, ids: np.array([len(ids_of(d[t])) for t in ids if t in d])
qd, qt = nl(txd, dev.trace_id), nl(txt, test.trace_id)
def lc(a):
    return dict(n=int(len(a)), mean=round(float(a.mean()), 2), le3=round(float((a <= 3).mean()), 3), ge7=round(float((a >= 7).mean()), 3))
res['line_counts'] = {
    'qwen_tx_dev': lc(qd), 'qwen_tx_test': lc(qt),
    'claude_tx_dev_all': lc(C_dev_all.n.to_numpy()), 'claude_tx_pool_all': lc(C_pool_all.n.to_numpy()),
    'gold_tx_train_all': lc(np.array([len(ids_of(x)) for x in tr.transcription_steps]))}
# bucket mix of the two transcribers side by side
res['line_count_bucket_mix'] = {k: {bname(lo): round(float(np.mean([lo <= v <= hi for v in a])), 3) for lo, hi in BUCKETS}
                                for k, a in [('qwen_tx_dev', qd), ('qwen_tx_test', qt), ('claude_tx_dev', C_dev_all.n.to_numpy()),
                                             ('claude_tx_pool', C_pool_all.n.to_numpy())]}
# grade mix
pool_lab = unl[unl.trace_id.isin(set(C_pool_all.t))]
gm = lambda s: {int(k): round(float(v), 3) for k, v in s.astype(int).value_counts(normalize=True).sort_index().items()}
res['grade_mix'] = {'train': gm(tr.grade), 'test': gm(test.grade), 'pool_labelled': gm(pool_lab.grade), 'pool_all': gm(unl.grade)}
tv = lambda a, b: round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)), 3)
res['grade_tv_distance'] = {'test_vs_train': tv(res['grade_mix']['test'], res['grade_mix']['train']),
                            'test_vs_pool_labelled': tv(res['grade_mix']['test'], res['grade_mix']['pool_labelled'])}
# question mix
qtr = set(tr.question_id); qpl = set(pool_lab.question_id)
res['question_overlap'] = {'test_rows_on_train_questions': round(float(test.question_id.isin(qtr).mean()), 3),
                           'test_rows_on_pool_labelled_questions': round(float(test.question_id.isin(qpl).mean()), 3),
                           'pool_labelled_rows_on_train_questions': round(float(pool_lab.question_id.isin(qtr).mean()), 3)}
qm = lambda s: s.value_counts(normalize=True).to_dict()
res['question_tv_distance'] = {'test_vs_train': tv(qm(test.question_id), qm(tr.question_id)),
                               'test_vs_pool_labelled': tv(qm(test.question_id), qm(pool_lab.question_id)),
                               'dev_vs_fit (same distribution reference)': tv(qm(dev.question_id), qm(fit.question_id))}
# label marginals: NONE and ANS-01 shares
TEST_MEMBERS = {'m32': 'runs/pod/text32-full/test.npz', 'm8': 'runs/pod/text-full/test.npz', 'vl': 'runs/pod/full-distill/test.npz'}
W = {'m32': .35, 'm8': .30, 'vl': .35}
PT = {k: aligned_bundle(ROOT / p, test.trace_id.tolist(), classes) for k, p in TEST_MEMBERS.items()}
lp = sum(W[k] * np.log(PT[k]) for k in W); PTb = normalize(np.exp(lp - lp.max(1, keepdims=True)))
from common import blend
PD = blend(dev.trace_id.tolist(), classes)
def marg(P):
    t1 = P.argmax(1)
    return {'top1_NONE': round(float((t1 == ci['NONE']).mean()), 3), 'top1_ANS-01': round(float((t1 == ci['ANS-01']).mean()), 3),
            'mean_p_ANS-01': round(float(P[:, ci['ANS-01']].mean()), 3)}
res['label_marginals'] = {
    'gold_train': {'NONE': round(float((tr.root_code == 'NONE').mean()), 3), 'ANS-01': round(float((tr.root_code == 'ANS-01').mean()), 3)},
    'claude_zeroshot_dev': {'NONE': round(float(C_dev_all.none.mean()), 3), 'ANS-01': round(float((C_dev_all.code == 'ANS-01').mean()), 3)},
    'claude_pool': {'NONE': round(float(C_pool_all.none.mean()), 3), 'ANS-01': round(float((C_pool_all.code == 'ANS-01').mean()), 3)},
    'blend_dev_pred': marg(PD), 'blend_test_pred (full-train members)': marg(PTb),
    'vl_test_pred': marg(PT['vl']), 'm32_test_pred': marg(PT['m32'])}
print(json.dumps({k: res[k] for k in ('line_counts', 'line_count_bucket_mix', 'grade_tv_distance', 'question_overlap', 'question_tv_distance', 'label_marginals')}, indent=1))
json.dump(res, open(OUT / 't1_premise.json', 'w'), indent=1)
