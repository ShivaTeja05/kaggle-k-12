"""Part 2: label-ambiguity ceiling. TF-IDF code similarity from taxonomy text; where do our and Claude's errors fall?"""
from lcommon import *

D = get_all(); y = D['y']; dev, tax, classes, preds = D['dev'], D['tax'], D['classes'], D['preds']
q = dev.question_id.to_numpy(); err = y != 'NONE'
S = {k: score(y, v) for k, v in preds.items()}

# ---------- TF-IDF (word uni+bigram, and char 3-5) built in numpy
codes = tax.id.astype(str).tolist()
text = {r.id: f"{r.name}. {r.name}. {r.description}" for r in tax.itertuples()}
STOP = set('a an the of to in or and is as by for on not be with its it at from that this than e g eg vs instead into are was were which when one two'.split())
def word_feats(s):
    w = [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]
    return w + [a + '_' + b for a, b in zip(w, w[1:])]
def char_feats(s):
    s = ' ' + re.sub(r'\s+', ' ', s.lower()) + ' '
    return [s[i:i + n] for n in (3, 4, 5) for i in range(len(s) - n + 1)]
def tfidf(featfn):
    docs = [Counter(featfn(text[c])) for c in codes]
    vocab = sorted(set().union(*docs)); vi = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(codes), len(vocab)))
    for i, d in enumerate(docs):
        for t, n in d.items():
            M[i, vi[t]] = 1 + np.log(n)
    df = (M > 0).sum(0); idf = np.log((1 + len(codes)) / (1 + df)) + 1
    M = M * idf; M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M @ M.T
SIMw, SIMc = tfidf(word_feats), tfidf(char_feats)
SIM = (SIMw + SIMc) / 2
ix = {c: i for i, c in enumerate(codes)}
sec = dict(zip(tax.id, tax.section))
off = SIM[~np.eye(len(codes), dtype=bool)]
nn_rank = np.argsort(-SIM, axis=1)  # col 0 is self
def nn_k(a, b):  # rank of b among a's neighbours (1 = nearest other code)
    return int(np.flatnonzero(nn_rank[ix[a]] == ix[b])[0])
pd.DataFrame(SIM, index=codes, columns=codes).round(3).to_csv(OUT / 'code_similarity_tfidf.csv')
nn_table = {c: [(codes[j], round(float(SIM[ix[c], j]), 3)) for j in nn_rank[ix[c]][1:4]] for c in codes}
thr = float(np.quantile(off, .95))  # "near-synonym" = top 5% of all off-diagonal pairs

# ---------- gold secondary annotations per dev row
def ann(s):
    try:
        return [m['category'] for m in json.loads(s) if not m.get('root')]
    except Exception:
        return []
sec_ann = [set(ann(s)) for s in dev.misconceptions]

def classify(t, p, secondary):
    """Relationship between true code t and wrong predicted top-1 p."""
    if t == 'NONE' or p == 'NONE':
        return 'none_vs_error'
    if p in secondary:
        return 'annotated_secondary'
    k = nn_k(t, p); s = SIM[ix[t], ix[p]]
    if k <= 3 or s >= thr:
        return 'near_synonym(top3nn_or_top5pct)'
    if fam(t) == fam(p):
        return 'same_family'
    if sec.get(t) == sec.get(p):
        return 'same_section'
    if (fam(t) in GENERIC) != (fam(p) in GENERIC):
        return 'generic_vs_specific'
    return 'unrelated'

cats = ['annotated_secondary', 'near_synonym(top3nn_or_top5pct)', 'same_family', 'same_section', 'generic_vs_specific', 'unrelated', 'none_vs_error']
res = {'similarity': {'near_synonym_threshold_top5pct': round(thr, 3), 'offdiag_mean': round(float(off.mean()), 3),
                      'examples_nearest': {c: nn_table[c] for c in ['ALG-02', 'ALG-03', 'ANS-01', 'ANS-02', 'WOP-01', 'WOP-02', 'CAL-01', 'OPS-01', 'MEA-01', 'MEA-05', 'FRM-01', 'NOT-01', 'NOT-02', 'PRO-03', 'CHK-01', 'OTH-00']}}}
models = ['blend', 'blend_sk', 'claude', 'claude_fewshot', 'claude_fable', 'scale_100']
rng = np.random.default_rng(1)
for k in models:
    P3 = preds[k]; top1 = P3[:, 0]; wrong = top1 != y
    lost = 1 - S[k]  # MAP points lost per row
    c_rows = Counter(); c_lost = Counter(); sims = []
    for i in np.flatnonzero(wrong):
        c = classify(y[i], top1[i], sec_ann[i]); c_rows[c] += 1; c_lost[c] += lost[i]
        if c != 'none_vs_error':
            sims.append(SIM[ix[y[i]], ix[top1[i]]])
    # random-confusion baseline: same true codes, wrong code drawn from the model's own wrong-top1 marginal
    base = []
    wrong_codes = [c for c in top1[wrong] if c != 'NONE']
    for _ in range(200):
        for i in np.flatnonzero(wrong & err):
            p = wrong_codes[rng.integers(len(wrong_codes))]
            if p != y[i] and p != 'NONE':
                base.append(SIM[ix[y[i]], ix[p]])
    res[k] = {'map3': round(float(S[k].mean()), 4), 'top1_wrong_rows': int(wrong.sum()), 'total_points_lost': round(float(lost.sum()), 2),
              'wrong_top1_by_relation_rows': dict(c_rows), 'points_lost_by_relation': {c: round(v, 2) for c, v in c_lost.items()},
              'points_lost_share': {c: round(c_lost[c] / lost.sum(), 3) for c in cats if c in c_lost},
              'mean_sim_true_vs_wrong_top1': round(float(np.mean(sims)), 3), 'random_baseline_sim': round(float(np.mean(base)), 3),
              'share_wrong_top1_near_synonym_or_secondary': round((c_rows['annotated_secondary'] + c_rows['near_synonym(top3nn_or_top5pct)']) / max(1, wrong.sum()), 3)}

# ---------- consensus disagreement: rows where all three Claude runs agree on top-1 but gold differs
cons = (preds['claude'][:, 0] == preds['claude_fewshot'][:, 0]) & (preds['claude'][:, 0] == preds['claude_fable'][:, 0])
cons_wrong = cons & (preds['claude'][:, 0] != y)
cw_rel = Counter(classify(y[i], preds['claude'][i, 0], sec_ann[i]) for i in np.flatnonzero(cons_wrong))
cons_wrong_any_top3 = cons_wrong & ~np.array([y[i] in preds['claude'][i] for i in range(len(y))])
also_blend = cons_wrong & (preds['blend_sk'][:, 0] == preds['claude'][:, 0])
res['claude_consensus'] = {'rows_all3_agree_top1': int(cons.sum()), 'acc_top1_when_all3_agree': round(float((preds['claude'][cons, 0] == y[cons]).mean()), 3),
                           'rows_all3_agree_and_wrong': int(cons_wrong.sum()), 'of_which_gold_not_in_claude_top3': int(cons_wrong_any_top3.sum()),
                           'of_which_blend_top1_also_same': int(also_blend.sum()), 'relation': dict(cw_rel),
                           'acc_top1_when_not_all_agree': round(float((preds['claude'][~cons, 0] == y[~cons]).mean()), 3)}
# ---------- irreducible-error band (estimates)
L = 1 - S['claude']; n = len(y)
amb = np.array([(preds['claude'][i, 0] != y[i]) and classify(y[i], preds['claude'][i, 0], sec_ann[i]) in ('annotated_secondary', 'near_synonym(top3nn_or_top5pct)') for i in range(n)])
amb_none = np.array([(preds['claude'][i, 0] != y[i]) and y[i] == 'NONE' and preds['claude'][i, 0] in sec_ann[i] for i in range(n)])  # minor annotated slip flagged as error
lower = cons_wrong & (amb | amb_none) & also_blend   # 3 Claude runs + our blend all agree on a defensible alternative
mid = cons_wrong & (amb | amb_none)
upper = (amb | amb_none) | cons_wrong_any_top3
band = {}
for nm, m in [('lower: 3 Claude runs + blend agree on a near-synonym/secondary', lower), ('mid: 3 Claude runs agree on a near-synonym/secondary', mid),
              ('upper: any Claude miss on near-synonym/secondary, or consensus miss with gold outside Claude top3', upper)]:
    band[nm] = {'rows': int(m.sum()), 'claude_points_lost_on_them': round(float(L[m].sum()), 2), 'map3_points_per_row': round(float(L[m].sum() / n), 4),
                'blend_sk_points_lost_on_them': round(float((1 - S['blend_sk'])[m].sum() / n), 4)}
res['irreducible_band_estimate'] = band
res['ceiling_if_band_irreducible'] = {k: round(1 - v['map3_points_per_row'], 3) for k, v in band.items()}
# how much of the blend-vs-Claude gap sits on rows where Claude's "win" is itself a near-synonym call
gap_rows = S['claude'] - S['blend_sk']
res['gap_split'] = {'total_gap_points_per_row': round(float(gap_rows.mean()), 4)}
blend_rel = np.array([classify(y[i], preds['blend_sk'][i, 0], sec_ann[i]) if preds['blend_sk'][i, 0] != y[i] else 'correct' for i in range(n)])
for c in ['correct'] + cats:
    m = blend_rel == c
    if m.any():
        res['gap_split'][f'blend_top1={c}'] = {'rows': int(m.sum()), 'gap_points_per_row': round(float(gap_rows[m].sum() / n), 4)}
json.dump(res, open(OUT / 'p2_ambiguity.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
