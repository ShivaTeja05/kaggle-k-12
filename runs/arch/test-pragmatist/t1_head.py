"""T1. Reproduce the proposal's reason-head result, then attack it.
 (a) reproduce: blend 0.718, blend+Sinkhorn, TF-IDF head on Claude dev reasoning (code ids stripped), fused + Sinkhorn
 (b) joint cross-fit of fusion weight AND Sinkhorn lambda (the proposal fixed lambda=0.5)
 (c) is the head reading a verbalised decision? strip the taxonomy NAMES (exact phrases) as well, and in a harsher
     variant every content word that occurs in any taxonomy name; measure agreement of head top-1 with Claude's top-1
     and the head's score on rows where Claude itself is wrong.
Dev head probabilities are cached in t1_heads.npz for the later tests."""
import sys, re
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from tp_common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy(); qid = dev.question_id.to_numpy()
ci = {c: i for i, c in enumerate(classes)}; mf = qfolds(qid)
B = blend(ids, classes); ref = row_scores(y, topk(B, classes))
pi = np.bincount(fit.root_code.map(ci), minlength=len(classes)) + .5; pi /= pi.sum()
top10 = np.argsort(-B, 1)[:, :10]; mask = np.zeros_like(B, bool); mask[np.arange(len(B))[:, None], top10] = True; mask[:, ci['NONE']] = True
fitpairs = set(zip(fit.question_id, fit.root_code))
up = np.array([(yy != 'NONE') and ((q, yy) not in fitpairs) for q, yy in zip(qid, y)])
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
cl_pts = row_scores(y, topk(C, classes)); cl_top1 = np.array(classes)[C.argmax(1)]

names = sorted({str(n).strip().lower() for n in tax.name if str(n).strip()}, key=len, reverse=True)
NAME = re.compile('|'.join(re.escape(n) for n in names), re.I)
STOP = set('a an the of in on to and or for with by as is are at from into not no its it be wrong error errors incorrect'.split())
words = sorted({w for n in names for w in re.findall(r'[a-z]+', n) if w not in STOP and len(w) > 2})
WORD = re.compile(r'\b(' + '|'.join(re.escape(w) for w in words) + r')\w*\b', re.I)
variants = {'ids_stripped': lambda s: s, 'names_stripped': lambda s: NAME.sub(' ', s),
            'name_words_stripped': lambda s: WORD.sub(' ', NAME.sub(' ', s))}

X, Y = [], []
for r in read_jsonl(ROOT/'runs/claude/unl_all_v4.jsonl'):
    a = r.get('answer') or {}
    if a.get('first_code') in ci and a.get('reasoning'): X.append(claude_text(a)); Y.append(a['first_code'])
dv = {r['trace_id']: claude_text(r['answer']) for r in read_jsonl(ROOT/'runs/claude/dev_opus.jsonl')}
Dtxt = [dv[t] for t in ids]


def fuse_scores(Pm, sinks):
    W = [0, .2, .35, .5, .65, .8]
    out = {}
    for w in W:
        F = geo([(B, 1-w), (Pm, w)]) if w else B
        for l in sinks:
            out[(w, l)] = row_scores(y, topk(sinkhorn(F, pi, l) if l else F, classes))
    return out


res = {'n_train': len(Y), 'n_name_words': len(words), 'ref': ref.mean(), 'ref_unseen_pair': ref[up].mean(), 'n_unseen_pair': int(up.sum())}
LAMS = [0, .25, .5, .75, 1.]
bs = {l: row_scores(y, topk(sinkhorn(B, pi, l) if l else B, classes)) for l in LAMS}
o, ch = crossfit(bs, mf)
res['blend_sinkhorn_fixed0.5'] = bs[.5].mean(); res['blend_sinkhorn_cf'] = dict(map3=o.mean(), ci=boot_ci(o, ref, qid), lam=ch)
bsink_cf = o
heads = {}
for vname, fn in variants.items():
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    Xt = vec.fit_transform([fn(s) for s in X])
    lr = LogisticRegression(C=4, max_iter=2000, class_weight='balanced').fit(Xt, Y)
    Pd = np.full((len(ids), len(classes)), 1e-6); pp = lr.predict_proba(vec.transform([fn(s) for s in Dtxt]))
    for j, c in enumerate(lr.classes_): Pd[:, ci[c]] = pp[:, j]
    Pd = normalize(Pd); Pm = normalize(Pd*mask + 1e-9); heads[vname] = Pm
    a = row_scores(y, topk(Pm, classes)); h1 = np.array(classes)[Pm.argmax(1)]
    sc = fuse_scores(Pm, LAMS)
    fixed = {w: sc[(w, .5)] for w in [0, .2, .35, .5, .65, .8]}
    nos = {w: sc[(w, 0)] for w in [0, .2, .35, .5, .65, .8]}
    o_fixed, chf = crossfit(fixed, mf); o_nos, chn = crossfit(nos, mf); o_joint, chj = crossfit(sc, mf)
    cw = cl_pts == 0  # Claude's top-3 misses the truth
    res[vname] = dict(
        alone_top10=a.mean(), unseen_pair=a[up].mean(), agree_top1_with_claude=float((h1 == cl_top1).mean()),
        alone_on_rows_claude_top3_misses=a[cw].mean(), n_claude_misses=int(cw.sum()),
        blend_on_rows_claude_top3_misses=ref[cw].mean(),
        fused_cf_no_sink=dict(map3=o_nos.mean(), ci_vs_ref=boot_ci(o_nos, ref, qid), w=chn),
        fused_cf_sink_fixed0p5=dict(map3=o_fixed.mean(), ci_vs_ref=boot_ci(o_fixed, ref, qid),
                                    ci_vs_blend_sink_cf=boot_ci(o_fixed, bsink_cf, qid), w=chf),
        fused_joint_cf_w_lam=dict(map3=o_joint.mean(), ci_vs_ref=boot_ci(o_joint, ref, qid),
                                  ci_vs_blend_sink_cf=boot_ci(o_joint, bsink_cf, qid), unseen_pair=o_joint[up].mean(),
                                  cfg=[list(c) for c in chj]))
    print(vname, res[vname]['alone_top10'], res[vname]['unseen_pair'], res[vname]['fused_joint_cf_w_lam']['map3'], flush=True)
# Claude's own codes as the 'student' (oracle upper bound of the architecture)
Cm = normalize(C*mask + 1e-9); sc = fuse_scores(Cm, LAMS); o, chc = crossfit(sc, mf)
res['claude_codes_as_student'] = dict(alone_top10=row_scores(y, topk(Cm, classes)).mean(), joint_cf=o.mean(),
                                      ci_vs_ref=boot_ci(o, ref, qid), cfg=[list(c) for c in chc])
res['claude_alone_full'] = cl_pts.mean(); res['claude_unseen_pair'] = cl_pts[up].mean()
np.savez_compressed(OUT/'t1_heads.npz', ids=np.array(ids), **heads)
dump(res, 't1_head.json')
