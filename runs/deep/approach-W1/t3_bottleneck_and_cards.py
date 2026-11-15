"""T3a: how well does the bottleneck input (root line + reasoning) identify the question, vs the full working?
   Nearest neighbour (char 2-5 TF-IDF cosine) from dev Claude outputs into the Claude pool (question labels from unlabelled.csv).
T3b: do code cards built from other questions' gold annotator texts make a code easier to match than name+description alone?
   Unsupervised cosine(reasoning, card) used to rank the blend+Sinkhorn top-10 on dev error rows (Claude dev reasoning = oracle diagnosis)."""
import json, re, sys
from collections import defaultdict, Counter
import numpy as np
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability']
from lcommon import get_all, boot_ci, ROOT
from k12.core import row_scores
from sklearn.feature_extraction.text import TfidfVectorizer
D = get_all(); dev, fit, y, classes, tax, unl, PS = D['dev'], D['fit'], D['y'], D['classes'], D['tax'], D['unl'], D['PS']
res = {}
def line(tx, rl):
    m = re.match(r'L(\d+)', str(rl or ''))
    if not m: return ''
    for l in str(tx).splitlines():
        if l.startswith(f'L{m.group(1)} |'): return l.split('|', 1)[1]
    return ''
pool = {}
for l in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(l); a = r.get('answer') or {}
    if a.get('first_code'): pool[r['trace_id']] = a
qid = dict(zip(unl.trace_id, unl.question_id)); ids = [t for t in pool if t in qid]
cd = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(ROOT / 'runs/claude/dev_opus.jsonl')}
dq = dev.question_id.to_numpy(); covered = np.array([q in set(qid.values()) for q in dq])
nodig = lambda s: re.sub(r'\d', '0', s)
def ident(fx, strip=False):
    f = (lambda s: nodig(fx(s))) if strip else fx
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
    X = v.fit_transform([f(pool[t]) for t in ids]); Q = v.transform([f(cd[t]) for t in dev.trace_id])
    S = (Q @ X.T).toarray(); top = np.array([qid[ids[j]] for j in S.argmax(1)])
    return round(float((top == dq)[covered].mean()), 3)
bott = lambda a: line(a.get('transcription', ''), a.get('root_line')) + ' ' + re.sub(r'\b[A-Za-z]{2,3}-\d{2}\b', '', a.get('reasoning', ''))
res['T3a_question_id_acc_on_covered_dev_rows'] = {
    'n_covered': int(covered.sum()), 'n_questions_in_pool': len(set(qid.values())),
    'full_working': ident(lambda a: a.get('transcription', '')),
    'bottleneck_rootline+reasoning': ident(bott),
    'bottleneck_digits_masked': ident(bott, True),
    'reasoning_only_digits_masked': ident(lambda a: re.sub(r'\b[A-Za-z]{2,3}-\d{2}\b', '', a.get('reasoning', '')), True)}
print(res)
# T3b cards
desc = {r['id']: f"{r['name']}. {r['description']}" for _, r in tax.iterrows()}; desc['NONE'] = 'No major error, the working is correct or has only minor slips.'
texts = defaultdict(list)
for q, c, m in zip(fit.question_id, fit.root_code, fit.misconceptions):
    try: ms = json.loads(m)
    except Exception: continue
    for x in ms:
        if x.get('root') and x.get('text'): texts[c].append((q, x['text']))
def card(c, q, k):
    others = [t for qq, t in texts.get(c, []) if qq != q]
    return desc.get(c, c) + ' ' + ' '.join(others[:k])
v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
v.fit(list(desc.values()) + [t for l in texts.values() for _, t in l] + [cd[t].get('reasoning', '') for t in dev.trace_id])
err = y != 'NONE'; pair = Counter(zip(fit.question_id, fit.root_code)); qs = set(fit.question_id)
newp = np.array([c != 'NONE' and pair[(q, c)] == 0 for q, c in zip(dq, y)])
bs = row_scores(y, D['preds']['blend_sk']); g = dq
for k in [0, 3, 8]:
    preds = []
    for i, t in enumerate(dev.trace_id):
        cand = [classes[j] for j in np.argsort(-PS[i])[:10] if classes[j] != 'NONE']
        r = v.transform([cd[t].get('reasoning', '')]); C = v.transform([card(c, dq[i], k) for c in cand])
        s = (C @ r.T).toarray().ravel(); preds.append([cand[j] for j in np.argsort(-s)[:3]])
    s = row_scores(y, np.array(preds))
    res[f'T3b_card_k{k}'] = {'error_rows': round(float(s[err].mean()), 3), 'new_pair_rows': round(float(s[newp].mean()), 3),
                             'seen_pair_rows': round(float(s[err & ~newp].mean()), 3)}
    if k: res[f'T3b_card_k{k}_minus_k0_newpair'] = boot_ci(s[newp], res['_k0'][newp], g[newp]); res[f'T3b_card_k{k}_minus_k0_err'] = boot_ci(s[err], res['_k0'][err], g[err])
    else: res['_k0'] = s
res.pop('_k0'); res['blend_sk_ref'] = {'error_rows': round(float(bs[err].mean()), 3), 'new_pair_rows': round(float(bs[newp].mean()), 3)}
print(json.dumps(res, indent=1)); json.dump(res, open(ROOT / 'runs/deep/approach-W1/t3_bottleneck_and_cards.json', 'w'), indent=1)
