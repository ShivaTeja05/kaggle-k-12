"""T1b: robustness of T1's gold-text result. Text only (no span), and text with every taxonomy name word masked."""
import json, re, sys
from collections import Counter
import numpy as np
sys.path[:0] = ['src', 'runs/arch/label-structure', 'runs/deep/learnability', 'runs/deep/approach-W1']
import importlib.util
from lcommon import get_all, boot_ci, ROOT
from k12.core import row_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
D = get_all(); dev, fit, y, classes, tax = D['dev'], D['fit'], D['y'], D['classes'], D['tax']
pair = Counter(zip(fit.question_id, fit.root_code)); qseen = set(fit.question_id)
bins = np.array(['NONE' if c == 'NONE' else 'q_unseen' if q not in qseen else ('pair0', 'pair1', 'pair2')[min(pair[(q, c)], 2)] if pair[(q, c)] < 3 else 'pair3+' for q, c in zip(dev.question_id, dev.root_code)])
names = set(w for n in tax.iloc[:, 1].astype(str) for w in re.findall(r'[a-z]{4,}', n.lower()))
def rt(m, span, mask):
    try: ms = json.loads(m)
    except Exception: return ''
    s = ' '.join((x.get('span', '') + ' . ' if span else '') + x.get('text', '') for x in ms if x.get('root'))
    return ' '.join('_' if w.lower() in names else w for w in s.split()) if mask else s
def run(span, mask):
    fe = fit[fit.root_code != 'NONE']; X = [rt(m, span, mask) for m in fe.misconceptions]; k = [i for i, x in enumerate(X) if x.strip()]
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(C=10., max_iter=3000).fit(hstack([w.fit_transform([X[i] for i in k]), c.fit_transform([X[i] for i in k])]).tocsr(), fe.root_code.to_numpy()[k])
    err = y != 'NONE'; Xd = [rt(m, span, mask) for m in dev.misconceptions[err]]
    p = lr.predict_proba(hstack([w.transform(Xd), c.transform(Xd)]).tocsr()); pred = np.array([lr.classes_[np.argsort(-r)[:3]] for r in p])
    s = np.zeros(len(y)); s[err] = row_scores(y[err], pred)
    out = {b: round(float(s[bins == b].mean()), 3) for b in ['q_unseen', 'pair0', 'pair1', 'pair2', 'pair3+']}
    bs = row_scores(y, D['preds']['blend_sk']); g = dev.question_id.to_numpy(); m = bins == 'pair0'
    out['vs_blend_sk@pair0'] = boot_ci(s[m], bs[m], g[m]); out['n_masked_words'] = len(names) if mask else 0
    return out
res = {f'span={a},mask={b}': run(a, b) for a, b in [(False, False), (False, True), (True, True)]}
print(json.dumps(res, indent=1)); json.dump(res, open(ROOT / 'runs/deep/approach-W1/t1b_gtext_variants.json', 'w'), indent=1)
