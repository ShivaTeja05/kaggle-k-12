"""T3c: question identifiability of an annotator-style error statement (gold root 'text') vs the gold working, fit -> dev (seen questions)."""
import json, re, sys
import numpy as np
sys.path[:0] = ['src', 'runs/arch/label-structure']
from common import load, ROOT
from sklearn.feature_extraction.text import TfidfVectorizer
tr, tax, classes, folds, dev, fit = load()
def rt(m, span):
    try: ms = json.loads(m)
    except Exception: return ''
    return ' '.join((x.get('span', '') + ' ' if span else '') + x.get('text', '') for x in ms if x.get('root'))
fe, de = fit[fit.root_code != 'NONE'], dev[(dev.root_code != 'NONE') & dev.question_id.isin(set(fit.question_id))]
out = {'n_dev_rows': len(de), 'n_fit_rows': len(fe)}
for name, f in [('gold_working', lambda r: r.transcription_steps), ('root_text', lambda r: rt(r.misconceptions, False)),
                ('root_text_digits_masked', lambda r: re.sub(r'\d', '0', rt(r.misconceptions, False))), ('span+text', lambda r: rt(r.misconceptions, True))]:
    v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=1, sublinear_tf=True)
    X = v.fit_transform([f(r) for r in fe.itertuples()]); Q = v.transform([f(r) for r in de.itertuples()])
    top = fe.question_id.to_numpy()[(Q @ X.T).toarray().argmax(1)]
    out[name] = round(float((top == de.question_id.to_numpy()).mean()), 3)
print(out); json.dump(out, open(ROOT / 'runs/deep/approach-W1/t3c_gold_text_ident.json', 'w'), indent=1)
