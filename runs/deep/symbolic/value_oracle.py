"""(i) Does knowing the root LINE help a cheap classifier pick the code?
TF-IDF + multinomial logistic regression, trained on fit rows (optionally + Claude-labelled unlabelled
photos), evaluated on dev error rows (non-NONE; 200 rows, all with a gold root annotation).
Views differ only in which text accompanies question/topic."""
import sys, json, re, collections, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load, boot_ci
from k12.core import row_scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

tr, tax, classes, folds, dev, fit = load()
sec = dict(zip(tax.id, tax.section))
R = lambda n: {r['trace_id']: r for r in map(json.loads, open(HERE / f'out/{n}.jsonl'))}
gold, mdev = R('gold_train'), R('machine_dev')
UNLC = R('claude_unl') if (HERE / 'out/claude_unl.jsonl').exists() else {}
txm = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))
from symutil import first_of, root_lines, map_span


def line_map(tx):
    d = {}
    for i, l in enumerate([l for l in str(tx).splitlines() if l.strip()]):
        m = re.match(r'\s*L(\d+)\s*\|\s*(.*)', l)
        d[int(m.group(1)) if m else i + 1] = m.group(2) if m else l
    return d


def located(rec, tx, mode='core_cont_else_last'):
    L = rec['lines']
    if not L:
        return ''
    ln = first_of(L, {'arith', 'eq', 'cont'}) if mode == 'core_cont_else_last' else first_of(L, {'arith', 'eq'})
    if ln is None:
        ln = L[-1]['line']
    return line_map(tx).get(ln, '')


def sym_tokens(rec):
    L = rec['lines']
    if not L:
        return ' symNOPARSE'
    toks = []
    for T in ('arith', 'eq', 'cont', 'bind'):
        ln = first_of(L, {T})
        if ln is not None:
            toks.append(f'sym{T}')
            x = next(x for x in L if x['line'] == ln)
            toks += [f'symop{o}' for o in x['ops']]
            pos = (ln - L[0]['line']) / max(1, L[-1]['line'] - L[0]['line'])
            toks.append('sympos' + ('early' if pos < .34 else 'mid' if pos < .67 else 'late'))
    if not toks:
        toks.append('symclean')
    return ' ' + ' '.join(toks)


def gold_root_text(r):
    g, span = root_lines(r.misconceptions)
    lm = line_map(r.transcription_steps)
    return span or '', ' '.join(lm.get(i, '') for i in sorted(g)) if g else ''


# ---------------------------------------------------------------- rows
def build_fit():
    rows = []
    for r in fit.itertuples():
        if r.root_code == 'NONE':
            continue
        span, rl = gold_root_text(r)
        rows.append(dict(y=r.root_code, topic=r.topic_name, q=f'{r.topic_name} || {r.question_text}', tx=r.transcription_steps,
                         span=span, rootline=rl, symline=located(gold[r.trace_id], r.transcription_steps),
                         symtok=sym_tokens(gold[r.trace_id]), src='fit'))
    return rows


def build_unl():
    rows = []
    if not UNLC:
        return rows
    for l in open(ROOT / 'data/claude_dataset/labels.jsonl'):
        r = json.loads(l)
        if r['split'] != 'unlabelled' or r['first_code'] in ('NONE', '', None) or r['first_code'] not in classes:
            continue
        if r['trace_id'] not in UNLC:
            continue
        tx = r['transcription']; lm = line_map(tx); rl = r.get('root_line') or ''
        m = re.match(r'\s*L(\d+)(?:\s*-\s*L?(\d+))?\s*(\|.*)?$', rl.strip())
        if m:
            a = int(m.group(1)); b = int(m.group(2) or a)
            rt = ' '.join(lm.get(i, '') for i in range(a, b + 1)) or (m.group(3) or '')[1:]
        else:
            rt = ''
        rec = UNLC[r['trace_id']]
        rows.append(dict(y=r['first_code'], topic=r['topic_name'], q=f"{r['topic_name']} || {r['question_text']}", tx=tx, span=rt, rootline=rt,
                         symline=located(rec, tx), symtok=sym_tokens(rec), src='unl'))
    return rows


def build_dev(machine):
    rows = []
    for r in dev.itertuples():
        if r.root_code == 'NONE':
            continue
        span, rl = gold_root_text(r)
        if machine:
            tx = txm[r.trace_id]; rec = mdev[r.trace_id]
        else:
            tx = r.transcription_steps; rec = gold[r.trace_id]
        rows.append(dict(y=r.root_code, topic=r.topic_name, q=f'{r.topic_name} || {r.question_text}', tx=tx, span=span, rootline=rl,
                         symline=located(rec, tx), symtok=sym_tokens(rec), qid=r.question_id))
    return rows


VIEWS = {
    'Q_only': [('q',)],
    'Q+transcript': [('q',), ('tx',)],
    'Q+gold_span': [('q',), ('span',)],
    'Q+gold_rootline': [('q',), ('rootline',)],
    'Q+transcript+gold_rootline': [('q',), ('tx',), ('rootline',)],
    'Q+sym_line': [('q',), ('symline',)],
    'Q+transcript+sym_line': [('q',), ('tx',), ('symline',)],
    'Q+transcript+sym_line+symtok': [('q',), ('tx',), ('symline', 'symtok')],
    'Q+transcript+symtok': [('q',), ('tx', 'symtok')],
    # question-blind views: how much code signal does the text itself carry?
    'transcript_only': [('tx',)],
    'gold_span_only': [('span',)],
    'gold_rootline_only': [('rootline',)],
    'sym_line_only': [('symline',)],
    'topic+gold_rootline': [('topic',), ('rootline',)],
    'topic+transcript': [('topic',), ('tx',)],
    'topic+sym_line': [('topic',), ('symline',)],
}


def featurise(train, tests, blocks):
    Xs_tr, Xs_te = [], [[] for _ in tests]
    for fields in blocks:
        get = lambda d: ' '.join(str(d[f]) for f in fields)
        v = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=150000)
        Xs_tr.append(v.fit_transform([get(d) for d in train]))
        for k, t in enumerate(tests):
            Xs_te[k].append(v.transform([get(d) for d in t]))
    return hstack(Xs_tr).tocsr(), [hstack(x).tocsr() for x in Xs_te]


def run_view(train, tests, blocks, C):
    Xtr, Xtes = featurise(train, tests, blocks)
    ytr = [d['y'] for d in train]
    clf = LogisticRegression(C=C, max_iter=300)
    clf.fit(Xtr, ytr)
    outs = []
    for X in Xtes:
        P = clf.predict_proba(X)
        outs.append((P, list(clf.classes_)))
    return outs


def scores(P, cls, rows):
    top = np.asarray(cls)[np.argsort(-P, 1)[:, :3]]
    y = [d['y'] for d in rows]
    pts = row_scores(y, top)
    secacc = np.mean([sec.get(t[0]) == sec.get(yy) for t, yy in zip(top, y)])
    return pts, float(secacc), top


# seen / unseen categories
fit_pairs = set(zip(fit.question_id, fit.root_code)); fit_q = set(fit.question_id)
dev_err = dev[dev.root_code != 'NONE']
cat = np.array(['seen_pair' if (q, c) in fit_pairs else ('unseen_pair' if q in fit_q else 'unseen_q')
                for q, c in zip(dev_err.question_id, dev_err.root_code)])
qg = dev_err.question_id.to_numpy()

if __name__ == '__main__':
    fitrows = build_fit(); unlrows = build_unl()
    dev_g, dev_m = build_dev(False), build_dev(True)
    print('fit', len(fitrows), 'unl', len(unlrows), 'dev', len(dev_g), {k: int((cat == k).sum()) for k in set(cat)}, flush=True)
    res = dict(n_fit=len(fitrows), n_unl=len(unlrows), n_dev_err=len(dev_g), categories={k: int((cat == k).sum()) for k in set(cat)})
    pts_store = {}
    which = sys.argv[1] if len(sys.argv) > 1 else 'fit_only'
    regimes = [('fit_only', fitrows, 3.0)] if which == 'fit_only' else [('fit+unl', fitrows + unlrows, 3.0)]
    SKIP_BIG = {'Q+gold_span', 'gold_span_only', 'Q+transcript+sym_line', 'Q+transcript+symtok'}
    for reg, train, C in regimes:
        views = [v for v in VIEWS if not (reg != 'fit_only' and v in SKIP_BIG)]
        for name in views:
            blocks = VIEWS[name]
            t0 = time.time()
            (Pg, cls), (Pm, _) = run_view(train, [dev_g, dev_m], blocks, C)
            for tname, P, rows in (('gold_tx', Pg, dev_g), ('machine_tx', Pm, dev_m)):
                pts, sa, _ = scores(P, cls, rows)
                key = f'{reg}|{name}|{tname}'
                pts_store[key] = pts
                res[key] = dict(map3_err_rows=round(float(pts.mean()), 4), top1_section_acc=round(sa, 4),
                                **{f'map3_{k}': round(float(pts[cat == k].mean()), 4) for k in ('seen_pair', 'unseen_pair', 'unseen_q')})
            print(reg, name, res[f'{reg}|{name}|gold_tx']['map3_err_rows'], res[f'{reg}|{name}|machine_tx']['map3_err_rows'], f'{time.time()-t0:.0f}s', flush=True)
        # paired, question-grouped bootstrap CIs vs Q+transcript
        for name in views:
            for tname in ('gold_tx', 'machine_tx'):
                ref = pts_store[f'{reg}|Q+transcript|{tname}']
                res[f'{reg}|{name}|{tname}']['delta_vs_Q+transcript_ci'] = boot_ci(pts_store[f'{reg}|{name}|{tname}'], ref, qg)
    json.dump(res, open(HERE / f'value_oracle_{which}.json', 'w'), indent=1)
    np.savez(HERE / f'value_oracle_pts_{which}.npz', **{k.replace('|', '__'): v for k, v in pts_store.items()})
