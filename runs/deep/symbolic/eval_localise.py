"""Evaluate symbolic localisation + final-answer flag against gold root steps.
Gold transcripts: all 1,500 train rows (the checker has no trained parameters) and dev fold 0.
Machine transcripts: dev fold 0 (gold root line mapped onto the machine transcript by span matching)."""
import sys, json, re, difflib, collections
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'runs/arch/label-structure'), str(HERE)]
from common import load, blend, boot_ci
from checker import check as crude_check
from sklearn.metrics import roc_auc_score

tr, tax, classes, folds, dev, fit = load()
R = lambda n: {r['trace_id']: r for r in map(json.loads, open(HERE / f'out/{n}.jsonl'))}
gold, mdev, cdev = R('gold_train'), R('machine_dev'), R('claude_dev')
txm = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))
cl = {r['trace_id']: r['answer']['transcription'] for r in map(json.loads, open(ROOT / 'runs/claude/dev_opus.jsonl')) if r.get('answer')}
unl_path = HERE / 'out/claude_unl.jsonl'
UNL = R('claude_unl') if unl_path.exists() else {}
meta_unl = {}
if UNL:
    for l in open(ROOT / 'data/claude_dataset/labels.jsonl'):
        r = json.loads(l)
        if r['split'] == 'unlabelled':
            meta_unl[r['trace_id']] = (r['question_id'], r['first_code'])


def root_lines(m):
    for a in (json.loads(m) if m else []):
        if a.get('root'):
            nums = [int(x) for x in re.findall(r'L(\d+)', a['step'])]
            if len(nums) == 2:
                nums = list(range(nums[0], nums[1] + 1))
            return set(nums), a['span']
    return None, None


def body(l):
    return re.sub(r'^\s*L\d+\s*\|\s*', '', l)


def norm(s):
    return re.sub(r'\s+', '', str(s).replace('−', '-').replace('×', 'x').lower())


def map_span(span, transcript):
    """Map a gold span onto the line numbers of another transcript (best fuzzy line)."""
    lines = [l for l in str(transcript).splitlines() if l.strip()]
    sp_ = norm(span); best, bi = 0, None
    for i, l in enumerate(lines):
        b = norm(body(l))
        if not b or not sp_:
            continue
        m = difflib.SequenceMatcher(None, sp_, b, autojunk=False)
        lm = m.find_longest_match(0, len(sp_), 0, len(b)).size / len(sp_)
        sc = max(lm, m.ratio())
        if sc > best:
            best, bi = sc, i
    if bi is None:
        return None, best
    mm = re.match(r'\s*L(\d+)', lines[bi])
    return (int(mm.group(1)) if mm else bi + 1), best


# ---------------------------------------------------------------- reference-path values
def ref_values(exclude_ids, allowed_ids):
    """question_id -> list of value-sets from clean (NONE) transcripts."""
    ref = collections.defaultdict(list)
    for r in tr.itertuples():
        if r.root_code == 'NONE' and r.trace_id in allowed_ids and r.trace_id not in exclude_ids:
            ref[r.question_id].append(set(v for x in gold[r.trace_id]['lines'] for v in x['values']))
    for tid, rec in UNL.items():
        q, c = meta_unl.get(tid, (None, None))
        if c == 'NONE':
            ref[q].append(set(v for x in rec['lines'] for v in x['values']))
    return ref


def qvals(text):
    vs = set()
    for t in re.findall(r'\d+(?:[.,]\d+)*', str(text)):
        for u in {t.replace('.', '').replace(',', '.'), t.replace(',', '.'), t.replace(',', '')}:
            try:
                vs.add(round(float(u), 6))
            except ValueError:
                pass
    return vs


def in_set(v, S):
    return any(abs(v - s) <= 1e-3 * max(1, abs(s)) for s in S)


def first_divergent(lines, refsets, qv, min_refs=2, min_support=1):
    if len(refsets) < min_refs:
        return None
    pool = collections.Counter()
    for S in refsets:
        for v in S:
            pool[v] += 1
    okvals = [v for v, c in pool.items() if c >= min_support] + list(qv)
    for x in lines:
        for v in x['values']:
            if abs(v) <= 1 or float(v).is_integer() and abs(v) <= 10:
                continue          # tiny integers / 0/1 are uninformative
            if not in_set(v, okvals):
                return x['line']
    return None


def typ(reason):
    if reason.startswith('cont:'): return 'cont'
    if 'contradicts' in reason: return 'bind'
    if re.match(r'^\w+: \[', reason): return 'eq'
    return 'arith'


def first_of(L, types):
    return next((x['line'] for x in L if any(typ(z) in types for z in x['reasons'])), None)


def predictors(rec, refsets, qv):
    L = rec['lines']
    nums = [x['line'] for x in L]
    last = nums[-1] if nums else None
    arith = first_of(L, {'arith'})
    core = first_of(L, {'arith', 'eq'})
    allf = first_of(L, {'arith', 'eq', 'cont'})
    div = first_of_div = first_divergent(L, refsets, qv) if refsets is not None else None
    cd = min([z for z in (core, div) if z is not None], default=None)
    return dict(arith=arith, core=core, core_cont=allf, diverge=div, core_or_diverge=cd,
                core_else_last=core if core is not None else last,
                core_cont_else_last=allf if allf is not None else last,
                core_or_diverge_else_last=cd if cd is not None else last,
                diverge_else_core_else_last=div if div is not None else (core if core is not None else last),
                last=last, first=nums[0] if nums else None)


def score_loc(rows):
    """rows: list of (gold_set, preds dict, n_lines)."""
    out = {}
    for k in rows[0][1]:
        cov = hit = hit1 = 0; n = len(rows); rel = []
        for g, p, nl in rows:
            v = p[k]
            if v is None:
                continue
            cov += 1; hit += v in g; hit1 += any(abs(v - z) <= 1 for z in g)
            rel.append(v - min(g))
        out[k] = dict(n=n, coverage=round(cov / n, 3), hit_given_flag=round(hit / max(cov, 1), 3),
                      hit_pm1_given_flag=round(hit1 / max(cov, 1), 3), hit_overall=round(hit / n, 3),
                      flag_before_root=round(np.mean([r < 0 for r in rel]), 3) if rel else None,
                      flag_after_root=round(np.mean([r > 0 for r in rel]), 3) if rel else None)
    # uniform-random line baseline
    out['random_line_expected'] = round(float(np.mean([len(g) / max(nl, 1) for g, _, nl in rows])), 3)
    return out


res = {}
dev_ids = set(dev.trace_id); fit_ids = set(fit.trace_id); all_ids = set(tr.trace_id)
# ---- gold transcripts, all train rows (leave-one-out reference)
rows = []; per_topic = collections.defaultdict(list)
for r in tr.itertuples():
    g, span = root_lines(r.misconceptions)
    if not g:
        continue
    refs = ref_values({r.trace_id}, all_ids)[r.question_id] if False else None
    rows.append((g, r))
REFALL = ref_values(set(), all_ids)


def loo(refsets, rec_tid):
    own = set(v for x in gold[rec_tid]['lines'] for v in x['values'])
    out = list(refsets);
    if own in out:
        out.remove(own)
    return out


rows_gold = []
for g, r in rows:
    rec = gold[r.trace_id]
    refs = loo(REFALL.get(r.question_id, []), r.trace_id)
    p = predictors(rec, refs, qvals(r.question_text) | qvals(r.correct_answer))
    rows_gold.append((g, p, len(rec['lines'])))
    per_topic[r.topic_name].append((g, p, len(rec['lines'])))
res['gold_all_train_err_rows'] = score_loc(rows_gold)
res['gold_dev_err_rows'] = score_loc([x for x, (g, r) in zip(rows_gold, rows) if r.trace_id in dev_ids])
tops = sorted(per_topic, key=lambda t: -len(per_topic[t]))[:12]
res['gold_by_topic_core'] = {t: {k: score_loc(per_topic[t])['core'][k] for k in ('n', 'coverage', 'hit_given_flag')} for t in tops}

# ---- parse coverage
def coverage(recs):
    L = [x for rec in recs for x in rec['lines']]
    rows_any = np.mean([any(x['n_checks'] for x in rec['lines']) for rec in recs])
    return dict(lines=len(L), line_has_parse=round(np.mean([x['n_parsed'] > 0 for x in L]), 3),
                line_has_check=round(np.mean([x['n_checks'] > 0 for x in L]), 3),
                rows_with_any_check=round(float(rows_any), 3),
                rows_flagged_core=round(np.mean([first_of(rec['lines'], {'arith', 'eq'}) is not None for rec in recs]), 3))
res['coverage_gold_train'] = coverage(list(gold.values()))
res['coverage_machine_dev'] = coverage(list(mdev.values()))
res['coverage_claude_dev'] = coverage(list(cdev.values()))
if UNL:
    res['coverage_claude_unl'] = coverage(list(UNL.values()))
# invalid-flag rate by label (does 'some line invalid' separate NONE from error rows?)
for nm, src in [('gold_train', gold), ('machine_dev', mdev)]:
    sub = tr[tr.trace_id.isin(src)]
    fl = np.array([first_of(src[t]['lines'], {'arith', 'eq'}) is not None for t in sub.trace_id])
    ar = np.array([any(x['n_invalid'] for x in src[t]['lines']) for t in sub.trace_id])
    isn = (sub.root_code == 'NONE').to_numpy()
    res[f'invalid_flag_rate_{nm}'] = dict(none_rows=round(fl[isn].mean(), 3), err_rows=round(fl[~isn].mean(), 3),
                                         arith_none_rows=round(ar[isn].mean(), 3), arith_err_rows=round(ar[~isn].mean(), 3))

# ---- machine & Claude transcripts on dev: map gold root to their line numbering
REFFIT = ref_values(dev_ids, fit_ids)
for nm, src, txt in [('machine_dev', mdev, txm), ('claude_dev', cdev, cl)]:
    rr = []; mapped = 0; tot = 0
    for r in dev.itertuples():
        g, span = root_lines(r.misconceptions)
        if not g or r.trace_id not in src:
            continue
        tot += 1
        ln, sc = map_span(span, txt[r.trace_id])
        if ln is None or sc < 0.6:
            continue
        mapped += 1
        rec = src[r.trace_id]
        p = predictors(rec, REFFIT.get(r.question_id, []), qvals(r.question_text) | qvals(r.correct_answer))
        rr.append(({ln}, p, len(rec['lines'])))
    res[f'{nm}_err_rows'] = dict(n_root_rows=tot, mapped=mapped, **score_loc(rr))
# gold on dev with fit-only references (same protocol as machine)
rr = []
for r in dev.itertuples():
    g, span = root_lines(r.misconceptions)
    if not g:
        continue
    rec = gold[r.trace_id]
    rr.append((g, predictors(rec, REFFIT.get(r.question_id, []), qvals(r.question_text) | qvals(r.correct_answer)), len(rec['lines'])))
res['gold_dev_err_rows_fitrefs'] = score_loc(rr)

# ---- NONE detection AUC
ids = dev.trace_id.tolist(); P = blend(ids, classes); ni = classes.index('NONE')
y_none = (dev.root_code == 'NONE').to_numpy().astype(int)
y_fac = dev.final_answer_correct.astype(str).str.lower().eq('true').to_numpy().astype(int)
auc = {}
def f(v):
    return 0.5 if v is None else v
for nm, src, txt in [('gold', gold, dict(zip(tr.trace_id, tr.transcription_steps))), ('machine', mdev, txm), ('claude', cdev, cl)]:
    for k in ('fac', 'fac1', 'facall'):
        s = np.array([f(src[t][k]) for t in ids])
        auc[f'{nm}_{k}_vs_NONE'] = round(roc_auc_score(y_none, s), 4)
        auc[f'{nm}_{k}_vs_finalcorrect'] = round(roc_auc_score(y_fac, s), 4)
    auc[f'{nm}_fac_parse_rate'] = round(np.mean([src[t]['fac'] is not None for t in ids]), 3)
    s = np.array([f(src[t]['fac']) for t in ids]) - 0.5 * np.array([first_of(src[t]['lines'], {'arith', 'eq'}) is not None for t in ids])
    auc[f'{nm}_fac_minus_invalid_vs_NONE'] = round(roc_auc_score(y_none, s), 4)
    cr = np.array([crude_check(txt[t], a, 2)[0] for t, a in zip(ids, dev.correct_answer)])
    auc[f'{nm}_crude_checker_vs_NONE'] = round(roc_auc_score(y_none, cr), 4)
auc['blend_PNONE_vs_NONE'] = round(roc_auc_score(y_none, P[:, ni]), 4)
auc['blend_PNONE_vs_finalcorrect'] = round(roc_auc_score(y_fac, P[:, ni]), 4)
res['none_auc_dev'] = auc
# all-train gold AUC (checker untrained)
yt = (tr.root_code == 'NONE').to_numpy().astype(int)
yf = tr.final_answer_correct.astype(str).str.lower().eq('true').to_numpy().astype(int)
s = np.array([f(gold[t]['fac']) for t in tr.trace_id])
res['none_auc_gold_all_train'] = dict(fac_vs_NONE=round(roc_auc_score(yt, s), 4), fac_vs_finalcorrect=round(roc_auc_score(yf, s), 4),
                                      fac_parse_rate=round(np.mean([gold[t]['fac'] is not None for t in tr.trace_id]), 3),
                                      crude_vs_NONE=round(roc_auc_score(yt, [crude_check(t, a, 2)[0] for t, a in zip(tr.transcription_steps, tr.correct_answer)]), 4))
json.dump(res, open(HERE / 'localisation.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
