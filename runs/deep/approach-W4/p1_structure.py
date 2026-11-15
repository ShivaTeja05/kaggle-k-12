"""W4 structure probe (CPU, numpy/pandas). Root vs secondary annotated codes: how often, where,
whether our blend / Claude pick the secondary, and whether Stage B candidates contain the secondaries."""
import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT/'runs/deep/approach-W4'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'runs/arch/label-structure')]
from common import load, blend, boot_ci
from k12.core import row_scores
from evaluate_claude import claude_scores

tr, tax, classes, folds, dev, fit = load()
LN = lambda s: [int(x) for x in re.findall(r'L(\d+)', str(s or ''))]
def nlines(tx): return sum(1 for l in str(tx).splitlines() if l.strip())

def parse(row):
    try: L = json.loads(row.misconceptions) if isinstance(row.misconceptions, str) else []
    except Exception: L = []
    roots = [m for m in L if m.get('root')]
    rl = min(LN(roots[0].get('step'))) if roots and LN(roots[0].get('step')) else None
    sec = []
    for m in L:
        if m.get('root'): continue
        c = m.get('category')
        if c and c != row.root_code and c not in [s[0] for s in sec]:
            l = LN(m.get('step')); sec.append((c, min(l) if l else None, m.get('severity')))
    return rl, sec

res = {}
# ---- train structure
st = Counter()
for r in tr.itertuples():
    if r.root_code == 'NONE': continue
    rl, sec = parse(r); n = nlines(r.transcription_steps); st['err'] += 1
    if rl is not None:
        st['root_has_line'] += 1; st['lines_after_root>=1'] += int(n > rl); st['lines_after_root>=2'] += int(n > rl + 1)
    if sec:
        st['has_secondary_code'] += 1
        for c, l, sv in sec:
            st['sec_total'] += 1; st['sec_major'] += int(sv == 'major')
            if l is not None and rl is not None:
                st['sec_after_root'] += int(l > rl); st['sec_same_line'] += int(l == rl); st['sec_before_root'] += int(l < rl)
res['train'] = dict(st) | {'share_err_rows_with_secondary_code': round(st['has_secondary_code']/st['err'], 3),
                           'share_secondary_after_root': round(st['sec_after_root']/st['sec_total'], 3),
                           'share_root_not_last_line': round(st['lines_after_root>=1']/st['root_has_line'], 3)}
sec_codes = Counter(c for r in tr.itertuples() if r.root_code != 'NONE' for c, _, _ in parse(r)[1])
res['train_top_secondary_codes'] = sec_codes.most_common(10)

# ---- dev: blend / Claude vs secondary
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy(); ci = {c: i for i, c in enumerate(classes)}
P = blend(ids, classes); C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
rk = lambda M: [[classes[j] for j in np.argsort(-m)[:3]] for m in M]
sB, sC = np.array(row_scores(truth, rk(P))), np.array(row_scores(truth, rk(C)))
groups = dev.question_id.to_numpy()
qcodes = {}
for q, c in zip(fit.question_id, fit.root_code): qcodes.setdefault(q, Counter())[c] += 1
has_sec = np.zeros(len(dev), bool); b_sec = np.zeros(len(dev), bool); c_sec = np.zeros(len(dev), bool)
sec_rank_b = []; err = truth != 'NONE'; newpair = np.zeros(len(dev), bool)
for i, r in enumerate(dev.itertuples()):
    newpair[i] = err[i] and qcodes.get(r.question_id, Counter())[r.root_code] == 0
    if not err[i]: continue
    _, sec = parse(r); S = {c for c, _, _ in sec if c in ci}
    if not S: continue
    has_sec[i] = True
    b_sec[i] = classes[P[i].argmax()] in S; c_sec[i] = classes[C[i].argmax()] in S
    order = list(np.argsort(-P[i])); sec_rank_b.append(min(order.index(ci[c]) for c in S) + 1)
def seg(mask, name):
    d, ci_ = boot_ci(np.where(mask, sC, 0), np.where(mask, sB, 0), groups)
    return {'n': int(mask.sum()), 'blend': round(float(sB[mask].mean()), 3), 'claude': round(float(sC[mask].mean()), 3),
            'gap_points_per_301': round(d, 4), 'gap_ci': [round(x, 4) for x in ci_]}
res['dev'] = {'err_rows': int(err.sum()), 'err_rows_with_secondary_code': int(has_sec.sum()),
              'seg_err_with_secondary': seg(has_sec, 'sec'), 'seg_err_without_secondary': seg(err & ~has_sec, 'nosec'),
              'seg_newpair_with_secondary': seg(newpair & has_sec, 'np_sec'),
              'blend_top1_is_secondary_on_sec_rows': [int(b_sec.sum()), round(float(b_sec[has_sec].mean()), 3)],
              'claude_top1_is_secondary_on_sec_rows': [int(c_sec.sum()), round(float(c_sec[has_sec].mean()), 3)],
              'blend_best_secondary_rank_median': float(np.median(sec_rank_b)),
              'blend_secondary_in_top3_share': round(float(np.mean(np.array(sec_rank_b) <= 3)), 3),
              'blend_top1_secondary_rows_score': round(float(sB[b_sec].mean()), 3) if b_sec.any() else None,
              'claude_on_blend_top1_secondary_rows': round(float(sC[b_sec].mean()), 3) if b_sec.any() else None}

# ---- Stage B candidate coverage of secondaries
trrow = {r.trace_id: r for r in tr.itertuples()}
def cover(path, codes_key):
    n = hit = n_rows = 0
    for l in open(path):
        x = json.loads(l); r = trrow.get(x['trace_id'])
        if r is None or r.root_code == 'NONE': continue
        if codes_key == 'codes' and x.get('source') != 'gold': continue
        S = {c for c, _, _ in parse(r)[1]}
        if not S: continue
        cand = set(x['codes'] if codes_key == 'codes' else x['codes_per_prompt'][0])
        n_rows += 1; n += len(S); hit += len(S & cand)
    return {'rows_with_secondary': n_rows, 'secondary_codes': n, 'in_candidates': hit, 'share': round(hit/max(n, 1), 3)}
res['cand_train_gold_secondary_coverage'] = cover(ROOT/'data/v5/cand_train.jsonl', 'codes')
res['cand_dev_secondary_coverage'] = cover(ROOT/'data/v5/cand_dev.jsonl', 'codes_per_prompt')
# Claude rows: ROOT_LINE '-' on error rows; WHY mentions a line
cl = Counter()
for l in open(ROOT/'data/v5/cand_train.jsonl'):
    x = json.loads(l)
    if x['source'] != 'claude': continue
    ans = x['target'].split('ANSWER: ')[-1].strip(); code = x['codes'][x['letters'].index(ans)]
    if code == 'NONE': continue
    cl['err'] += 1; cl['root_dash'] += int(x['target'].startswith('ROOT_LINE: -'))
res['cand_train_claude_error_rows_root_dash'] = dict(cl)
json.dump(res, open(OUT/'p1_structure.json', 'w'), indent=1); print(json.dumps(res, indent=1))
