"""Step 1 (DEEP_PLAN): corrected candidate-student data, v5b. Same row format as scripts/build_candidate_dataset.py
(whose helpers it imports; data/v5 is not touched), writing runs/deep/stageB/data/cand_{train,dev,test}.jsonl.

Changes over v5:
(a) hygiene: the gold verdict phrase "[student answered ... correctly]" is stripped from every transcript; code IDs and
    "fits X best" sentences are removed from WHY targets (cap 70 words, SPEC); gold rows pass their annotated non-root
    codes as distractors.
(b) confusions: distractors come from each truth's confusion neighbours (Claude co-ranking + 3x gold co-annotation, fit rows
    only; top 8, at most 3 generic per row) instead of same-section siblings; ~1,200 short contrast rows over the sibling
    clusters and the gold generic/specific convention; the truth's same-question example is masked at the smallest rate p
    that brings the visible-example likelihood ratio into the dev band.
(c) one weight: label shift^s x (pair count)^-pp x stratified root-position weight x K if the truth is unseen for the
    question (LOO) x 2 for gold rows on questions the Claude pool never covers; Claude NONE mass reset to the gold share.
    (K, pp, s) is the recipe closest to the plan's (2, .5, .5) that passes the LR / ESS / ANS-01 gates (scan in build_meta).
(d) optional (--smart-fill, off): low-confidence dev/test rows fill slots 9-10 with the specific neighbour of a generic
    top-1 and the topic's frequent codes unseen for the question. It lowered dev recall@10 in gate_v5b, so it is not used.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_candidate_dataset import (DEV_BLEND, FIELDS, LETTERS, TEST_BLEND, Exemplars, blend, h, line_text, marginal,
                                     missing, norm_tx, render, root_line, sinkhorn, tail, why)
from k12.core import label_space, make_folds, read_csv

GENERIC = {'CAL', 'WOP', 'ANS', 'OTH', 'NOT', 'FRM', 'VIS', 'CHK', 'UNT', 'REL'}
CLUSTERS = [('ALG-04', 'ALG-03', 'ALG-06'), ('WOP-01', 'WOP-02', 'ANS-02'), ('MEA-04', 'FRM-01'), ('CAL-01', 'INT-04'),
            ('OTH-00', 'VIS-01')]
VERDICT = re.compile(r'\[[^\]]*student[^\]]*answer[^\]]*correct[^\]]*\]', re.I)
FIT = re.compile(r'\b(match|matches|matching|fits?|fitting|best|closest|code|category|covers?|classif\w*|label\w*)\b', re.I)
OTHER = re.compile(r'\b(alternatives?|could|possible|also|another|other|rather|instead|not|or)\b', re.I)
CONVENTION = {True: 'The other, more general slip in the working follows from this specific error, so the specific error is the root.',
              False: 'Here the general error comes first and the specific slip follows from it, so the general error is the root.'}


def fam(c):
    return 'NONE' if c == 'NONE' else c.split('-')[0].upper()


def is_generic(c):
    return c != 'NONE' and fam(c) in GENERIC


def strip_verdict(text):
    lines = [VERDICT.sub('', x).rstrip() for x in str(text).splitlines()]
    return '\n'.join(x for x, raw in zip(lines, str(text).splitlines()) if x.strip() or not VERDICT.search(raw))


def make_cleaner(ids):
    code = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(ids, key=len, reverse=True))) + r')\b')
    paren = re.compile(r'\s*\((?:[^()]*\b(?:' + '|'.join(map(re.escape, ids)) + r')\b[^()]*)\)')
    colon = re.compile(r'^.*?\b(?:' + '|'.join(map(re.escape, ids)) + r')\b\s*[:—-]\s*(.{12,})$')

    def clean(text):
        """Drop code IDs: parenthetical IDs go; 'matches X: <meaning>' keeps the meaning; other ID sentences are dropped."""
        out = []
        for s in re.split(r'(?<=[.!?])\s+', ' '.join(str(text).split())):
            s = paren.sub('', s)
            if not code.search(s):
                out.append(s); continue
            m = colon.match(s)
            if m and not code.search(m.group(1)):
                out.append(m.group(1)[0].upper() + m.group(1)[1:])
            elif not FIT.search(s):
                out.append(code.sub('another error' if OTHER.search(s) else 'this error', s))
        return ' '.join(out).strip()
    return clean, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--claude', default='runs/claude/unl_all_v4.jsonl')
    ap.add_argument('--explain', default='runs/claude/train_explain_opus.jsonl')
    ap.add_argument('--out', default='runs/deep/stageB/data')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--per-code-cap', type=int, default=200)
    ap.add_argument('--none-cap', type=int, default=2000)
    ap.add_argument('--gold-repeats', type=int, default=2)
    ap.add_argument('--max-chars', type=int, default=1500)
    ap.add_argument('--lam', type=float, default=.5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--why-words', type=int, default=70)
    ap.add_argument('--min-span-chars', type=int, default=6)
    ap.add_argument('--q-codes', type=int, default=3, help='same-question codes per train row')
    ap.add_argument('--neighbours', type=int, default=8)
    ap.add_argument('--max-generic', type=int, default=3)
    ap.add_argument('--contrast-rows', type=int, default=1200)
    ap.add_argument('--contrast-per-code', type=int, default=150)
    ap.add_argument('--mask-grid', default='0,.05,.1,.15,.2,.3,.5,.7')
    ap.add_argument('--lr-band', default='.45,.6')
    ap.add_argument('--none-share', type=float, default=.338)
    ap.add_argument('--smart-fill', action=argparse.BooleanOptionalAction, default=False, help='off: lowered dev recall in gate_v5b')
    args = ap.parse_args()
    R = ROOT
    train, test = read_csv(R / 'data/raw/train.csv'), read_csv(R / 'data/raw/test.csv')
    tax_frame = read_csv(R / 'data/raw/taxonomy_v1.csv')
    classes = label_space(tax_frame)
    taxonomy = {r['id']: r for r in tax_frame.to_dict('records')}
    clean_why, code_re = make_cleaner([c for c in classes if c != 'NONE'] + ['NONE'])
    folds = make_folds(train)
    dev, fit = train.loc[folds == 0].reset_index(drop=True), train.loc[folds != 0].reset_index(drop=True)
    dev_ids = set(dev.trace_id)
    unl = read_csv(R / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
    tx_dev = {k: strip_verdict(v) for k, v in json.loads((R / 'runs/pod/tx/tx_dev.json').read_text()).items()}
    tx_test = {k: strip_verdict(v) for k, v in json.loads((R / 'runs/pod/tx/tx_test.json').read_text()).items()}
    stats = Counter()

    qmeta = {}
    for frame in (train, test, unl.reset_index()):
        for r in frame[['question_id', *FIELDS]].to_dict('records'):
            for f in FIELDS:
                if not missing(r[f]):
                    qmeta.setdefault(r['question_id'], {}).setdefault(f, r[f])

    def meta_of(r):
        return {f: qmeta.get(r['question_id'], {}).get(f, r[f]) if missing(r[f]) else r[f] for f in FIELDS}

    held = {(q, norm_tx(t)) for q, t in zip(dev.question_id, dev.transcription_steps)}
    held |= {(q, norm_tx(tx_dev.get(t, ''))) for q, t in zip(dev.question_id, dev.trace_id)}
    held |= {(q, norm_tx(tx_test.get(t, ''))) for q, t in zip(test.question_id, test.trace_id)}
    held = {k for k in held if k[1]}
    claude = []
    for line in open(R / args.claude):
        r = json.loads(line)
        a = r.get('answer') or {}
        if a.get('first_code') in classes and str(a.get('transcription', '')).strip() and r['trace_id'] in unl.index:
            q = unl.at[r['trace_id'], 'question_id']
            if (q, norm_tx(a['transcription'])) in held:
                stats['claude_dropped_dev_test_duplicates'] += 1; continue
            claude.append(dict(trace_id=r['trace_id'], question_id=q, topic=unl.at[r['trace_id'], 'topic_name'], **a))
    assert not {c['trace_id'] for c in claude} & set(train.trace_id)
    claude_spans = [(c['trace_id'], c['question_id'], c['first_code'],
                     line_text(c['transcription'], root_line(c['root_line'])) if c['first_code'] != 'NONE' else '') for c in claude]

    def misc(r):
        return json.loads(r['misconceptions']) if r['misconceptions'] else []

    def gold_spans(frame):
        out = []
        for r in frame.to_dict('records'):
            m = [e for e in misc(r) if e.get('root')]
            if r['root_code'] != 'NONE' and m:
                out.append((r['trace_id'], r['question_id'], r['root_code'], str(m[0].get('span', '')).strip()))
        return out

    ex_fit = Exemplars(gold_spans(fit), claude_spans, args.seed, args.min_span_chars)
    ex_all = Exemplars(gold_spans(train), claude_spans, args.seed, args.min_span_chars)
    explain = {r['trace_id']: r['answer'] for r in map(json.loads, open(R / args.explain))}

    # Confusion neighbours: Claude co-ranking (all pool rows) + 3x gold co-annotation on fit rows. No dev labels.
    co = defaultdict(Counter)
    for c in claude:
        cs = [x for x in (c.get('first_code'), c.get('second_code'), c.get('third_code')) if x in classes and x != 'NONE']
        for a in cs:
            for b in cs:
                if a != b:
                    co[a][b] += 1
    for r in fit.to_dict('records'):
        cats = [e.get('category') for e in misc(r) if e.get('category') in classes]
        for a in cats:
            for b in cats:
                if a != b:
                    co[a][b] += 3
    neighbours = {c: [x for x, _ in co[c].most_common(args.neighbours)] for c in classes}

    qcount, topic_count, qtopic = {}, defaultdict(Counter), {}
    for q, c, t in (list(zip(fit.question_id, fit.root_code, fit.topic_name)) +
                    [(c['question_id'], c['first_code'], c['topic']) for c in claude]):
        qcount.setdefault(q, Counter())[c] += 1
        if c != 'NONE':
            topic_count[t][c] += 1
        qtopic[q] = t

    tids = test.trace_id.tolist()
    PT = sinkhorn(blend([(R / p, w) for p, w in TEST_BLEND], tids, classes), marginal(train.root_code, classes), args.lam, 50)
    cooc = defaultdict(Counter)  # label-free test top-10 co-listing, as in v5 (fallback distractors after neighbours)
    for q in PT:
        top = [classes[j] for j in np.argsort(-q, kind='stable')[:args.k] if classes[j] != 'NONE']
        for a in top:
            for b in top:
                if a != b:
                    cooc[a][b] += 1

    def train_candidates(truth, extra, q, own_code, rng, k):
        cands = []
        add = lambda c: c in classes and c not in cands and len(cands) < k and cands.append(c)
        for c in [truth, 'NONE', *extra]:
            add(c)
        cnt = Counter(qcount.get(q, Counter()))
        cnt[own_code] -= 1
        for c, n in [x for x in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if x[1] > 0 and x[0] not in cands][:args.q_codes]:
            add(c)
        nb = [c for c in neighbours.get(truth, []) if c not in cands]
        generic = sum(is_generic(c) for c in cands if c != truth)
        for c in (rng.permutation(nb) if nb else []):
            if is_generic(c):
                if generic >= args.max_generic:
                    continue
                generic += 1
            add(str(c))
        nb = cooc.get(truth)
        if nb:
            keys = sorted(nb)
            pw = np.asarray([nb[c] for c in keys], float)
            for i in rng.choice(len(keys), size=len(keys), replace=False, p=pw / pw.sum()):
                add(keys[i])
        for c in rng.permutation(classes):
            add(str(c))
        return [cands[i] for i in rng.permutation(len(cands))]

    def target(line, reason, letter, convention=None):
        text = why(clean_why(reason), args.why_words - (20 if convention is not None else 0))
        if convention is not None:
            text = f'{text} {CONVENTION[convention]}'.strip()
        return f'ROOT_LINE: {line}\nWHY: {text}\nANSWER: {letter}'

    def row_for(kind, src, truth, codes, soft_codes, meta, tx, q, own, line, reason, base_w, convention=None):
        letter = dict(zip(codes, LETTERS))
        soft = {letter[x]: w for x, w in soft_codes if x in letter}
        z = sum(soft.values())
        return dict(truth=truth, meta=meta, tx=tx, q=q, exs=examples_for(codes, q, own, ex_fit), line=line,
                    row=dict(trace_id=own, source=src, kind=kind, phase='a' if src == 'claude' else 'b',
                             target=target(line, reason, letter[truth], convention), letters=list(LETTERS[:len(codes)]),
                             codes=codes, soft={k: v / z for k, v in soft.items()}, weight=base_w))

    def examples_for(codes, q, own, bank):
        out = {}
        for c in codes:
            if c != 'NONE':
                src, span = bank(q, c, own)
                if span:
                    out[c] = (src, span)
        return out

    pending = []
    rng = np.random.default_rng(args.seed)
    by_code = defaultdict(list)
    for c in claude:
        by_code[c['first_code']].append(c)
    picked = []
    for code in sorted(by_code):
        group = by_code[code]
        idx = rng.permutation(len(group))[:args.none_cap if code == 'NONE' else args.per_code_cap]
        picked += [group[i] for i in sorted(idx)]
    for c in picked:
        r = np.random.default_rng([args.seed, h(c['trace_id']), 0])
        codes = train_candidates(c['first_code'], [c['second_code'], c['third_code']], c['question_id'], c['first_code'], r, args.k)
        line = '-' if c['first_code'] == 'NONE' else root_line(c['root_line'])
        pending.append(row_for('main', 'claude', c['first_code'], codes,
                               list(zip([c['first_code'], c['second_code'], c['third_code']], (.7, .2, .1))),
                               meta_of(unl.loc[c['trace_id']]), tail(c['transcription'], args.max_chars), c['question_id'],
                               c['trace_id'], line, c['reasoning'], .8))
    gold_info = {}
    for g in fit.to_dict('records'):
        ms = misc(g)
        m = [e for e in ms if e.get('root')]
        others = [e.get('category') for e in ms if not e.get('root') and e.get('category') in classes and e.get('category') != g['root_code']]
        cats = [g['root_code']] + others if g['root_code'] != 'NONE' else []
        both = any(map(is_generic, cats)) and any(c != 'NONE' and not is_generic(c) for c in cats)
        gold_info[g['trace_id']] = dict(
            line=root_line(m[0]['step']) if m and g['root_code'] != 'NONE' else '-', others=list(dict.fromkeys(others)),
            reason=(explain.get(g['trace_id']) or {}).get('reasoning') or (m[0].get('text', '') if m else ''),
            convention=(not is_generic(g['root_code'])) if both else None)
        stats['gold_both_generic_specific_root_specific' if both and not is_generic(g['root_code']) else
              'gold_both_generic_specific_root_generic' if both else 'gold_single_family'] += 1
    for rep in range(args.gold_repeats):
        for g in fit.to_dict('records'):
            gi = gold_info[g['trace_id']]
            r = np.random.default_rng([args.seed, h(g['trace_id']), rep + 1])
            codes = train_candidates(g['root_code'], gi['others'][:3], g['question_id'], g['root_code'], r, args.k)
            pending.append(row_for('main', 'gold', g['root_code'], codes, [(g['root_code'], 1.)], meta_of(g),
                                   tail(strip_verdict(g['transcription_steps']), args.max_chars), g['question_id'], g['trace_id'],
                                   gi['line'], gi['reason'], 1., gi['convention']))

    # Contrast rows: 2-3 candidates, truth vs its sibling-cluster mates / annotated generic-specific partner.
    cluster_of = {c: cl for cl in CLUSTERS for c in cl}
    pool = []
    for x in pending:
        t, src = x['truth'], x['row']['source']
        if x['row']['kind'] != 'main' or t == 'NONE':
            continue
        gi = gold_info.get(x['row']['trace_id']) if src == 'gold' else None
        if gi and gi['convention'] is not None and x['row']['target'].count('ANSWER') == 1:
            mates = [c for c in gi['others'] if is_generic(c) != is_generic(t)][:1] + [c for c in neighbours[t] if c not in gi['others']][:1]
            pool.append((0, x, mates))
        elif t in cluster_of:
            mates = [c for c in cluster_of[t] if c != t][:2]
            if t in ('OTH-00', 'VIS-01'):
                mates = mates[:1] + [c for c in neighbours[t] if c != 'NONE' and not is_generic(c)][:1]
            pool.append((1, x, mates))
    order = sorted(range(len(pool)), key=lambda i: (pool[i][0], h(args.seed, pool[i][1]['row']['trace_id'], i)))
    seen_gold, per_code, contrast = set(), Counter(), []
    for i in order:
        prio, x, mates = pool[i]
        t, tid = x['truth'], x['row']['trace_id']
        if len(contrast) >= args.contrast_rows or (prio == 0 and tid in seen_gold) or per_code[t] >= args.contrast_per_code or not mates:
            continue
        seen_gold.add(tid); per_code[t] += 1
        r = np.random.default_rng([args.seed, h(tid), 99])
        codes = [t, *dict.fromkeys(m for m in mates if m != t and m in classes)][:3]
        codes = [codes[j] for j in r.permutation(len(codes))]
        src = x['row']['source']
        if src == 'claude':
            c = next(c for c in picked if c['trace_id'] == tid)
            soft = list(zip([c['first_code'], c['second_code'], c['third_code']], (.7, .2, .1)))
            reason, conv = c['reasoning'], None
        else:
            gi = gold_info[tid]; soft, reason, conv = [(t, 1.)], gi['reason'], gi['convention']
        y = row_for('contrast', src, t, codes, soft, x['meta'], x['tx'], x['q'], tid, x['line'], reason, .5, conv)
        contrast.append(y)
    pending += contrast
    stats['contrast_rows'] = len(contrast)
    stats['contrast_by_truth'] = dict(sorted(Counter(x['truth'] for x in contrast).items()))

    # Weights (c).
    pi_gold = marginal(fit.root_code, classes)
    ci = {c: i for i, c in enumerate(classes)}
    claude_main = [x for x in pending if x['row']['source'] == 'claude' and x['row']['kind'] == 'main']
    pi_pool = marginal([x['truth'] for x in claude_main], classes)
    pool_questions = set(unl.question_id)
    nlines = lambda tx: max(1, sum(bool(re.match(r'^\s*L\d+', s)) for s in tx.splitlines()))

    def stratum(x):
        n = nlines(x['tx'])
        return (0 if n <= 2 else 1 if n <= 4 else 2 if n <= 7 else 3, fam(x['truth']) == 'ANS')

    def posbin(x):
        m = re.match(r'L(\d+)', x['line'])
        return None if not m else min(2, int(3 * (int(m.group(1)) - 1) / nlines(x['tx'])))
    gold_pos, pool_pos = defaultdict(Counter), defaultdict(Counter)
    for g in fit.to_dict('records'):
        gi = gold_info[g['trace_id']]
        x = dict(tx=strip_verdict(g['transcription_steps']), truth=g['root_code'], line=gi['line'])
        if g['root_code'] != 'NONE' and posbin(x) is not None:
            gold_pos[stratum(x)][posbin(x)] += 1
    for x in claude_main:
        if x['truth'] != 'NONE' and posbin(x) is not None:
            pool_pos[stratum(x)][posbin(x)] += 1

    def pos_weight(x):
        b = posbin(x)
        if x['row']['source'] != 'claude' or x['truth'] == 'NONE' or b is None:
            return 1.
        s = stratum(x)
        g = (gold_pos[s][b] + 1) / (sum(gold_pos[s].values()) + 3)
        p = (pool_pos[s][b] + 1) / (sum(pool_pos[s].values()) + 3)
        return float(np.clip(g / p, .5, 2.))

    for x in pending:
        cnt = Counter(qcount.get(x['q'], Counter())); cnt[x['truth']] -= 1
        x['pair'] = max(cnt.get(x['truth'], 0), 0); x['unseen'] = x['truth'] != 'NONE' and x['pair'] == 0
        x['seen_d'] = [cnt.get(c, 0) > 0 for c in x['row']['codes'] if c not in (x['truth'], 'NONE')]
        x['pos'] = pos_weight(x)
    cl = [x for x in pending if x['row']['source'] == 'claude']
    before = sum(x['row']['weight'] for x in cl)

    def weigh(K, pp, shift):
        w = []
        for x in pending:
            row, t = x['row'], x['truth']
            v = row['weight']
            if row['source'] == 'claude':
                v *= (pi_gold[ci[t]] / pi_pool[ci[t]]) ** shift
            if t != 'NONE':
                v *= (x['pair'] + 1) ** -pp * x['pos'] * (K if x['unseen'] else 1.)
            if row['source'] == 'gold' and x['q'] not in pool_questions:
                v *= 2.
            w.append(v)
        w = np.array(w); isc = np.array([x['row']['source'] == 'claude' for x in pending])
        isn = np.array([x['truth'] == 'NONE' for x in pending])
        cap = float(np.quantile(w[isc], .99)); w[isc] = np.minimum(w[isc], cap)
        mN, mE = w[isc & isn].sum(), w[isc & ~isn].sum()
        w[isc & isn] *= args.none_share / mN * before; w[isc & ~isn] *= (1 - args.none_share) / mE * before
        return w

    def lr(w, truth_flag, distractor_flags):
        e = [i for i, x in enumerate(pending) if x['truth'] != 'NONE' and distractor_flags(x)]
        pt = np.average([truth_flag(pending[i]) for i in e], weights=w[e])
        pd_ = np.average([np.mean(distractor_flags(pending[i])) for i in e], weights=w[e])
        return float((1 - pt) / (1 - pd_))
    loo_lr = lambda w: lr(w, lambda x: not x['unseen'], lambda x: x['seen_d'])
    v5_rows = [json.loads(l) for l in open(R / 'data/v5/cand_train.jsonl')]
    v5_truth = lambda r: r['codes'][r['letters'].index(r['target'].rsplit('ANSWER:', 1)[1].strip())]
    ans_ref = sum(r['weight'] for r in v5_rows if v5_truth(r) == 'ANS-01') / sum(r['weight'] for r in v5_rows)
    lo, hi = map(float, args.lr_band.split(','))
    scan = []
    for K in (1., 1.5, 2.):
        for pp in (0., .25, .5):
            for shift in (.5, .33):
                w = weigh(K, pp, shift)
                ans = w[[x['truth'] == 'ANS-01' for x in pending]].sum() / w.sum()
                scan.append(dict(K=K, pair_power=pp, shift_power=shift, loo_lr=round(loo_lr(w), 4),
                                 ess=int(w.sum() ** 2 / (w ** 2).sum()), ans01_ratio=round(float(ans / ans_ref), 3)))
    ok = [s for s in scan if lo <= s['loo_lr'] <= hi and s['ess'] >= 4000 and abs(s['ans01_ratio'] - 1) <= .2]
    # Prefer the plan's recipe (K=2, 1/sqrt pair, shift^.5) when it passes, else the passing recipe closest to it.
    plan = lambda s: (abs(s['K'] - 2) + abs(s['pair_power'] - .5) * 2 + abs(s['shift_power'] - .5) * 3)
    chosen = min(ok or scan, key=lambda s: (not ok and abs(s['loo_lr'] - (lo + hi) / 2), plan(s)))
    stats['weight_scan'] = scan; stats['weight_chosen'] = chosen; stats['weight_recipe_passed'] = bool(ok)
    w = weigh(chosen['K'], chosen['pair_power'], chosen['shift_power'])
    for x, v in zip(pending, w):
        x['w'] = float(v)
    cap = float(w.max())

    # Leave-pair-out masking of the truth's example: smallest p whose weighted visible-example LR lands in the band.
    def masked(x, p):
        return np.random.default_rng([args.seed, h(x['row']['trace_id']), x['row']['kind'] == 'contrast', 7]).random() < p
    dflag = lambda x: [c in x['exs'] for c in x['row']['codes'] if c not in (x['truth'], 'NONE')]
    grid = [float(v) for v in args.mask_grid.split(',')]
    mask_scan = {p: (lr(w, lambda x, p=p: x['truth'] in x['exs'] and not masked(x, p), dflag),) for p in grid}
    p_mask = next((p for p in grid if lo <= mask_scan[p][0] <= hi), min(grid, key=lambda p: abs(mask_scan[p][0] - (lo + hi) / 2)))
    for x in pending:
        if x['truth'] != 'NONE' and x['truth'] in x['exs'] and masked(x, p_mask):
            del x['exs'][x['truth']]
    for x in pending:
        x['row']['weight'] = round(float(x['w']), 5)

    rows = []
    for x in pending:
        rows.append({k: x['row'][k] for k in ('trace_id', 'source', 'phase', 'kind')} |
                    dict(prompt=render(x['meta'], x['tx'], x['row']['codes'], taxonomy, x['exs'])) |
                    {k: x['row'][k] for k in ('target', 'letters', 'codes', 'soft', 'weight')} |
                    dict(question_id=x['q'], unseen=x['unseen']))
        rows[-1]['source_ids'] = sorted({s for s, _ in x['exs'].values()})

    # Dev/test candidates, with the optional smart fill for low-confidence rows.
    def top_k(P, frame, codes_seen):
        tops, n_smart = [], 0
        for p, (q, topic) in zip(P, zip(frame.question_id, frame.topic_name)):
            order = [classes[j] for j in np.argsort(-p, kind='stable')]
            top = order[:args.k]
            if 'NONE' not in top:
                top[-1] = 'NONE'
            if args.smart_fill and p.max() < .5:
                head = [c for c in top[:args.k - 2]]
                extra = []
                if is_generic(top[0]):
                    extra += [c for c in neighbours[top[0]] if c != 'NONE' and not is_generic(c) and c not in head][:1]
                seen = codes_seen.get(q, Counter())
                extra += [c for c, _ in topic_count[topic].most_common() if seen.get(c, 0) == 0 and c not in head + extra]
                tailc = [c for c in top[args.k - 2:] if c == 'NONE']
                fill = (extra + [c for c in top[args.k - 2:] if c not in extra])
                new = head + tailc
                for c in fill:
                    if len(new) < args.k and c not in new:
                        new.append(c)
                n_smart += new != top
                top = head + [c for c in new[len(head):] if c != 'NONE'] + tailc
                top = top[:args.k] if 'NONE' in top[:args.k] else top[:args.k - 1] + ['NONE']
            tops.append(top)
        return tops, n_smart

    def ranked(frame, tops, bank, texts):
        out = []
        for g, top in zip(frame.to_dict('records'), tops):
            exs = examples_for(top, g['question_id'], g['trace_id'], bank)
            tx, meta = tail(texts[g['trace_id']], args.max_chars), meta_of(g)
            orders = [top, top[::-1]]
            out.append(dict(trace_id=g['trace_id'], prompts=[render(meta, tx, o, taxonomy, exs) for o in orders],
                            codes_per_prompt=orders, letters=list(LETTERS[:len(top)]), source_ids=sorted({s for s, _ in exs.values()})))
        return out

    dids = dev.trace_id.tolist()
    PD = sinkhorn(blend([(R / p, w) for p, w in DEV_BLEND], dids, classes), marginal(fit.root_code, classes), args.lam, 30)
    fit_counts = {}
    for q, c in list(zip(fit.question_id, fit.root_code)) + [(c['question_id'], c['first_code']) for c in claude]:
        fit_counts.setdefault(q, Counter())[c] += 1
    all_counts = {}
    for q, c in list(zip(train.question_id, train.root_code)) + [(c['question_id'], c['first_code']) for c in claude]:
        all_counts.setdefault(q, Counter())[c] += 1
    dev_tops, stats['dev_smart_filled'] = top_k(PD, dev, fit_counts)
    test_tops, stats['test_smart_filled'] = top_k(PT, test, all_counts)
    dev_rows = ranked(dev, dev_tops, ex_fit, tx_dev)
    for d, t in zip(dev_rows, dev.root_code):
        d['truth'] = t
    test_rows = ranked(test, test_tops, ex_all, tx_test)

    out = R / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (('cand_train', rows), ('cand_dev', dev_rows), ('cand_test', test_rows)):
        with open(out / f'{name}.jsonl', 'w') as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')
    leak = dict(dev_ids_in_train_rows=len(dev_ids & {x['trace_id'] for x in rows}),
                dev_ids_in_train_exemplar_sources=len(dev_ids & {s for x in rows for s in x['source_ids']}),
                dev_ids_in_dev_exemplar_sources=len(dev_ids & {s for x in dev_rows for s in x['source_ids']}))
    assert not any(leak.values()), leak
    meta = dict(settings=vars(args), stats=dict(stats), mask_scan={str(p): [round(v, 4) for v in s] for p, s in mask_scan.items()},
                p_mask=p_mask, leakage=leak, claude_weight_cap=round(cap, 4),
                rows=dict(Counter(f"{x['source']}/{x['kind']}" for x in rows)), train=len(rows), dev=len(dev_rows), test=len(test_rows),
                verdict_left=dict(train=sum(bool(VERDICT.search(x['prompt'])) for x in rows),
                                  dev=sum(bool(VERDICT.search(p)) for x in dev_rows for p in x['prompts']),
                                  test=sum(bool(VERDICT.search(p)) for x in test_rows for p in x['prompts'])),
                why_with_code_id=sum(bool(code_re.search(x['target'].split('WHY:', 1)[1].rsplit('ANSWER:', 1)[0])) for x in rows))
    (out / 'build_meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
