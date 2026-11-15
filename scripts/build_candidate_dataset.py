"""Candidate-aware student dataset (runs/arch/student/SPEC.md).

Train rows come from Claude-labelled unlabelled photos (phase a) and gold fit rows (phase b);
dev/test rows list the top-10 codes of the Sinkhorn-balanced blend in two orders. No fold-0
trace_id is ever a train row or an exemplar source for train rows.

Deviations from SPEC.md, each to keep train rows from carrying cues that dev/test rows lack:
- train candidates take at most --max-siblings same-section codes, then codes that co-occur with the
  truth in the label-free test blend top-10 (dev candidate lists are not section-heavy);
- the truth's example is dropped at the rate that makes "has an example" uninformative per source;
- example spans shorter than --min-span-chars are skipped; WHY keeps its concluding sentence and is
  capped at --why-words (60); Claude NONE rows get ROOT_LINE "-";
- Claude rows whose transcript equals a dev or test transcript on the same question are dropped;
- literal "nan" header fields are filled from other rows of the same question.
"""
import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from k12.core import aligned_bundle, label_space, make_folds, normalize, read_csv

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NONE_LINE = 'NONE: No major error - the working is correct or has only minor slips.'
LINE = re.compile(r'^\s*L(\d+)\s*\|\s?(.*)$')
CONCLUSION = re.compile(r'\b(match|matches|root|major error|no error|NONE|minor slips?)\b', re.I)
FIELDS = ('grade', 'topic_name', 'question_text', 'correct_answer')
DEV_BLEND = [('runs/pod/text32/validation_tx_dev_epoch_1.npz', .35), ('runs/pod/tx/text_tx_dev_e3.npz', .30),
             ('runs/pod/dev-distill/validation_epoch_2.npz', .35)]
TEST_BLEND = [('runs/pod/text32-full/test.npz', .35), ('runs/pod/text-full/test.npz', .30),
              ('runs/pod/full-distill/test.npz', .35)]


def h(*parts):
    return int(hashlib.md5(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def clip(text, n):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[:n].rstrip()


def tail(text, n):
    text = str(text).strip()
    return text if len(text) <= n else text[-n:]


def root_line(value):
    m = re.match(r'^\s*L(\d+)', str(value or ''))
    return f'L{m.group(1)}' if m else '-'


def line_text(transcription, line):
    for raw in str(transcription).splitlines():
        m = LINE.match(raw)
        if m and f'L{m.group(1)}' == line:
            return m.group(2).strip()
    return ''


def why(text, words=60):
    """First sentences up to `words`, keeping the last concluding sentence (it names the error) when it fits."""
    sentences = [x for x in re.split(r'(?<=[.!?])\s+', ' '.join(str(text).split())) if x]
    n = lambda ix: len(' '.join(sentences[i] for i in ix).split())
    if not sentences or n(range(len(sentences))) <= words:
        return ' '.join(sentences)
    concl = next((i for i in range(len(sentences) - 1, -1, -1) if CONCLUSION.search(sentences[i])), len(sentences) - 1)
    keep = [concl] if n([concl]) <= words // 2 else []
    for i in range(len(sentences)):
        if i not in keep:
            if n(keep + [i]) > words:
                break
            keep.append(i)
    return ' '.join(sentences[i] for i in sorted(keep)) if keep else ' '.join(sentences[0].split()[:words])


def norm_tx(text):
    return ' '.join(re.sub(r'(?m)^\s*L\d+\s*\|\s?', '', str(text)).lower().split())


def missing(value):
    return str(value).strip().lower() in ('', 'nan', 'none', 'null')


def token_counter(path, python):
    """Token counts with the tokenizers library, or via another interpreter that has it (the venv may not)."""
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(path))
        return lambda texts: [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]
    except ImportError:
        code = ('import sys,json;from tokenizers import Tokenizer;t=Tokenizer.from_file(sys.argv[1]);'
                'print(json.dumps([len(e.ids) for e in t.encode_batch(json.load(sys.stdin),add_special_tokens=False)]))')
        return lambda texts: json.loads(subprocess.run([python, '-c', code, str(path)], input=json.dumps(list(texts)),
                                                       capture_output=True, text=True, check=True).stdout)


def blend(paths, ids, classes):
    lp = sum(w * np.log(aligned_bundle(p, ids, classes)) for p, w in paths)
    return normalize(np.exp(lp - lp.max(1, keepdims=True)))


def sinkhorn(P, target, lam, iters):
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(lam * b))


def marginal(codes, classes):
    ci = {c: i for i, c in enumerate(classes)}
    pi = np.bincount([ci[c] for c in codes], minlength=len(classes)) + .5
    return pi / pi.sum()


class Exemplars:
    """Same-question example span per code; gold spans first, then Claude root lines, never the row itself."""

    def __init__(self, gold, claude, seed, min_chars):
        self.pool = {}
        for frame_rows, kind in ((gold, 0), (claude, 1)):
            for tid, q, c, span in frame_rows:
                if len(' '.join(str(span).split())) >= min_chars:
                    self.pool.setdefault((q, c), []).append((kind, h(seed, tid), tid, span))
        for v in self.pool.values():
            v.sort()

    def __call__(self, q, c, own):
        for _, _, tid, span in self.pool.get((q, c), []):
            if tid != own:
                return tid, span
        return None, ''


def render(meta, transcript, codes, taxonomy, examples):
    lines = []
    for letter, c in zip(LETTERS, codes):
        if c == 'NONE':
            lines.append(f'{letter}. {NONE_LINE}')
            continue
        t = taxonomy[c]
        ex = f' On this question, e.g. "{clip(examples[c][1], 60)}"' if examples.get(c) else ''
        # Full description: the student learns what a code means from this text. Cutting it at 140
        # characters ended 23% of codes mid-sentence; the longest is 328 characters and fits the budget.
        description = ' '.join(str(t["description"]).split())
        if description and description[-1] not in '.!?':
            description += '.'
        lines.append(f'{letter}. {c}: {t["name"]} - {description}{ex}')
    return (f'Grade: {meta["grade"]}\nTopic: {meta["topic_name"]}\nQuestion: {meta["question_text"]}\n'
            f'Answer key: {meta["correct_answer"]}\nStudent working (line by line):\n{transcript}\n\n'
            'Candidate error codes:\n' + '\n'.join(lines) + '\n'
            'Find the root major error the rest of the working follows from (or NONE if the working has no '
            'major error) and choose the matching candidate.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--claude', default='runs/claude/unl_all_v4.jsonl')
    ap.add_argument('--explain', default='runs/claude/train_explain_opus.jsonl')
    ap.add_argument('--out', default='data/v5')
    ap.add_argument('--report', default='reports/cand_dataset.json')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--per-code-cap', type=int, default=200)
    ap.add_argument('--none-cap', type=int, default=2000)
    ap.add_argument('--gold-repeats', type=int, default=2)
    ap.add_argument('--max-chars', type=int, default=1500)
    ap.add_argument('--lam', type=float, default=.5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--why-words', type=int, default=60)
    ap.add_argument('--min-span-chars', type=int, default=6)
    ap.add_argument('--max-siblings', type=int, default=1)
    ap.add_argument('--tokenizer', default=str(Path.home() / '.cache/huggingface/hub/models--trl-internal-testing--'
                                               'tiny-Qwen2ForCausalLM-2.5/snapshots/*/tokenizer.json'))
    ap.add_argument('--tokenizer-python', default='/usr/bin/python3')
    args = ap.parse_args()
    R = Path(args.root)
    train, test = read_csv(R / 'data/raw/train.csv'), read_csv(R / 'data/raw/test.csv')
    tax_frame = read_csv(R / 'data/raw/taxonomy_v1.csv')
    classes = label_space(tax_frame)
    taxonomy = {r['id']: r for r in tax_frame.to_dict('records')}
    section = {}
    for c, r in taxonomy.items():
        section.setdefault(r['section'], []).append(c)
    folds = make_folds(train)
    dev, fit = train.loc[folds == 0].reset_index(drop=True), train.loc[folds != 0].reset_index(drop=True)
    dev_ids = set(dev.trace_id)
    unl = read_csv(R / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
    tx_dev = json.loads((R / 'runs/pod/tx/tx_dev.json').read_text())
    tx_test = json.loads((R / 'runs/pod/tx/tx_test.json').read_text())

    # Question metadata, to fill literal "nan" header fields from other rows of the same question.
    qmeta, fills = {}, Counter()
    for frame in (train, test, unl.reset_index()):
        for r in frame[['question_id', *FIELDS]].to_dict('records'):
            for f in FIELDS:
                if not missing(r[f]):
                    qmeta.setdefault(r['question_id'], {}).setdefault(f, r[f])

    def meta_of(r):
        out = {}
        for f in FIELDS:
            out[f] = r[f]
            if missing(r[f]) and f in qmeta.get(r['question_id'], {}):
                out[f] = qmeta[r['question_id']][f]; fills[f] += 1
        return out

    # Claude rows: valid first_code, a non-empty transcription, and not a copy of a dev/test transcript.
    held = {(q, norm_tx(t)) for q, t in zip(dev.question_id, dev.transcription_steps)}
    held |= {(q, norm_tx(tx_dev.get(t, ''))) for q, t in zip(dev.question_id, dev.trace_id)}
    held |= {(q, norm_tx(tx_test.get(t, ''))) for q, t in zip(test.question_id, test.trace_id)}
    held = {k for k in held if k[1]}
    claude, dupes = [], []
    for line in open(R / args.claude):
        r = json.loads(line)
        a = r.get('answer') or {}
        if a.get('first_code') in classes and str(a.get('transcription', '')).strip() and r['trace_id'] in unl.index:
            q = unl.at[r['trace_id'], 'question_id']
            if (q, norm_tx(a['transcription'])) in held:
                dupes.append(r['trace_id']); continue
            claude.append(dict(trace_id=r['trace_id'], question_id=q, **a))
    assert not {c['trace_id'] for c in claude} & set(train.trace_id)
    claude_spans = [(c['trace_id'], c['question_id'], c['first_code'],
                     line_text(c['transcription'], root_line(c['root_line'])) if c['first_code'] != 'NONE' else '')
                    for c in claude]

    def gold_spans(frame):
        out = []
        for r in frame.itertuples(index=False):
            m = [e for e in (json.loads(r.misconceptions) if r.misconceptions else []) if e.get('root')]
            if r.root_code != 'NONE' and m:
                out.append((r.trace_id, r.question_id, r.root_code, str(m[0].get('span', '')).strip()))
        return out

    ex_fit = Exemplars(gold_spans(fit), claude_spans, args.seed, args.min_span_chars)
    ex_all = Exemplars(gold_spans(train), claude_spans, args.seed, args.min_span_chars)
    explain = {r['trace_id']: r['answer'] for r in map(json.loads, open(R / args.explain))}

    # Codes seen per question (fit gold + Claude), for train candidate step 4.
    qcount = {}
    for q, c in list(zip(fit.question_id, fit.root_code)) + [(c['question_id'], c['first_code']) for c in claude]:
        qcount.setdefault(q, Counter())[c] += 1

    sources = {'train': set(), 'dev': set(), 'test': set()}
    ex_stats = {s: Counter() for s in ('claude', 'gold', 'dev', 'test')}

    def examples_for(codes, q, own, bank):
        out = {}
        for c in codes:
            if c != 'NONE':
                src, span = bank(q, c, own)
                if span:
                    out[c] = (src, span)
        return out

    def use_examples(codes, exs, split, stat):
        for c in codes:
            if c != 'NONE':
                stat['codes'] += 1
                if c in exs:
                    stat['with_example'] += 1; sources[split].add(exs[c][0])
        return exs

    # Test blend first: its label-free top-10 lists give realistic train distractors (codes listed together).
    def top_k(P):
        tops, swapped = [], 0
        for q in P:
            top = [classes[j] for j in np.argsort(-q, kind='stable')[:args.k]]
            if 'NONE' not in top:
                top[-1] = 'NONE'; swapped += 1
            tops.append(top)
        return tops, swapped

    tids = test.trace_id.tolist()
    PT = sinkhorn(blend([(R / p, w) for p, w in TEST_BLEND], tids, classes), marginal(train.root_code, classes), args.lam, 50)
    test_tops, test_swapped = top_k(PT)
    cooc = {}
    for top in test_tops:
        for a in top:
            for b in top:
                if a != b and 'NONE' not in (a, b):
                    cooc.setdefault(a, Counter())[b] += 1

    def train_candidates(truth, extra, q, own_code, rng):
        cands = []
        add = lambda c: c in classes and c not in cands and len(cands) < args.k and cands.append(c)
        for c in [truth, 'NONE', *extra]:
            add(c)
        cnt = Counter(qcount.get(q, Counter()))
        cnt[own_code] -= 1
        taken = 0
        for c, n in sorted(cnt.items(), key=lambda x: (-x[1], x[0])):
            if n > 0 and taken < 4 and c not in cands and len(cands) < args.k:
                add(c); taken += 1
        sib = [c for c in section.get(taxonomy.get(truth, {}).get('section'), []) if c != truth]
        for c in (rng.permutation(sib)[:args.max_siblings] if sib else []):
            add(str(c))
        nb = cooc.get(truth)
        if nb:
            keys = sorted(nb)
            w = np.asarray([nb[c] for c in keys], float)
            for i in rng.choice(len(keys), size=len(keys), replace=False, p=w / w.sum()):
                add(keys[i])
        for c in rng.permutation(classes):
            add(str(c))
        return [cands[i] for i in rng.permutation(len(cands))]

    pending = []
    # Phase a: Claude rows, capped per first_code.
    rng = np.random.default_rng(args.seed)
    by_code = {}
    for c in claude:
        by_code.setdefault(c['first_code'], []).append(c)
    picked = []
    for code in sorted(by_code):
        group = by_code[code]
        cap = args.none_cap if code == 'NONE' else args.per_code_cap
        idx = rng.permutation(len(group))[:cap]
        picked += [group[i] for i in sorted(idx)]
    for c in picked:
        r = np.random.default_rng([args.seed, h(c['trace_id']), 0])
        codes = train_candidates(c['first_code'], [c['second_code'], c['third_code']], c['question_id'], c['first_code'], r)
        letter = dict(zip(codes, LETTERS))
        soft = {letter[x]: w for x, w in zip([c['first_code'], c['second_code'], c['third_code']], (.7, .2, .1)) if x in letter}
        z = sum(soft.values())
        line = '-' if c['first_code'] == 'NONE' else root_line(c['root_line'])
        target = f"ROOT_LINE: {line}\nWHY: {why(c['reasoning'], args.why_words)}\nANSWER: {letter[c['first_code']]}"
        pending.append((c['first_code'], meta_of(unl.loc[c['trace_id']]), tail(c['transcription'], args.max_chars),
                        examples_for(codes, c['question_id'], c['trace_id'], ex_fit),
                        dict(trace_id=c['trace_id'], source='claude', phase='a', target=target,
                             letters=list(LETTERS[:len(codes)]), codes=codes, soft={k: v / z for k, v in soft.items()},
                             weight=.8 * (.3 if c['first_code'] == 'OTH-00' else 1.))))
    # Phase b: gold fit rows, repeated with different candidate shuffles.
    for rep in range(args.gold_repeats):
        for g in fit.to_dict('records'):
            r = np.random.default_rng([args.seed, h(g['trace_id']), rep + 1])
            codes = train_candidates(g['root_code'], [], g['question_id'], g['root_code'], r)
            letter = dict(zip(codes, LETTERS))
            m = [e for e in (json.loads(g['misconceptions']) if g['misconceptions'] else []) if e.get('root')]
            line = root_line(m[0]['step']) if m and g['root_code'] != 'NONE' else '-'
            reason = (explain.get(g['trace_id']) or {}).get('reasoning') or (m[0].get('text', '') if m else '')
            pending.append((g['root_code'], meta_of(g), tail(g['transcription_steps'], args.max_chars),
                            examples_for(codes, g['question_id'], g['trace_id'], ex_fit),
                            dict(trace_id=g['trace_id'], source='gold', phase='b',
                                 target=f"ROOT_LINE: {line}\nWHY: {why(reason, args.why_words)}\nANSWER: {letter[g['root_code']]}",
                                 letters=list(LETTERS[:len(codes)]), codes=codes, soft={letter[g['root_code']]: 1.}, weight=1.)))

    # Drop the truth's example at the rate that makes "this candidate has an example" uninformative per source.
    def presence(items):
        t = [truth in exs for truth, _, _, exs, row in items if truth != 'NONE']
        d = [c in exs for truth, _, _, exs, row in items if truth != 'NONE' for c in row['codes'] if c not in (truth, 'NONE')]
        return float(np.mean(t)), float(np.mean(d))
    example_balance = {}
    for src in ('claude', 'gold'):
        items = [x for x in pending if x[4]['source'] == src]
        t, d = presence(items)
        keep = min(1., d / max(t, 1e-9))
        for i, (truth, _, _, exs, row) in enumerate(items):
            if truth in exs and np.random.default_rng([args.seed, h(row['trace_id']), i, 7]).random() > keep:
                del exs[truth]
        example_balance[src] = dict(before=dict(truth=round(t, 4), distractor=round(d, 4)), keep_truth_example=round(keep, 4),
                                    after=dict(zip(('truth', 'distractor'), map(lambda v: round(v, 4), presence(items)))))
    rows, truths = [], []
    for truth, meta, tx, exs, row in pending:
        use_examples(row['codes'], exs, 'train', ex_stats[row['source']])
        rows.append(dict(row, prompt=render(meta, tx, row['codes'], taxonomy, exs)))
        truths.append(truth)
    rows = [{k: r[k] for k in ('trace_id', 'source', 'phase', 'prompt', 'target', 'letters', 'codes', 'soft', 'weight')} for r in rows]

    def ranked(frame, tops, bank, split, texts):
        out = []
        for g, top in zip(frame.to_dict('records'), tops):
            exs = use_examples(top, examples_for(top, g['question_id'], g['trace_id'], bank), split, ex_stats[split])
            tx, meta = tail(texts[g['trace_id']], args.max_chars), meta_of(g)
            orders = [top, top[::-1]]
            out.append(dict(trace_id=g['trace_id'], prompts=[render(meta, tx, o, taxonomy, exs) for o in orders],
                            codes_per_prompt=orders, letters=list(LETTERS[:len(top)])))
        return out

    dids = dev.trace_id.tolist()
    PD = sinkhorn(blend([(R / p, w) for p, w in DEV_BLEND], dids, classes), marginal(fit.root_code, classes), args.lam, 30)
    dev_tops, dev_swapped = top_k(PD)
    dev_rows = ranked(dev, dev_tops, ex_fit, 'dev', tx_dev)
    for d, t in zip(dev_rows, dev.root_code):
        d['truth'] = t
    test_rows = ranked(test, test_tops, ex_all, 'test', tx_test)

    out = R / args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (('cand_train', rows), ('cand_dev', dev_rows), ('cand_test', test_rows)):
        with open(out / f'{name}.jsonl', 'w') as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + '\n')

    # Report.
    lens = token_counter(sorted(glob.glob(args.tokenizer))[0], args.tokenizer_python)
    pct = lambda xs: {f'p{q}': int(np.percentile(xs, q)) for q in (50, 90, 95, 99, 100)}
    full = lens([x['prompt'] + '\n' + x['target'] for x in rows])
    tgt = lens([x['target'] for x in rows])
    letter_ok = all(n == 1 for n in lens([' ' + L for L in LETTERS[:args.k]]))
    train_ids = {x['trace_id'] for x in rows}
    dev_tx = {' '.join(str(t).split()) for t in dev.transcription_steps}
    recall = float(np.mean([d['truth'] in d['codes_per_prompt'][0] for d in dev_rows]))
    sec = {c: r['section'] for c, r in taxonomy.items()}

    def cues(items):  # (codes, truth, prompt) with a non-NONE truth among the candidates
        items = [x for x in items if x[1] != 'NONE' and x[1] in x[0]]
        has = lambda p, i: 'On this question, e.g.' in p.split('\nCandidate error codes:\n', 1)[1].split('\n')[i]
        t = [has(p, codes.index(truth)) for codes, truth, p in items]
        d = [has(p, i) for codes, truth, p in items for i, c in enumerate(codes) if c not in (truth, 'NONE')]
        plural = []
        for codes, truth, p in items:
            n = Counter(sec[c] for c in codes if c != 'NONE')
            plural.append(n[sec[truth]] == max(n.values()))
        return dict(n=len(items), truth_has_example=round(float(np.mean(t)), 4), distractor_has_example=round(float(np.mean(d)), 4),
                    truth_section_is_plurality=round(float(np.mean(plural)), 4),
                    same_section_candidates=round(float(np.mean([sum(sec.get(c) == sec[truth] for c in codes if c not in (truth, 'NONE'))
                                                                 for codes, truth, p in items])), 2))
    qc = fit.groupby('question_id').root_code.apply(set).to_dict()
    dev_items = [(d['codes_per_prompt'][0], d['truth'], d['prompts'][0]) for d in dev_rows]
    unseen_pair = [x for x, q in zip(dev_items, dev.question_id) if x[1] != 'NONE' and x[1] not in qc.get(q, set())]
    shortcut_checks = {'train_claude': cues([(x['codes'], t, x['prompt']) for x, t in zip(rows, truths) if x['source'] == 'claude']),
                       'train_gold': cues([(x['codes'], t, x['prompt']) for x, t in zip(rows, truths) if x['source'] == 'gold']),
                       'dev_error_rows': cues(dev_items), 'dev_unseen_pair_rows': cues(unseen_pair)}
    report = dict(
        rows={'train_by_source_phase': {f'{s}/{p}': n for (s, p), n in sorted(Counter((x['source'], x['phase']) for x in rows).items())},
              'train': len(rows), 'dev': len(dev_rows), 'test': len(test_rows),
              'claude_valid_pool': len(claude), 'claude_picked': len(picked), 'gold_fit_unique': len(fit)},
        per_code={'claude_first_code': dict(sorted(Counter(c['first_code'] for c in picked).items())),
                  'gold_root_code_per_copy': dict(sorted(Counter(fit.root_code).items())),
                  'dev_truth': dict(sorted(Counter(dev.root_code).items()))},
        dev_recall_at_k=recall, k=args.k,
        dev_recall_at_k_error_rows=float(np.mean([d['truth'] in d['codes_per_prompt'][0] for d in dev_rows if d['truth'] != 'NONE'])),
        dev_top1_balanced=float(np.mean([d['truth'] == d['codes_per_prompt'][0][0] for d in dev_rows])),
        none_swapped_in=dict(dev=dev_swapped, test=test_swapped, note='rows where NONE was outside the stable top-k'),
        shortcut_checks=shortcut_checks, example_balance=example_balance,
        claude_dropped_dev_test_duplicates=dict(n=len(dupes), trace_ids=dupes), header_nan_fills=dict(fills),
        classes_never_a_train_answer=sorted(set(classes) - set(truths)),
        train_truth_in_candidates=float(np.mean([t in x['codes'] and x['codes'][LETTERS.index(x['target'][-1])] == t
                                                 for x, t in zip(rows, truths)])),
        train_rows_k_candidates_with_none=float(np.mean([len(x['codes']) == args.k and 'NONE' in x['codes'] for x in rows])),
        example_share={s: dict(codes=v['codes'], with_example=v['with_example'],
                               share=round(v['with_example'] / max(1, v['codes']), 4)) for s, v in ex_stats.items()},
        example_note='share over non-NONE candidate entries; NONE has a fixed line and no example',
        tokens=dict(tokenizer='tiny-Qwen2ForCausalLM-2.5 (Qwen2.5 vocab)',
                    train_prompt_plus_target=pct(full), train_target=pct(tgt),
                    train_over_2048=int(sum(n > 2048 for n in full)),
                    dev_prompt=pct(lens([p for d in dev_rows for p in d['prompts']])),
                    test_prompt=pct(lens([p for d in test_rows for p in d['prompts']])),
                    candidate_letters_single_token=letter_ok),
        leakage=dict(dev_ids=len(dev_ids), dev_ids_in_train_rows=len(dev_ids & train_ids),
                     dev_ids_in_train_exemplar_sources=len(dev_ids & sources['train']),
                     dev_ids_in_dev_exemplar_sources=len(dev_ids & sources['dev']),
                     train_exemplar_sources=len(sources['train']),
                     train_exemplar_sources_gold=len(sources['train'] & set(train.trace_id)),
                     dev_gold_transcripts_verbatim_in_train_prompts=int(sum(
                         any(t and t in ' '.join(x['prompt'].split()) for t in dev_tx) for x in rows if x['source'] == 'gold')),
                     dev_question_ids_with_claude_rows_in_train=int(len(set(dev.question_id) & {c['question_id'] for c in picked}))),
        settings=vars(args) | dict(dev_blend=DEV_BLEND, test_blend=TEST_BLEND, dev_sinkhorn_iters=30, test_sinkhorn_iters=50))
    assert report['leakage']['dev_ids_in_train_rows'] == 0 and report['leakage']['dev_ids_in_train_exemplar_sources'] == 0
    Path(R / args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ('rows', 'dev_recall_at_k', 'train_truth_in_candidates', 'example_share',
                                             'tokens', 'leakage', 'none_swapped_in', 'shortcut_checks', 'example_balance',
                                             'header_nan_fills', 'classes_never_a_train_answer')} |
                     {'claude_dropped_dev_test_duplicates': len(dupes)}, indent=1))
    for x in (next(r for r in rows if r['source'] == 'claude' and r['codes'][LETTERS.index(r['target'][-1])] != 'NONE'),
              next(r for r in rows if r['source'] == 'gold' and r['soft'] and r['codes'][LETTERS.index(r['target'][-1])] != 'NONE')):
        print(f"\n===== SAMPLE {x['source']} {x['trace_id']} codes={x['codes']} soft={x['soft']} weight={x['weight']}\n"
              f"{x['prompt']}\n{x['target']}")
    print(f"\n===== SAMPLE dev {dev_rows[0]['trace_id']} truth={dev_rows[0]['truth']}\n{dev_rows[0]['prompts'][0]}")


if __name__ == '__main__':
    main()
