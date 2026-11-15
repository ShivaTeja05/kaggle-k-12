"""Independent data-lens checks for data/v5 (throwaway)."""
import json, re, sys, glob, collections, hashlib
from pathlib import Path
import numpy as np
from k12.core import read_csv, label_space, make_folds, aligned_bundle, normalize
R = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12')
L = 'ABCDEFGHIJ'
tr = read_csv(R/'data/raw/train.csv'); te = read_csv(R/'data/raw/test.csv'); tax = read_csv(R/'data/raw/taxonomy_v1.csv')
classes = label_space(tax); folds = make_folds(tr)
dev = tr[folds == 0]; fit = tr[folds != 0]
dev_ids, fit_ids = set(dev.trace_id), set(fit.trace_id)
T = [json.loads(x) for x in open(R/'data/v5/cand_train.jsonl')]
D = [json.loads(x) for x in open(R/'data/v5/cand_dev.jsonl')]
X = [json.loads(x) for x in open(R/'data/v5/cand_test.jsonl')]
out = {}
P = lambda k, v: (out.__setitem__(k, v), print(k, '=', v))
P('n', [len(T), len(D), len(X)])
# (1) leakage: ids
P('dev_ids_as_train_rows', len(dev_ids & {r['trace_id'] for r in T}))
P('test_ids_as_train_rows', len(set(te.trace_id) & {r['trace_id'] for r in T}))
unl = read_csv(R/'data/semisupervised/competition_data/unlabelled.csv')
P('unl_ids_overlap_train_test', [len(set(unl.trace_id) & set(tr.trace_id)), len(set(unl.trace_id) & set(te.trace_id))])
P('unl_image_overlap_train_test', [len(set(unl.image_file) & set(tr.image_file)), len(set(unl.image_file) & set(te.image_file))] if 'image_file' in unl else 'no image_file col')
# exemplar spans extracted from prompts
EX = re.compile(r' On this question, e\.g\. "(.*)"$')
CL = re.compile(r'^([A-J])\. (\S+?)(?::| -|$)')
def norm(s): return ' '.join(str(s).split())
def clip(s, n=60): s = norm(s); return s if len(s) <= n else s[:n].rstrip()
def cand_lines(p):
    blk = p.split('\nCandidate error codes:\n', 1)[1].split('\nFind the root major error')[0]
    return blk.split('\n')
gold_span = collections.defaultdict(set)  # (q, code, clipped span) -> trace ids
for r in tr.itertuples():
    for e in (json.loads(r.misconceptions) if r.misconceptions else []):
        if e.get('root') and r.root_code != 'NONE':
            gold_span[(r.question_id, r.root_code, clip(e.get('span', '')))].add(r.trace_id)
# Claude spans
LINE = re.compile(r'^\s*L(\d+)\s*\|\s?(.*)$')
claude = {}
for line in open(R/'runs/claude/unl_all_v4.jsonl'):
    r = json.loads(line); claude[r['trace_id']] = r.get('answer') or {}
uq = dict(zip(unl.trace_id, unl.question_id))
cl_span = collections.defaultdict(set)
for tid, a in claude.items():
    if a.get('first_code') in classes and a.get('first_code') != 'NONE' and tid in uq:
        m = re.match(r'^\s*L(\d+)', str(a.get('root_line') or ''))
        if not m: continue
        for raw in str(a.get('transcription', '')).splitlines():
            mm = LINE.match(raw)
            if mm and mm.group(1) == m.group(1):
                cl_span[(uq[tid], a['first_code'], clip(mm.group(2).strip()))].add(tid); break
qid = {**dict(zip(tr.trace_id, tr.question_id)), **dict(zip(te.trace_id, te.question_id)), **uq}
def audit(rows, split, codes_key):
    bad_dev_only, own, unknown, total, srcs_dev = [], 0, [], 0, 0
    for r in rows:
        prompts = r['prompts'][:1] if 'prompts' in r else [r['prompt']]
        codes = r['codes_per_prompt'][0] if 'prompts' in r else r['codes']
        for ln, c in zip(cand_lines(prompts[0]), codes):
            m = EX.search(ln)
            if not m: continue
            total += 1
            s = m.group(1); key = (qid[r['trace_id']], c, s)
            g = gold_span.get(key, set()); cs = cl_span.get(key, set())
            gf = g - dev_ids; gd = g & dev_ids
            if r['trace_id'] in g | cs and not ((g | cs) - {r['trace_id']}): own += 1
            if split != 'test' and gd and not gf and not cs: bad_dev_only.append((r['trace_id'], c, s))
            if not g and not cs: unknown.append((r['trace_id'], c, s))
    return dict(spans=total, dev_only_source=len(bad_dev_only), own_only=own, unknown_source=len(unknown),
                ex_dev_only=bad_dev_only[:3], ex_unknown=unknown[:3])
P('train_exemplar_audit', audit(T, 'train', 'codes'))
P('dev_exemplar_audit', audit(D, 'dev', 'codes_per_prompt'))
P('test_exemplar_audit', audit(X, 'test', 'codes_per_prompt'))
# dev gold spans verbatim in train prompts (only spans unique to fold-0 rows)
fit_spans = {norm(e.get('span', '')) for r in fit.itertuples() for e in (json.loads(r.misconceptions) if r.misconceptions else [])}
dev_only_spans = {norm(e.get('span', '')) for r in dev.itertuples() for e in (json.loads(r.misconceptions) if r.misconceptions else [])} - fit_spans
dev_only_spans = {s for s in dev_only_spans if len(s) >= 25}
tp = [norm(r['prompt']) for r in T]
hits = [(s, sum(s in p for p in tp)) for s in dev_only_spans]
hits = [h for h in hits if h[1]]
P('dev_only_spans_ge25ch', len(dev_only_spans)); P('dev_only_spans_found_in_train_prompts', len(hits)); P('examples', hits[:5])
# dev gold transcripts in train prompts / dev prompts use machine tx
dev_tx = [norm(t) for t in dev.transcription_steps]
P('dev_gold_transcripts_in_train_prompts(>=40ch)', sum(any(t in p for p in tp) for t in dev_tx if len(t) >= 40))
txd = json.load(open(R/'runs/pod/tx/tx_dev.json')); txt = json.load(open(R/'runs/pod/tx/tx_test.json'))
def work(p): return p.split('Student working (line by line):\n', 1)[1].split('\n\nCandidate error codes:\n')[0]
P('dev_prompt_transcript_is_tail_of_machine_tx', np.mean([all(txd[r['trace_id']].strip().endswith(work(p)) for p in r['prompts']) for r in D]))
P('test_prompt_transcript_is_tail_of_machine_tx', np.mean([all(txt[r['trace_id']].strip().endswith(work(p)) for p in r['prompts']) for r in X]))
P('max_working_chars', [max(len(work(r['prompt'])) for r in T), max(len(work(p)) for r in D for p in r['prompts']), max(len(work(p)) for r in X for p in r['prompts'])])
# Claude transcriptions near-identical to dev gold transcriptions (possible duplicate photos)
dev_tx_set = {re.sub(r'L\d+ \| ', '', t).lower() for t in dev_tx}
ct = {tid: re.sub(r'L\d+ \| ', '', norm(a.get('transcription', ''))).lower() for tid, a in claude.items()}
used_claude = {r['trace_id'] for r in T if r['source'] == 'claude'}
P('claude_train_rows_whose_tx_equals_a_dev_gold_tx', sum(ct[t] in dev_tx_set for t in used_claude))
P('claude_train_rows_whose_tx_equals_a_test_machine_tx', sum(ct[t] in {re.sub(r'L\d+ \| ', '', norm(v)).lower() for v in txt.values()} for t in used_claude))
# (2) truth in candidates
truth = {}
for r in T:
    truth_code = claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else tr.set_index('trace_id').root_code[r['trace_id']] if False else None
fr = dict(zip(tr.trace_id, tr.root_code))
tt = [claude[r['trace_id']]['first_code'] if r['source'] == 'claude' else fr[r['trace_id']] for r in T]
ans = [r['target'].rsplit('ANSWER: ', 1)[1] for r in T]
P('train_truth_in_codes', np.mean([t in r['codes'] for r, t in zip(T, tt)]))
P('train_answer_letter_maps_to_truth', np.mean([r['codes'][L.index(a)] == t for r, a, t in zip(T, ans, tt)]))
P('dev_recall10', np.mean([r['truth'] in r['codes_per_prompt'][0] for r in D]))
P('dev_truth_matches_gold', np.mean([r['truth'] == fr[r['trace_id']] for r in D]))
P('dev_ids_equal_fold0', {r['trace_id'] for r in D} == dev_ids)
# (3) NONE / letters / soft
def chk_codes(codes, letters):
    return len(codes) == 10 == len(letters) and len(set(codes)) == 10 and 'NONE' in codes and letters == list(L) and all(c in classes for c in codes)
P('train_codes_ok', np.mean([chk_codes(r['codes'], r['letters']) for r in T]))
P('dev_codes_ok', np.mean([all(chk_codes(c, r['letters']) for c in r['codes_per_prompt']) and r['codes_per_prompt'][1] == r['codes_per_prompt'][0][::-1] for r in D]))
P('test_codes_ok', np.mean([all(chk_codes(c, r['letters']) for c in r['codes_per_prompt']) and r['codes_per_prompt'][1] == r['codes_per_prompt'][0][::-1] for r in X]))
# prompt letters/codes consistent with codes list
def plines_ok(p, codes):
    ls = cand_lines(p)
    if len(ls) != len(codes): return False
    for l, c, ln in zip(L, codes, ls):
        exp = f'{l}. NONE: No major error' if c == 'NONE' else f'{l}. {c}: '
        if not ln.startswith(exp): return False
        if c == 'NONE' and ln != f'{l}. NONE: No major error - the working is correct or has only minor slips.': return False
    return True
P('train_prompt_candidate_block_ok', np.mean([plines_ok(r['prompt'], r['codes']) for r in T]))
P('dev_prompt_candidate_block_ok', np.mean([all(plines_ok(p, c) for p, c in zip(r['prompts'], r['codes_per_prompt'])) for r in D]))
P('test_prompt_candidate_block_ok', np.mean([all(plines_ok(p, c) for p, c in zip(r['prompts'], r['codes_per_prompt'])) for r in X]))
P('soft_keys_subset_and_sum1', np.mean([set(r['soft']) <= set(r['letters']) and abs(sum(r['soft'].values()) - 1) < 1e-9 for r in T]))
P('soft_argmax_is_answer', np.mean([max(r['soft'], key=r['soft'].get) == a for r, a in zip(T, ans)]))
# expected soft for Claude rows
bad_soft = []
for r in T:
    if r['source'] != 'claude': 
        if r['soft'] != {ans[T.index(r)] if False else r['target'][-1]: 1.0}: bad_soft.append(r['trace_id'])
        continue
    a = claude[r['trace_id']]; lt = dict(zip(r['codes'], L)); e = collections.defaultdict(float)
    for c, w in zip([a['first_code'], a.get('second_code'), a.get('third_code')], (.7, .2, .1)):
        if c in lt: e[lt[c]] += w
    z = sum(e.values()); e = {k: v / z for k, v in e.items()}
    if set(e) != set(r['soft']) or any(abs(e[k] - r['soft'][k]) > 1e-9 for k in e): bad_soft.append((r['trace_id'], a['first_code'], a.get('second_code'), a.get('third_code'), r['soft']))
P('soft_mismatch_vs_independent_(dup-codes summed)', len(bad_soft)); P('soft_mismatch_examples', bad_soft[:4])
dup = collections.Counter()
for r in T:
    if r['source'] == 'claude':
        a = claude[r['trace_id']]; cs = [a['first_code'], a.get('second_code'), a.get('third_code')]
        dup['first==second' if cs[0] == cs[1] else 'first==third' if cs[0] == cs[2] else 'second==third' if cs[1] == cs[2] and cs[1] else 'distinct'] += 1
        dup['2nd_not_in_classes'] += cs[1] not in classes; dup['3rd_not_in_classes'] += cs[2] not in classes
        dup['2nd_in_cands'] += cs[1] in r['codes']; dup['3rd_in_cands'] += cs[2] in r['codes']
P('claude_code_dup_stats', dict(dup))
# (4) target format
TG = re.compile(r'^ROOT_LINE: (L\d+|-)\nWHY: ([^\n]+)\nANSWER: ([A-J])$')
m = [TG.match(r['target']) for r in T]
P('target_format_ok', np.mean([x is not None for x in m]))
P('why_words_max', max(len(x.group(2).split()) for x in m if x)); P('why_empty', sum(not x.group(2).strip() for x in m if x))
P('why_contains_ANSWER_or_ROOT_LINE', sum(('ANSWER' in x.group(2) or 'ROOT_LINE' in x.group(2)) for x in m if x))
P('why_word_hist', np.percentile([len(x.group(2).split()) for x in m if x], [5, 50, 95]).tolist())
P('why_ends_with_sentence_punct', np.mean([x.group(2).rstrip()[-1] in '.!?"\')' for x in m if x]))
P('root_line_dash_by_source_truthNONE', collections.Counter((r['source'], t == 'NONE', x.group(1) == '-') for r, t, x in zip(T, tt, m)))
# gold ROOT_LINE equals root misconception step
gm = {}
for r in fit.itertuples():
    e = [e for e in (json.loads(r.misconceptions) if r.misconceptions else []) if e.get('root')]
    gm[r.trace_id] = (r.root_code, e[0]['step'] if e else None, len(e))
gl = collections.Counter()
for r, x in zip(T, m):
    if r['source'] == 'gold':
        code, step, ne = gm[r['trace_id']]
        mm = re.match(r'^\s*L(\d+)', str(step or ''))
        exp = '-' if code == 'NONE' or not mm else f'L{mm.group(1)}'
        gl['ok' if x.group(1) == exp else 'bad'] += 1
        gl[f'n_root_entries={ne}'] += 1
        if code != 'NONE' and not mm: gl['nonNONE_unparseable_step:' + str(step)] += 1
P('gold_root_line_check', dict(gl))
# claude ROOT_LINE vs answer.root_line; claude NONE rows with non-empty root_line
cl = collections.Counter()
for r, x in zip(T, m):
    if r['source'] == 'claude':
        a = claude[r['trace_id']]; mm = re.match(r'^\s*L(\d+)', str(a.get('root_line') or ''))
        cl['ok' if x.group(1) == (f'L{mm.group(1)}' if mm else '-') else 'bad'] += 1
        if a['first_code'] == 'NONE' and mm: cl['NONE_with_root_line'] += 1
        if a['first_code'] != 'NONE' and not mm: cl['error_without_root_line'] += 1
P('claude_root_line_check', dict(cl))
# WHY source for gold rows
ex = {json.loads(l)['trace_id']: json.loads(l).get('answer') or {} for l in open(R/'runs/claude/train_explain_opus.jsonl')}
P('gold_fit_with_explain', sum(bool(t in ex and (ex[t].get('reasoning') or '')) for t in fit_ids))
agree = [ex[t].get('first_code') == fr[t] for t in fit_ids if t in ex and ex[t].get('first_code')]
P('explain_first_code_agrees_gold', [float(np.mean(agree)) if agree else None, len(agree)])
# reasoning that names a code different from truth
CODE = re.compile(r'\b([A-Z][A-Za-z]{1,2}-\d\d|NONE)\b')
ment = collections.Counter()
for r, t, x in zip(T, tt, m):
    named = set(CODE.findall(x.group(2)))
    if named: ment[(r['source'], 'names_truth' if t in named else 'names_other_only')] += 1
P('why_mentions_codes', dict(ment))
# (5) token lengths
tokp = sorted(glob.glob(str(Path.home()/'.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/*/tokenizer.json')))[0]
import subprocess
code = 'import sys,json;from tokenizers import Tokenizer;t=Tokenizer.from_file(sys.argv[1]);print(json.dumps([len(e.ids) for e in t.encode_batch(json.load(sys.stdin),add_special_tokens=False)]))'
lens = lambda xs: json.loads(subprocess.run(['/usr/bin/python3', '-c', code, tokp], input=json.dumps(xs), capture_output=True, text=True, check=True).stdout)
pc = lambda v: {f'p{q}': int(np.percentile(v, q)) for q in (50, 95, 99, 100)}
full = lens([r['prompt'] + '\n' + r['target'] for r in T])
P('tok_train_full', pc(full)); P('tok_train_target', pc(lens([r['target'] for r in T])))
P('tok_dev_prompt', pc(lens([p for r in D for p in r['prompts']]))); P('tok_test_prompt', pc(lens([p for r in X for p in r['prompts']])))
P('letter_single_token', lens([' ' + l for l in L])); P('ANSWER_colon_letter_tokens', [lens(['ANSWER: A']), lens(['\nANSWER: A'])])
# (6) test coverage
P('test_ids_cover', [len({r['trace_id'] for r in X}), set(te.trace_id) == {r['trace_id'] for r in X}, len(X) == len(set(r['trace_id'] for r in X)), all(len(r['prompts']) == 2 for r in X)])
P('test_none_rows_have_truth_key', sum('truth' in r for r in X))
# (7) balance / repeats
gold = [r for r in T if r['source'] == 'gold']
cnt = collections.Counter(r['trace_id'] for r in gold)
P('gold_repeat_counts', dict(collections.Counter(cnt.values()))); P('gold_ids_equal_fit', set(cnt) == fit_ids)
by = collections.defaultdict(list)
for r in gold: by[r['trace_id']].append(r)
P('gold_copies_with_different_order', np.mean([v[0]['codes'] != v[1]['codes'] for v in by.values()]))
P('gold_copies_with_same_candidate_set', np.mean([set(v[0]['codes']) == set(v[1]['codes']) for v in by.values()]))
P('gold_copies_identical_prompt', np.mean([v[0]['prompt'] == v[1]['prompt'] for v in by.values()]))
P('phase_source_weight', dict(collections.Counter((r['source'], r['phase'], round(r['weight'], 4)) for r in T)))
P('oth00_weight_ok', all(abs(r['weight'] - (.24 if t == 'OTH-00' else .8)) < 1e-9 for r, t in zip(T, tt) if r['source'] == 'claude'))
P('claude_dup_trace', len(used_claude) == sum(r['source'] == 'claude' for r in T))
ccnt = collections.Counter(t for r, t in zip(T, tt) if r['source'] == 'claude')
P('claude_cap_violations', {c: n for c, n in ccnt.items() if n > (2000 if c == 'NONE' else 200)})
# pool sizes to confirm cap saturation
pool = collections.Counter(a['first_code'] for t, a in claude.items() if a.get('first_code') in classes and str(a.get('transcription', '')).strip() and t in uq)
P('claude_pool_vs_picked_for_short_codes', {c: (pool[c], ccnt.get(c, 0)) for c in pool if ccnt.get(c, 0) < min(pool[c], 2000 if c == 'NONE' else 200)})
tot = collections.Counter(tt)
P('train_label_share_NONE', tot['NONE'] / len(T)); P('dev_NONE_share', float((dev.root_code == 'NONE').mean()))
P('classes_missing_in_train', sorted(set(classes) - set(tt))); P('dev_truth_codes_missing_in_train', sorted(set(dev.root_code) - set(tt)))
# answer letter position balance
P('answer_letter_hist', dict(sorted(collections.Counter(ans).items())))
P('NONE_letter_position_hist', dict(sorted(collections.Counter(L[r['codes'].index('NONE')] for r in T).items())))
P('dev_truth_rank_hist', dict(sorted(collections.Counter(r['codes_per_prompt'][0].index(r['truth']) if r['truth'] in r['codes_per_prompt'][0] else -1 for r in D).items())))
# truth position in train vs dev candidate code ordering: is truth always adjacent to NONE etc? check that shuffle is not trivially leaking
P('train_truth_letter_equals_NONE_letter_minus1', np.mean([r['codes'].index(t) == r['codes'].index('NONE') - 1 for r, t in zip(T, tt) if t != 'NONE']))
json.dump(out, open(R/'runs/arch/student/checks/check_data.json', 'w'), indent=1, default=str)
