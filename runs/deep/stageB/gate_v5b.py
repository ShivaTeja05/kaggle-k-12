"""Step 2 (DEEP_PLAN): CPU gates on runs/deep/stageB/data before any GPU spend. Compares with data/v5 where the plan says
"vs current". Writes runs/deep/stageB/gate_v5b.json; exit code 1 if any gate fails."""
from collections import Counter
import glob, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from build_candidate_dataset import DEV_BLEND, blend, marginal, sinkhorn, token_counter
from k12.core import label_space, make_folds, read_csv, topk

D, V5 = ROOT / 'runs/deep/stageB/data', ROOT / 'data/v5'
load = lambda p: [json.loads(l) for l in open(p)]
rows, v5 = load(D / 'cand_train.jsonl'), load(V5 / 'cand_train.jsonl')
dev_b, dev_5 = load(D / 'cand_dev.jsonl'), load(V5 / 'cand_dev.jsonl')
tr = read_csv(ROOT / 'data/raw/train.csv'); classes = label_space(read_csv(ROOT / 'data/raw/taxonomy_v1.csv'))
folds = make_folds(tr); dev, fit = tr[folds == 0].reset_index(drop=True), tr[folds != 0].reset_index(drop=True)
truth = lambda r: r['codes'][r['letters'].index(r['target'].rsplit('ANSWER:', 1)[1].strip())]
VERDICT = re.compile(r'\[[^\]\n]*student[^\]\n]*answer[^\]\n]*correct[^\]\n]*\]', re.I)
CODE = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(classes, key=len, reverse=True))) + r')\b')
res, gates = {}, {}

# 1. Confusion coverage: dev error rows whose blend+Sinkhorn top-1 is a wrong non-NONE code; the (truth, top-1) pair must
#    appear as truth/distractor in >= 5 train rows.
ids = dev.trace_id.tolist(); y = dev.root_code.to_numpy()
PS = sinkhorn(blend([(ROOT / p, w) for p, w in DEV_BLEND], ids, classes), marginal(fit.root_code, classes), .5, 30)
top1 = topk(PS, classes)[:, 0]
pairs = [(t, p) for t, p in zip(y, top1) if t != 'NONE' and p not in ('NONE', t)]


def coverage(data):
    seen = Counter((truth(r), c) for r in data for c in r['codes'] if c not in (truth(r), 'NONE'))
    return float(np.mean([seen[p] >= 5 for p in pairs]))
res['confusion_coverage'] = dict(n_pairs=len(pairs), v5b=round(coverage(rows), 3), v5=round(coverage(v5), 3))
gates['confusion_coverage>=0.70'] = res['confusion_coverage']['v5b'] >= .70

# 2. Unseen likelihood ratios (error rows, weighted): visible cue = candidate shows a same-question example; LOO cue = code was
#    seen for the question among fit gold + Claude rows (own row excluded). Dev reference ~0.49-0.50 for both.
EX = 'On this question, e.g.'


def cards(prompt):
    blk = prompt.split('\nCandidate error codes:\n', 1)[1].split('\nFind the root', 1)[0]
    return {m.group(2): EX in ln for ln in blk.split('\n') if (m := re.match(r'^([A-J])\. ([A-Za-z]+-?\d*):', ln))}


def lr_visible(data, weighted=True):
    t, d, w = [], [], []
    for r in data:
        c = truth(r)
        if c == 'NONE':
            continue
        cs = cards(r['prompt']); ds = [v for k, v in cs.items() if k not in (c, 'NONE')]
        if not ds:
            continue
        t.append(cs.get(c, False)); d.append(np.mean(ds)); w.append(r['weight'] if weighted else 1.)
    pt, pd_ = np.average(t, weights=w), np.average(d, weights=w)
    return round(float((1 - pt) / (1 - pd_)), 4)


unl = read_csv(ROOT / 'data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
qmap = {**unl.question_id.to_dict(), **dict(zip(tr.trace_id, tr.question_id))}
qcount, own = {}, {}
for l in open(ROOT / 'runs/claude/unl_all_v4.jsonl'):
    x = json.loads(l); a = x.get('answer') or {}
    if x['trace_id'] in unl.index and a.get('first_code') in classes:
        qcount.setdefault(qmap[x['trace_id']], Counter())[a['first_code']] += 1; own[x['trace_id']] = a['first_code']
for q, c, t in zip(fit.question_id, fit.root_code, fit.trace_id):
    qcount.setdefault(q, Counter())[c] += 1; own[t] = c


def lr_loo(data, weighted=True):
    t, d, w = [], [], []
    for r in data:
        c = truth(r)
        if c == 'NONE':
            continue
        cnt = Counter(qcount.get(qmap[r['trace_id']], Counter())); cnt[own.get(r['trace_id'])] -= 1
        ds = [cnt.get(x, 0) > 0 for x in r['codes'] if x not in (c, 'NONE')]
        if not ds:
            continue
        t.append(cnt.get(c, 0) > 0); d.append(np.mean(ds)); w.append(r['weight'] if weighted else 1.)
    pt, pd_ = np.average(t, weights=w), np.average(d, weights=w)
    return round(float((1 - pt) / (1 - pd_)), 4)


res['unseen_lr'] = dict(visible_v5b=lr_visible(rows), visible_v5=lr_visible(v5), loo_v5b=lr_loo(rows), loo_v5=lr_loo(v5),
                        dev_reference=0.49)
gates['visible_unseen_lr_in_[0.45,0.6]'] = .45 <= res['unseen_lr']['visible_v5b'] <= .6
gates['loo_unseen_lr_in_[0.45,0.6]'] = .45 <= res['unseen_lr']['loo_v5b'] <= .6

# 3-5. Weights.
w = np.array([r['weight'] for r in rows])
res['kish_ess'] = dict(v5b=int(w.sum() ** 2 / (w ** 2).sum()), rows=len(rows))
gates['kish_ess>=4000'] = res['kish_ess']['v5b'] >= 4000
share = lambda data, code: sum(r['weight'] for r in data if truth(r) == code) / sum(r['weight'] for r in data)
res['ans01_weighted_share'] = dict(v5b=round(share(rows, 'ANS-01'), 4), v5=round(share(v5, 'ANS-01'), 4))
res['ans01_weighted_share']['ratio'] = round(res['ans01_weighted_share']['v5b'] / res['ans01_weighted_share']['v5'], 3)
gates['ans01_share_within_20pct'] = abs(res['ans01_weighted_share']['ratio'] - 1) <= .2
cl = [r for r in rows if r['source'] == 'claude']
res['claude_none_share'] = round(share(cl, 'NONE'), 4)
gates['claude_none_share~0.338'] = abs(res['claude_none_share'] - .338) <= .015
res['weight_mass'] = {k: round(sum(r['weight'] for r in rows if f"{r['source']}/{r.get('kind')}" == k), 1)
                      for k in sorted({f"{r['source']}/{r.get('kind')}" for r in rows})}
res['generic_share_of_error_weight'] = dict(
    claude=round(sum(r['weight'] for r in cl if truth(r) != 'NONE' and truth(r).split('-')[0] in
                     {'CAL', 'WOP', 'ANS', 'OTH', 'NOT', 'FRM', 'VIS', 'CHK', 'UNT', 'REL'}) /
                 sum(r['weight'] for r in cl if truth(r) != 'NONE'), 3), gold_reference=0.398)

# 6. Tokens.
lens = token_counter(sorted(glob.glob(str(Path.home() / '.cache/huggingface/hub/models--trl-internal-testing--'
                                          'tiny-Qwen2ForCausalLM-2.5/snapshots/*/tokenizer.json')))[0], '/usr/bin/python3')
full = lens([r['prompt'] + '\n' + r['target'] for r in rows])
res['tokens_prompt_plus_target'] = {f'p{q}': int(np.percentile(full, q)) for q in (50, 95, 99, 100)}
res['tokens_dev_prompt_p99'] = int(np.percentile(lens([p for d in dev_b for p in d['prompts']]), 99))
res['letters_single_token'] = all(n == 1 for n in lens([' ' + L for L in 'ABCDEFGHIJ']))
gates['p99_tokens<=1600'] = res['tokens_prompt_plus_target']['p99'] <= 1600 and res['letters_single_token']

# 7. ROOT_LINE refers to a line present in the (possibly tail-cut) transcript.
bad = []
for r in rows:
    line = re.search(r'^ROOT_LINE: (\S+)', r['target'], re.M).group(1)
    work = r['prompt'].split('Student working (line by line):\n', 1)[1].split('\n\nCandidate error codes:', 1)[0]
    if line != '-' and not re.search(rf'(?m)^\s*{line}\s*\|', work):
        bad.append(r['trace_id'])
    if line == '-' and truth(r) != 'NONE':
        bad.append(r['trace_id'] + ':dash')
res['root_line_missing'] = dict(n=len(bad), examples=bad[:5])
res['root_line_dash_on_error_rows'] = sum(x.endswith(':dash') for x in bad)
gates['root_lines_exist'] = len([b for b in bad if not b.endswith(':dash')]) <= .005 * len(rows)

# 8. Smart fill: recall@10 on dev, by slice, vs v5 candidates.
qc = fit.groupby('question_id').root_code.apply(set).to_dict()
sl = {t: 'none' if c == 'NONE' else 'unseen_q' if q not in qc else 'seen_pair' if c in qc[q] else 'new_pair'
      for t, q, c in zip(dev.trace_id, dev.question_id, dev.root_code)}


def recall(data, keep=lambda s: True):
    xs = [d['truth'] in d['codes_per_prompt'][0] for d in data if keep(sl[d['trace_id']])]
    return round(float(np.mean(xs)), 4)
nu = lambda s: s in ('new_pair', 'unseen_q')
res['dev_recall10'] = dict(v5b=recall(dev_b), v5=recall(dev_5), new_unseen_v5b=recall(dev_b, nu), new_unseen_v5=recall(dev_5, nu),
                           seen_pair_v5b=recall(dev_b, lambda s: s == 'seen_pair'), seen_pair_v5=recall(dev_5, lambda s: s == 'seen_pair'))
smart = json.loads((D / 'build_meta.json').read_text())['settings']['smart_fill']
res['smart_fill_adopted'] = smart  # off by default: it lowered new/unseen recall (0.849 vs 0.859) when tried
gates['dev_recall'] = ((res['dev_recall10']['new_unseen_v5b'] - res['dev_recall10']['new_unseen_v5'] >= .03 if smart else True)
                       and res['dev_recall10']['v5b'] >= res['dev_recall10']['v5'])

# Hygiene checks (hard).
res['hygiene'] = dict(
    verdict_in_train_transcripts=sum(bool(VERDICT.search(r['prompt'].split('\n\nCandidate error codes:')[0])) for r in rows),
    verdict_in_dev_transcripts=sum(bool(VERDICT.search(p.split('\n\nCandidate error codes:')[0])) for d in dev_b for p in d['prompts']),
    code_id_in_why=sum(bool(CODE.search(r['target'].split('WHY:', 1)[1].rsplit('ANSWER:', 1)[0])) for r in rows),
    dev_ids_in_train=len(set(dev.trace_id) & {r['trace_id'] for r in rows}),
    dev_ids_as_train_exemplars=len(set(dev.trace_id) & {s for r in rows for s in r.get('source_ids', [])}),
    truth_in_candidates=float(np.mean([truth(r) in r['codes'] for r in rows])))
gates['hygiene'] = (not any(res['hygiene'][k] for k in ('verdict_in_train_transcripts', 'verdict_in_dev_transcripts', 'code_id_in_why',
                                                        'dev_ids_in_train', 'dev_ids_as_train_exemplars'))
                    and res['hygiene']['truth_in_candidates'] == 1.)
res['gates'] = gates; res['all_pass'] = all(gates.values())
(ROOT / 'runs/deep/stageB/gate_v5b.json').write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
sys.exit(0 if res['all_pass'] else 1)
