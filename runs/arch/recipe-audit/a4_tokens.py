"""Token-length audit with a Qwen2.5/Qwen3-vocabulary BPE (151,665 tokens; proxy for the Qwen3 tokenizer).

Run with /usr/bin/python3 -I (the project venv has no tokenizers package). Stdlib + tokenizers only.
"""
import csv, json, statistics, sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path('/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'); OUT = ROOT / 'runs/arch/recipe-audit'
tok = Tokenizer.from_file(str(Path.home() / '.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda/tokenizer.json'))
n = lambda s: len(tok.encode(s).ids)
csv.field_size_limit(10**9)
rd = lambda p: list(csv.DictReader(open(p, newline='', encoding='utf-8')))
train = rd(ROOT / 'data/raw/train.csv'); tax = rd(ROOT / 'data/raw/taxonomy_v1.csv'); test = rd(ROOT / 'data/raw/test.csv')
dev_ids = set(json.loads((OUT / 'dev_ids.json').read_text()))
byid = {r['trace_id']: r for r in train}
defs_short = 'Error codes:\n' + '\n'.join(f"{r['id']}: {r['name']}" for r in tax)
defs_long = 'Error codes:\n' + '\n'.join(f"{r['id']}: {r['name']} - {r['description']}" for r in tax)
def prompt(r, tr, d):
    return (f"{d}\n\nGrade: {r['grade']}\nTopic: {r['topic_name']}\nQuestion: {r['question_text']}\n"
            f"Answer key: {r['correct_answer']}\nStudent working (line by line):\n{tr}\n\n"
            "Identify the root major error the rest of the working follows from, or NONE if there is no "
            "major error. ROOT:")
def q(xs, ps=(.5, .9, .99, 1.)):
    xs = sorted(xs); return {str(p): xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] for p in ps}
R = {'tokenizer': 'Qwen2.5 BPE (151,665 vocab) as proxy for Qwen3', 'defs_short_tokens': n(defs_short), 'defs_long_tokens': n(defs_long)}
tx_dev = json.loads((ROOT / 'runs/pod/tx/tx_dev.json').read_text())
tx_test = json.loads((ROOT / 'runs/pod/tx/tx_test.json').read_text())
dev = [byid[t] for t in byid if t in dev_ids]
for name, d in (('short', defs_short), ('long', defs_long)):
    for src in ('gold', 'tx'):
        L = [n(prompt(r, r['transcription_steps'] if src == 'gold' else tx_dev[r['trace_id']], d)) for r in dev]
        R[f'dev_prompt_{name}defs_{src}'] = {'quantiles': q(L), 'over_3072': sum(x > 3072 for x in L)}
L = [n(prompt(r, tx_test[r['trace_id']], defs_short)) for r in test]
R['test_prompt_shortdefs_tx'] = {'quantiles': q(L), 'over_3072': sum(x > 3072 for x in L)}
# question text and transcripts alone
R['question_text_tokens_train'] = q([n(r['question_text']) for r in train])
R['gold_transcript_tokens_train'] = q([n(r['transcription_steps']) for r in train])
R['gold_transcript_over_512_(transcriber_target_cap)'] = sum(n(r['transcription_steps'] + '<|im_end|>') > 512 for r in train)
txl = [n(v) for v in tx_dev.values()]; ttl = [n(v) for v in tx_test.values()]
R['tx_dev_tokens'] = q(txl); R['tx_test_tokens'] = q(ttl)
R['tx_dev_>=500_tokens(likely hit 512 generation cap)'] = sum(x >= 500 for x in txl)
R['tx_test_>=500_tokens'] = sum(x >= 500 for x in ttl)
# repetition loops in machine transcripts: same line repeated >=4 times
def loopy(s):
    ls = [l.split('|', 1)[-1].strip() for l in s.splitlines() if l.strip()]
    return any(ls.count(x) >= 4 and len(x) > 3 for x in set(ls))
R['tx_dev_repetition_loops'] = sum(loopy(v) for v in tx_dev.values()); R['tx_test_repetition_loops'] = sum(loopy(v) for v in tx_test.values())
# line counts: gold vs machine on dev
gl = [len([l for l in r['transcription_steps'].splitlines() if l.strip()]) for r in dev]
ml = [len([l for l in tx_dev[r['trace_id']].splitlines() if l.strip()]) for r in dev]
R['dev_line_count_gold_vs_tx'] = {'gold_median': statistics.median(gl), 'tx_median': statistics.median(ml),
                                   'rows_line_count_differs': sum(a != b for a, b in zip(gl, ml))}
# Explanation targets (text_diagnose explain loss cap 256 tokens incl. eos)
v3 = [json.loads(l) for l in (ROOT / 'data/v3/train_v3.jsonl').read_text().splitlines()]
ex = [n(r['target'].split('\nREADING:', 1)[0]) for r in v3]
R['gold_explanation_tokens'] = q(ex); R['gold_explanation_over_255'] = sum(x > 255 for x in ex)
ps = [json.loads(l) for l in (ROOT / 'data/v3/claude_unl_v4.jsonl').read_text().splitlines()]
pex = [n(r['target'].split('\nREADING:', 1)[0]) for r in ps]
R['claude_explanation_tokens'] = q(pex); R['claude_explanation_over_255'] = sum(x > 255 for x in pex)
R['claude_explanation_over_255_share'] = round(sum(x > 255 for x in pex) / len(pex), 3)
# VL classifier teacher target cap 384 (pod_train --teacher-tokens): how often READING is cut / WHY is cut
full = [n(r['target']) for r in v3]; R['v3_full_target_tokens'] = q(full); R['v3_targets_over_384'] = sum(x > 384 for x in full)
pfull = [n(r['target']) for r in ps]; R['claude_full_target_over_384_share'] = round(sum(x > 384 for x in pfull) / len(pfull), 3)
R['claude_head_over_384'] = sum(x > 384 for x in pex)
# Style gap: Claude transcripts used as fit text for pseudo rows vs gold
ctr = [r['target'].split('READING:\n', 1)[-1] for r in ps]
R['claude_transcript_tokens'] = q([n(x) for x in ctr[:3000]])
# Gold explanation brevity vs Claude: words
R['gold_why_words_median'] = statistics.median(len(r['target'].split('\nREADING:')[0].split()) for r in v3 if r['root_code'] != 'NONE')
R['claude_why_words_median'] = statistics.median(len(x.split('\nREADING:')[0].split()) for x in (r['target'] for r in ps) if not x.startswith('ROOT: NONE'))
(OUT / 'a4_tokens.json').write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
