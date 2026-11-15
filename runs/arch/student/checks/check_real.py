"""Throwaway CPU checks of src/k12/candidate_student.py on real data/v5 rows (tiny Qwen2)."""
import json, sys, types
from pathlib import Path
import numpy as np, torch
from k12.candidate_student import Student, read_jsonl, ANSWER, SEED, WORKING, CANDIDATES
from k12.core import aligned_bundle

C = Path('runs/arch/student/checks')
args = types.SimpleNamespace(model=None, device='cpu', load_4bit=False, max_length=2048)
st = Student(args, adapter=str(C / 'train/adapter'))
tok = st.tokenizer
rows = read_jsonl(C / 'train40.jsonl')
torch.manual_seed(0)

print('== 1. answer-position indexing (decoded token at letter position) ==')
for r in [rows[0], rows[10], rows[25]]:
    x = st.train_item(r)
    a = x['answer']
    print(r['trace_id'], r['source'], 'target letter', r['target'][-1],
          '| tok@answer=%r tok@answer+1=%r last=%r' % (tok.decode([x['ids'][a]]), tok.decode([x['ids'][a + 1]]), tok.decode([x['ids'][-1]])),
          '| decoded tail:', repr(tok.decode(x['ids'][a - 3:])))
    assert tok.decode([x['ids'][a + 1]]).strip() == r['target'][-1]

print('== 2. soft letter loss uses logits at position predicting the letter (off-by-one) ==')
items = [st.train_item(r) for r in rows[:4]]
st.model.eval()
with torch.no_grad():
    _, lm, letter = st.loss(items, 1., 1.)
    manual, shifted_m1, shifted_p1, lm_manual = [], [], [], []
    for x in items:
        ids = torch.tensor([x['ids']])
        logits = st.model(input_ids=ids).logits[0].float()
        soft = torch.tensor(x['soft'], dtype=torch.float32)
        f = lambda pos: -(soft * logits[pos, x['letters']].log_softmax(-1)).sum().item()
        manual.append(f(x['answer'])); shifted_m1.append(f(x['answer'] - 1)); shifted_p1.append(f(x['answer'] + 1))
        # LM CE over target tokens excluding the letter, via full model logits
        tgt = [c for c in range(x['start'] - 1, len(x['ids']) - 1) if c != x['answer']]
        ce = torch.nn.functional.cross_entropy(logits[tgt], ids[0, [c + 1 for c in tgt]])
        lm_manual.append(ce.item())
        assert x['ids'][x['answer'] + 1] in x['letters'] and x['ids'][x['answer'] + 1] == x['letters'][int(np.argmax(x['soft']))] or x['soft'].max() < 1
print('letter_loss(st.loss)=%.6f manual@answer=%.6f manual@answer-1=%.6f manual@answer+1=%.6f'
      % (letter.item(), np.mean(manual), np.mean(shifted_m1), np.mean(shifted_p1)))
print('lm_loss(st.loss)=%.6f manual full-model=%.6f' % (lm.item(), np.mean(lm_manual)))
assert abs(letter.item() - np.mean(manual)) < 1e-4 and abs(lm.item() - np.mean(lm_manual)) < 1e-4
# the gold letter of each item is the arg-max of soft for gold rows; verify soft letter equals the token after answer
for x, r in zip(items, rows[:4]):
    top = r['letters'][int(np.argmax(x['soft']))]
    print(r['trace_id'], 'soft top', top, 'target letter', r['target'][-1], 'match', top == r['target'][-1])

print('== 3. train/predict token consistency ==')
bad_seed = bad_seq = 0
for r in read_jsonl('data/v5/cand_train.jsonl')[::20]:
    head = st.encode('\n' + r['target'][:r['target'].rindex(ANSWER) + len(ANSWER)])
    seed = st.encode(SEED)
    bad_seed += head[:len(seed)] != seed
    # predict path: prompt ids + encode('\n' + text) where text = 'ROOT_LINE:' + generated, cut at ANSWER:
    text = r['target'][:r['target'].rindex(ANSWER) + len(ANSWER)]
    x = st.train_item(r)
    seq = st.prompt_ids(r['prompt'], 10 ** 6)[0] + st.encode('\n' + text)
    bad_seq += seq != x['ids'][:x['answer'] + 1]
print('rows checked', len(read_jsonl('data/v5/cand_train.jsonl')[::20]), 'seed-prefix mismatches', bad_seed, 'predict-vs-train sequence mismatches', bad_seq)

print('== 4. truncation keeps header, candidate block and target (real rows, small max_length) ==')
allrows = read_jsonl('data/v5/cand_train.jsonl')
long = sorted(allrows, key=lambda r: -len(r['prompt']))[:50]
for ml in (700, 500):
    st.max_length = ml
    n_cut = ok = 0; over = []
    for r in long:
        x = st.train_item(r)
        n_cut += x['cut']
        dec = tok.decode(x['ids'])
        i, j = r['prompt'].find(WORKING), r['prompt'].find(CANDIDATES)
        header, cand = r['prompt'][:i + len(WORKING)], r['prompt'][j:]
        good = dec.startswith(header) and cand in dec and dec.endswith(r['target'] + tok.eos_token)
        ok += good
        if len(x['ids']) > ml: over.append(len(x['ids']))
    print('max_length', ml, 'rows', len(long), 'cut', n_cut, 'header+candidates+target intact', ok, 'over max_length', len(over), over[:5])
st.max_length = 2048

print('== 5. generation stops after ANSWER: (forced continuation via logits processor) ==')
from transformers import LogitsProcessor, LogitsProcessorList
forced_text = ' L2\nWHY: Siswa salah menjumlahkan.\nANSWER: C\nWHY: extra text that must not appear'
forced = st.encode(forced_text)
class Force(LogitsProcessor):
    def __init__(self, start): self.start = start
    def __call__(self, input_ids, scores):
        k = input_ids.shape[1] - self.start
        s = torch.full_like(scores, -1e9)
        s[:, forced[k] if k < len(forced) else tok.eos_token_id] = 0
        return s
dev = read_jsonl(C / 'dev12.jsonl')
prompts = [st.prompt_ids(d['prompts'][0], 2048 - 160 - 20)[0] + st.encode(SEED) for d in dev[:3]]
width = max(map(len, prompts))
ids = torch.full((3, width), tok.pad_token_id); mask = torch.zeros_like(ids)
for r, s in enumerate(prompts): ids[r, width - len(s):] = torch.tensor(s); mask[r, width - len(s):] = 1
out = st.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=60, do_sample=False, stop_strings=[ANSWER],
                        tokenizer=tok, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                        logits_processor=LogitsProcessorList([Force(width)]))
for o in out:
    t = tok.decode(o[width:], skip_special_tokens=True)
    print('generated %d new tokens: %r' % (len(o) - width, t))
    assert t.rstrip().endswith('ANSWER:') or t.rstrip().endswith('ANSWER')

print('== 6. predict outputs load with aligned_bundle ==')
for d in ['train/predict_dev12', 'predict']:
    with np.load(C / d / 'predictions.npz') as b:
        ids_, classes = b['ids'].tolist(), b['classes'].tolist()
    p = aligned_bundle(C / d / 'predictions.npz', ids_, classes)
    gens = [json.loads(l) for l in (C / d / 'generations.jsonl').read_text().splitlines()]
    print(d, 'shape', p.shape, 'row sums', float(p.sum(1).min()), float(p.sum(1).max()),
          'gens', len(gens), 'orders', sorted({g['order'] for g in gens}), 'parsed', sum(g['parsed'] for g in gens),
          'max gen chars', max(len(g['text']) for g in gens))
a = np.load(C / 'train/predict_dev12/predictions.npz')['p']; b = np.load(C / 'predict/predictions.npz')['p']
print('train --predict vs standalone predict max abs diff', float(np.abs(a - b).max()))
