"""Throwaway: token counts per row with the Qwen2.5/Qwen3 BPE (tiny Qwen2 tokenizer), padded batch tokens, GPU cost estimate."""
import json, math, types
import numpy as np
from k12.candidate_student import Student, read_jsonl, SEED

args = types.SimpleNamespace(model=None, device='cpu', load_4bit=False, max_length=2048)
st = Student(args)
rows = read_jsonl('data/v5/cand_train.jsonl')
items = [st.train_item(r) for r in rows]
L = np.array([len(x['ids']) for x in items]); phase = np.array([r['phase'] for r in rows])
tgt = np.array([len(x['ids']) - x['start'] for x in items])
padded = 0
for name in ('a', 'b'):  # same plan as candidate_student.train, seed 2026, epoch 0, batch 4
    ix = np.flatnonzero(phase == name)
    ix = ix[np.random.default_rng(2026 + ord(name)).permutation(len(ix))]
    padded += sum(4 * L[ix[s:s + 4]].max() if len(ix[s:s + 4]) == 4 else len(ix[s:s + 4]) * L[ix[s:s + 4]].max() for s in range(0, len(ix), 4))
print(json.dumps({'rows': len(L), 'truncated': int(sum(x['cut'] for x in items)), 'tokens_total': int(L.sum()), 'tokens_padded_bs4': int(padded),
                  'pad_overhead': round(padded / L.sum() - 1, 3), 'p50': int(np.median(L)), 'p95': int(np.percentile(L, 95)), 'max': int(L.max()),
                  'mean': round(float(L.mean()), 1), 'target_mean': round(float(tgt.mean()), 1),
                  'by_phase_mean': {p: round(float(L[phase == p].mean()), 1) for p in 'ab'}}))
ref_s_per_tok = 1.0 / 1400  # text_diagnose 8B, A100 80GB, batch 4, explain loss on
for basis, toks in (('real', L.sum()), ('padded', padded)):
    h = toks * ref_s_per_tok / 3600
    print(basis, 'A100 h=%.2f' % h, {'A100 $1.79': round(h * 1.79, 2), 'L40S $1.09 (1.0-1.4x A100)': [round(h * f * 1.09, 2) for f in (1.0, 1.4)],
                                    'A6000 $0.59 (2.2x)': round(h * 2.2 * .59, 2)}, 'A6000 h=%.2f' % (h * 2.2))
for split in ('dev', 'test'):
    rs = read_jsonl(f'data/v5/cand_{split}.jsonl')
    budget = 2048 - 160 - len(st.encode(SEED)) - 8
    P = np.array([len(st.prompt_ids(p, budget)[0]) for r in rs for p in r['prompts']])
    print(split, json.dumps({'rows': len(rs), 'prompts': len(P), 'prompt_tokens_mean': round(float(P.mean()), 1), 'prompt_p95': int(np.percentile(P, 95)),
                             'max': int(P.max()), 'prefill_tokens_total': int(P.sum())}))
