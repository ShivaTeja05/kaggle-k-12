"""Candidate-aware student: read the working, write a short diagnosis, then pick one lettered candidate code.

Prompts and targets are rendered by scripts/build_candidate_dataset.py, so this file stays generic.
Training adds a soft cross-entropy over the candidate letters at the answer position to the LM loss on
the diagnosis. Prediction generates the diagnosis greedily and reads the letter distribution right
after "ANSWER:", once per candidate order, then maps letters back to the 76-way label space.

Deviation from SPEC.md: each phase has its own 5% warmup + cosine schedule (phase b peaks at
--phase-b-lr-mult x lr). One cosine over the whole epoch left the gold rows at about 2% of the peak lr.
"""
import argparse
from collections import Counter
import contextlib
import glob
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import label_space, map3, read_csv, save_bundle

WORKING, CANDIDATES, ANSWER, SEED = 'Student working (line by line):\n', '\n\nCandidate error codes:', 'ANSWER:', '\nROOT_LINE:'
TINY = '~/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/*'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


class Student:
    def __init__(self, args, adapter=None, rank=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        path = args.model or sorted(glob.glob(os.path.expanduser(TINY)))[-1]
        self.cuda = args.device == 'cuda'
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        tok = self.tokenizer = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.max_length, self.path = args.max_length, path
        self.letter_ids = {}
        for letter in 'ABCDEFGHIJ':
            self.letter(letter)
        quantization = None
        if args.load_4bit:
            if not self.cuda:
                raise ValueError('--load-4bit needs --device cuda')
            from transformers import BitsAndBytesConfig
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs = dict(dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
                      quantization_config=quantization) if self.cuda else dict(dtype=torch.float32)
        base = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        elif rank:
            self.model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=.05,
                                                         target_modules=TARGETS, task_type='CAUSAL_LM'))
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        else:
            self.model = base
        causal = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        self.decoder, self.lm_head = causal.get_decoder(), causal.get_output_embeddings()

    def letter(self, letter):
        if letter not in self.letter_ids:
            ids = self.tokenizer.encode(' ' + letter, add_special_tokens=False)
            assert len(ids) == 1, f'" {letter}" is {len(ids)} tokens'
            self.letter_ids[letter] = ids[0]
        return self.letter_ids[letter]

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt_ids(self, text, budget):
        """Left-truncate only the student's working; the header, candidate block and target always survive."""
        i = text.find(WORKING)
        j = text.find(CANDIDATES, i + 1) if i >= 0 else -1
        if j < 0:
            ids = self.encode(text)
            return ids[max(0, len(ids) - budget):], len(ids) > budget
        head, work, tail = self.encode(text[:i + len(WORKING)]), self.encode(text[i + len(WORKING):j]), self.encode(text[j:])
        if len(head) + len(tail) > budget:
            raise ValueError(f'header + candidate block is {len(head) + len(tail)} tokens, over the budget {budget}; raise --max-length')
        keep = budget - len(head) - len(tail)
        return head + work[max(0, len(work) - keep):] + tail, keep < len(work)

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if self.cuda else contextlib.nullcontext()

    def train_item(self, row):
        target, letters = row['target'], row['letters']
        k = target.rindex(ANSWER) + len(ANSWER)
        answer = target[k:].strip()
        assert answer in letters, (row.get('trace_id'), answer, letters)
        head = self.encode('\n' + target[:k])
        tail = [self.letter(answer), self.tokenizer.eos_token_id]
        prompt, cut = self.prompt_ids(row['prompt'], self.max_length - len(head) - len(tail))
        soft = np.asarray([float(row['soft'].get(x, 0.)) for x in letters])
        assert soft.sum() > 0, row.get('trace_id')
        return {'ids': prompt + head + tail, 'start': len(prompt), 'answer': len(prompt) + len(head) - 1,
                'letters': [self.letter(x) for x in letters], 'soft': soft / soft.sum(),
                'weight': float(row.get('weight', 1.)), 'cut': cut}

    def hidden(self, ids, mask):
        with self.autocast():
            return self.decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

    def loss(self, items, lm_weight, letter_weight):
        """Per row: weight x (lm_weight x target-token CE without the letter + letter_weight x soft letter CE)."""
        pad = self.tokenizer.pad_token_id
        width = max(len(x['ids']) for x in items)
        ids = torch.full((len(items), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        rows, cols = [], []
        for r, x in enumerate(items):
            ids[r, :len(x['ids'])] = torch.tensor(x['ids']); mask[r, :len(x['ids'])] = 1
            c = [c for c in range(x['start'] - 1, len(x['ids']) - 1) if c != x['answer']]
            rows += [r] * len(c); cols += c
        ids, mask = ids.to(self.device), mask.to(self.device)
        hidden = self.hidden(ids, mask)
        rows, cols = torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)
        token_ce = hidden.new_zeros(len(items), dtype=torch.float32)
        for s in range(0, len(rows), 1024):
            r, c = rows[s:s + 1024], cols[s:s + 1024]
            ce = F.cross_entropy(self.lm_head(hidden[r, c]).float(), ids[r, c + 1], reduction='none')
            token_ce = token_ce.index_add(0, r, ce)
        lm = token_ce / torch.bincount(rows, minlength=len(items)).clamp(min=1)
        answer = self.lm_head(hidden[torch.arange(len(items), device=self.device),
                                     torch.tensor([x['answer'] for x in items], device=self.device)]).float()
        letter = torch.stack([-(torch.tensor(x['soft'], dtype=torch.float32, device=self.device)
                                * answer[r, x['letters']].log_softmax(-1)).sum() for r, x in enumerate(items)])
        weight = torch.tensor([x['weight'] for x in items], device=self.device)
        return (weight * (lm_weight * lm + letter_weight * letter)).mean(), lm.mean(), letter.mean()

    @torch.inference_mode()
    def letter_probs(self, sequences, letters):
        pad = self.tokenizer.pad_token_id
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, s in enumerate(sequences):
            ids[r, :len(s)] = torch.tensor(s); mask[r, :len(s)] = 1
        hidden = self.hidden(ids.to(self.device), mask.to(self.device))
        ends = torch.tensor([len(s) - 1 for s in sequences], device=self.device)
        last = self.lm_head(hidden[torch.arange(len(sequences), device=self.device), ends]).float()
        return [last[r, [self.letter(x) for x in l]].softmax(-1).cpu().numpy() for r, l in enumerate(letters)]

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens):
        tok, pad = self.tokenizer, self.tokenizer.pad_token_id
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, s in enumerate(prompts):  # left padding for batched generation
            ids[r, width - len(s):] = torch.tensor(s); mask[r, width - len(s):] = 1
        with self.autocast():
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                      max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                                      top_k=None, stop_strings=[ANSWER], tokenizer=tok, pad_token_id=pad,
                                      eos_token_id=tok.eos_token_id)
        return [tok.decode(o[width:], skip_special_tokens=True) for o in out]


def train(args):
    seed_all(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    needed = [args.train, Path(args.data) / 'taxonomy_v1.csv', *(args.predict or []), *([args.interim_predict] if args.interim_predict else [])]
    if missing := [str(x) for x in needed if not Path(x).exists()]:
        raise FileNotFoundError(f'missing inputs: {missing}')
    rows = read_jsonl(args.train, args.limit)
    student = Student(args, rank=args.rank)
    model = student.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    items = [student.train_item(r) for r in rows]
    phase = [r.get('phase', 'a') for r in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)

    def plan(epoch):  # phase "a" rows first, then phase "b", each shuffled; accumulation windows never mix phases
        windows = []
        for name in ('a', 'b'):
            ix = np.flatnonzero(np.asarray(phase) == name)
            ix = ix[np.random.default_rng(args.seed + 1000 * epoch + ord(name)).permutation(len(ix))]
            batches = [ix[s:s + args.batch_size] for s in range(0, len(ix), args.batch_size)]
            windows += [(name, batches[s:s + args.grad_accum]) for s in range(0, len(batches), args.grad_accum)]
        return windows
    total = Counter(name for name, _ in plan(0))
    total = {name: args.epochs * total[name] for name in 'ab'}
    updates = sum(total.values())

    def lr_at(name, k):  # own 5% warmup + cosine per phase; phase b peaks at phase_b_lr_mult x lr
        n, peak = max(1, total[name]), args.lr * (args.phase_b_lr_mult if name == 'b' else 1.)
        return peak * min(1., (k + 1) / max(1, .05 * n)) * .5 * (1 + math.cos(math.pi * k / n))
    interim = max(1, int(args.interim_at * updates)) if args.interim_predict else -1
    print(json.dumps({'stage': 'train_start', 'rows': len(rows), 'phase_a': phase.count('a'), 'phase_b': phase.count('b'),
                      'updates': updates, 'updates_a': total['a'], 'updates_b': total['b'], 'interim_step': interim,
                      'truncated': sum(x['cut'] for x in items), 'max_tokens': max(len(x['ids']) for x in items),
                      'model': student.path, 'trainable': sum(p.numel() for p in params)}), flush=True)
    step, seen, start, log, history, done = 0, 0, time.monotonic(), [], [], Counter()
    model.train()
    for epoch in range(args.epochs):
        for name, window in plan(epoch):
            for idx in window:
                loss, lm, letter = student.loss([items[i] for i in idx], args.lm_weight, args.letter_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'non-finite loss at step {step}')
                (loss / len(window)).backward()
                seen += len(idx); log.append((loss.item(), lm.item(), letter.item()))
            lr = lr_at(name, done[name]); done[name] += 1
            for group in optimizer.param_groups:
                group['lr'] = lr
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % args.log_every == 0 or step == updates:
                seconds = time.monotonic() - start
                m = np.mean(log, axis=0)
                record = {'stage': 'train', 'epoch': epoch + 1, 'step': step, 'phase': name, 'lr': lr,
                          'loss': round(m[0], 5), 'lm_loss': round(m[1], 5), 'letter_loss': round(m[2], 5), 'rows': seen,
                          'seconds': round(seconds, 1), 'rows_per_s': round(seen / max(seconds, 1e-9), 3)}
                history.append(record); log = []
                print(json.dumps(record), flush=True)
            if args.save_every and step % args.save_every == 0 and step < updates:
                model.save_pretrained(out / 'checkpoint')
                (out / 'checkpoint' / 'step.json').write_text(json.dumps({'step': step, 'of': updates, 'rows': seen}))
            if step == interim:
                predict_rows(student, read_jsonl(args.interim_predict, args.interim_limit), args, out / f'interim_step{step}')
                model.train()
    if args.save_adapter:
        model.save_pretrained(out / 'adapter')
    (out / 'train_metrics.json').write_text(json.dumps({'args': vars(args), 'history': history}, indent=2))
    for path in args.predict or []:
        predict_rows(student, read_jsonl(path, args.limit), args, out / f'predict_{Path(path).stem}')
    print('K12_CANDIDATE_TRAIN_COMPLETE', flush=True)


def predict_rows(student, rows, args, out):
    out.mkdir(parents=True, exist_ok=True)
    model, tok = student.model, student.tokenizer
    model.eval()
    classes = label_space(read_csv(Path(args.data) / 'taxonomy_v1.csv'))
    ci = {c: i for i, c in enumerate(classes)}
    seed = student.encode(SEED)
    budget = args.max_length - args.max_new_tokens - len(seed) - 8
    jobs = []
    for r, row in enumerate(rows):
        for o in range(min(args.orders, len(row['prompts']))):
            ids, cut = student.prompt_ids(row['prompts'][o], budget)
            jobs.append({'row': r, 'order': o, 'ids': ids, 'cut': cut, 'codes': row['codes_per_prompt'][o],
                         'letters': row.get('letters_per_prompt', [row['letters']] * 2)[o]})
            assert len(jobs[-1]['codes']) == len(jobs[-1]['letters']) and all(c in ci for c in jobs[-1]['codes'])
    jobs.sort(key=lambda j: -len(j['ids']))
    start = time.monotonic()
    for s in range(0, len(jobs), args.gen_batch_size):
        batch = jobs[s:s + args.gen_batch_size]
        texts = student.generate([j['ids'] + seed for j in batch], args.max_new_tokens)
        sequences = []
        for j, text in zip(batch, texts):
            text = SEED.lstrip('\n') + text
            j['parsed'] = ANSWER in text
            j['text'] = text[:text.index(ANSWER) + len(ANSWER)] if j['parsed'] else text.rstrip() + '\n' + ANSWER
            sequences.append(j['ids'] + student.encode('\n' + j['text']))
        for j, p in zip(batch, student.letter_probs(sequences, [j['letters'] for j in batch])):
            j['probs'] = p
        done = s + len(batch)
        if (s // args.gen_batch_size) % args.log_every == 0 or done == len(jobs):
            seconds = time.monotonic() - start
            print(json.dumps({'stage': 'predict', 'prompts': done, 'of': len(jobs),
                              'parse_rate': round(np.mean([j['parsed'] for j in jobs[:done]]), 4),
                              'seconds': round(seconds, 1), 'prompts_per_s': round(done / max(seconds, 1e-9), 3)}), flush=True)
    seconds = time.monotonic() - start
    per_order = np.zeros((args.orders, len(rows), len(classes)))
    p = np.zeros((len(rows), len(classes)))
    count, candidate = np.zeros(len(rows)), np.zeros((len(rows), len(classes)), dtype=bool)
    jobs.sort(key=lambda j: (j['row'], j['order']))
    with open(out / 'generations.jsonl', 'w') as f:
        for j in jobs:
            cols = [ci[c] for c in j['codes']]
            np.add.at(per_order[j['order'], j['row']], cols, j['probs'])
            np.add.at(p[j['row']], cols, j['probs'])
            count[j['row']] += 1; candidate[j['row'], cols] = True
            f.write(json.dumps({'trace_id': rows[j['row']]['trace_id'], 'order': j['order'] + 1, 'text': j['text'],
                                'parsed': j['parsed'], 'truncated': j['cut'],
                                'letter_probs': {l: round(float(x), 6) for l, x in zip(j['letters'], j['probs'])}},
                               ensure_ascii=False) + '\n')
    p = np.where(candidate, p / count[:, None], 1e-4)
    p /= p.sum(1, keepdims=True)
    ids = [r['trace_id'] for r in rows]
    metrics = {'rows': len(rows), 'prompts': len(jobs), 'orders': args.orders, 'seconds': round(seconds, 1),
               'rows_per_s': round(len(rows) / max(seconds, 1e-9), 3),
               'parse_rate': float(np.mean([j['parsed'] for j in jobs])),
               'parse_rate_per_order': [float(np.mean([j['parsed'] for j in jobs if j['order'] == o]))
                                        for o in range(args.orders)],
               'truncated': int(sum(j['cut'] for j in jobs)), 'model': student.path, 'adapter': getattr(args, 'adapter', None) or (args.save_adapter and str(Path(args.output) / 'adapter')) or None}
    if all('truth' in r for r in rows):  # development reporting only; never an input
        truth = [r['truth'] for r in rows]
        metrics['map3'] = map3(truth, p, classes)
        metrics['map3_per_order'] = [map3(truth, np.where(candidate, q, 1e-4), classes) for q in per_order]
        metrics['recall_candidates'] = float(np.mean([candidate[i, ci[t]] for i, t in enumerate(truth)]))
    save_bundle(out / 'predictions.npz', ids, classes, p, kind='candidate_student', **{k: v for k, v in metrics.items()
                                                                                       if not isinstance(v, list)})
    (out / 'predict_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({'stage': 'predict_done', **metrics}), flush=True)
    return p


def predict(args):
    seed_all(args.seed)
    student = Student(args, adapter=args.adapter)
    predict_rows(student, read_jsonl(args.input, args.limit), args, Path(args.output))
    print('K12_CANDIDATE_PREDICT_COMPLETE', flush=True)


def main(argv=None):
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--model', help='base model path; default on cpu: cached tiny Qwen2 for smoke tests')
    shared.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    shared.add_argument('--load-4bit', action='store_true')
    shared.add_argument('--output', required=True)
    shared.add_argument('--data', default='data/raw', help='directory with taxonomy_v1.csv (label space)')
    shared.add_argument('--max-length', type=int, default=2048)
    shared.add_argument('--max-new-tokens', type=int, default=224)
    shared.add_argument('--orders', type=int, choices=[1, 2], default=2)
    shared.add_argument('--gen-batch-size', type=int, default=16)
    shared.add_argument('--log-every', type=int, default=25)
    shared.add_argument('--limit', type=int, help='smoke test: first N rows of each input')
    shared.add_argument('--seed', type=int, default=2026)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', parents=[shared])
    t.add_argument('--train', required=True, help='cand_train.jsonl')
    t.add_argument('--predict', nargs='*', help='dev/test jsonl files to score with the trained adapter')
    t.add_argument('--epochs', type=int, default=1); t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--batch-size', type=int, default=4); t.add_argument('--grad-accum', type=int, default=1)
    t.add_argument('--rank', type=int, default=32); t.add_argument('--phase-b-lr-mult', type=float, default=.3)
    t.add_argument('--lm-weight', type=float, default=1.); t.add_argument('--letter-weight', type=float, default=1.)
    t.add_argument('--save-adapter', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--save-every', type=int, default=0, help='also save OUT/checkpoint every N updates')
    t.add_argument('--interim-predict', help='dev jsonl to score mid-training (PLAN step 4 interim abort)')
    t.add_argument('--interim-at', type=float, default=.5, help='fraction of all updates at which to run --interim-predict')
    t.add_argument('--interim-limit', type=int, default=100, help='first N rows of --interim-predict')
    q = sub.add_parser('predict', parents=[shared])
    q.add_argument('--input', required=True, help='cand_dev.jsonl or cand_test.jsonl')
    q.add_argument('--adapter', help='LoRA adapter directory from train')
    args = p.parse_args(argv)
    train(args) if args.command == 'train' else predict(args)


if __name__ == '__main__':
    main()
