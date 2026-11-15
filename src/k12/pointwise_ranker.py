"""Pointwise ranker: does THIS candidate code explain THIS working? Yes or No.

Each (working, candidate) pair is one prompt. The candidate's code, name and description are text in the
prompt, so the same weights judge a code on every question. The score is logit(" Yes") - logit(" No") at
the last position, read from the frozen language-model head. A row's candidates form one group and the loss
is a soft cross-entropy over the group's scores (one right candidate against several wrong ones from the
same row), following the first-place Eedi 2024 ranker.

    build  (CPU)  listwise candidate files (scripts/build_candidate_dataset.py format) -> pointwise jsonl
    train  (GPU)  LoRA fine-tune, then score dev/test files into probability bundles
    predict       score a file with a saved adapter
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np

SPLIT = '\n\nCandidate error codes:\n'
QUESTION = ('\nIs this candidate the root major error that the rest of the working follows from? '
            'Answer Yes or No.\nAnswer:')
TINY = '~/.cache/huggingface/hub/models--trl-internal-testing--tiny-Qwen2ForCausalLM-2.5/snapshots/*'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
LINE = re.compile(r'^([A-J])\. (.+)$')


def read_jsonl(path, limit=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------- build (CPU)
def split_listwise(prompt, codes):
    """A listwise prompt -> (context ending with the working, [candidate text per code])."""
    context, rest = prompt.split(SPLIT, 1)
    texts = [m.group(2) for m in map(LINE.match, rest.split('\n')) if m]
    if len(texts) != len(codes) or any(not t.startswith(c) for t, c in zip(texts, codes)):
        raise ValueError(f'candidate lines do not match codes: {codes} vs {[t[:12] for t in texts]}')
    return context, texts


def build(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ('train', 'dev', 'test'):
        rows, made = read_jsonl(Path(args.source) / f'cand_{name}.jsonl'), []
        for r in rows:
            if name == 'train':
                if r.get('kind') == 'contrast':
                    continue  # 2-3 candidate drill rows written for the listwise student
                context, texts = split_listwise(r['prompt'], r['codes'])
                letter_code = dict(zip(r['letters'], r['codes']))
                soft = {letter_code[letter]: share for letter, share in r['soft'].items() if share > 0}
                made.append(dict(trace_id=r['trace_id'], source=r['source'], weight=r['weight'], context=context,
                                 candidates=[dict(code=c, text=t) for c, t in zip(r['codes'], texts)], soft=soft))
            else:
                context, texts = split_listwise(r['prompts'][0], r['codes_per_prompt'][0])
                row = dict(trace_id=r['trace_id'], context=context,
                           candidates=[dict(code=c, text=t) for c, t in zip(r['codes_per_prompt'][0], texts)])
                if 'truth' in r:
                    row['truth'] = r['truth']
                made.append(row)
        if name == 'train':  # the listwise file repeats each teacher row; a pointwise group needs it once
            seen, unique = set(), []
            for row in made:
                key = (row['trace_id'].split(':')[0], row['source'], tuple(c['code'] for c in row['candidates']))
                if key not in seen:
                    seen.add(key); unique.append(row)
            made = unique
        (out / f'pw_{name}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in made) + '\n')
        report[name] = dict(rows=len(made), pairs=sum(len(r['candidates']) for r in made))
        if name == 'train':
            report[name]['by_source'] = {s: sum(r['source'] == s for r in made) for s in sorted({r['source'] for r in made})}
            report[name]['truth_in_candidates'] = float(np.mean([max(r['soft'], key=r['soft'].get) in
                                                                 {c['code'] for c in r['candidates']} for r in made]))
        elif made and 'truth' in made[0]:
            report[name]['truth_in_candidates'] = float(np.mean([r['truth'] in {c['code'] for c in r['candidates']} for r in made]))
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- model
def pair_text(context, candidate_text):
    return f'{context}\n\nCandidate diagnosis: {candidate_text}{QUESTION}'


class Ranker:
    def __init__(self, args, adapter=None, rank=None, trainable=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        self.torch = torch
        path = args.model or sorted(glob.glob(os.path.expanduser(TINY)))[-1]
        self.cuda = args.device == 'cuda'
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        tok = self.tokenizer = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.max_length, self.path = args.max_length, path
        yes, no = tok.encode(' Yes', add_special_tokens=False), tok.encode(' No', add_special_tokens=False)
        assert len(yes) == 1 and len(no) == 1, f'" Yes"/" No" are {len(yes)}/{len(no)} tokens'
        self.yes, self.no = yes[0], no[0]
        quantization = None
        if args.load_4bit:
            from transformers import BitsAndBytesConfig
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs = dict(dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa',
                      quantization_config=quantization) if self.cuda else dict(dtype=torch.float32)
        base = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if adapter:
            self.model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
            if trainable:  # continue a saved ranker: same float32 trainable weights as a fresh adapter
                for p in self.model.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
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
        self.tail = tok.encode(QUESTION, add_special_tokens=False)

    def encode_pair(self, context, candidate_text):
        """Left-truncate only the context (question + working); the candidate and the question always survive."""
        tok = self.tokenizer
        end = tok.encode(f'\n\nCandidate diagnosis: {candidate_text}', add_special_tokens=False) + self.tail
        head = tok.encode(context, add_special_tokens=False)
        keep = self.max_length - len(end)
        if keep < 64:
            raise ValueError('candidate text leaves no room for the working; raise --max-length')
        return head[max(0, len(head) - keep):] + end, len(head) > keep

    def scores(self, sequences):
        """logit(Yes) - logit(No) at the last real position of each sequence."""
        torch, pad = self.torch, self.tokenizer.pad_token_id
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, s in enumerate(sequences):  # right padding; the last real position is len(s) - 1
            ids[r, :len(s)] = torch.tensor(s); mask[r, :len(s)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.cuda else torch.autocast('cpu', enabled=False)
        with context:
            hidden = self.decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = hidden[torch.arange(len(sequences), device=self.device),
                      torch.tensor([len(s) - 1 for s in sequences], device=self.device)]
        weight = self.lm_head.weight[[self.yes, self.no]]  # only two rows of the vocabulary matrix are needed
        logits = last.to(weight.dtype) @ weight.T
        return (logits[:, 0] - logits[:, 1]).float()


# ---------------------------------------------------------------- train
def make_group(row, size, rng):
    """All candidates with a positive share, plus wrong candidates of the same row up to `size`."""
    positive = [c for c in row['candidates'] if row['soft'].get(c['code'], 0) > 0]
    negative = [c for c in row['candidates'] if row['soft'].get(c['code'], 0) <= 0]
    rng.shuffle(negative)
    group = positive + negative[:max(0, size - len(positive))]
    rng.shuffle(group)
    target = np.asarray([row['soft'].get(c['code'], 0.) for c in group], dtype=np.float64)
    return group, target / target.sum()


def select_rows(rows, claude_rows, seed):
    """All teacher rows, plus a seeded sample of Claude-labelled rows."""
    gold = [r for r in rows if r['source'] != 'claude']
    claude = [r for r in rows if r['source'] == 'claude']
    random.Random(seed).shuffle(claude)
    return gold + claude[:claude_rows if claude_rows >= 0 else len(claude)]


def train(args):
    import torch
    from torch.nn import functional as F
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(read_jsonl(args.train), args.claude_rows, args.seed)
    rows = [r for r in rows if sum(r['soft'].get(c['code'], 0) for c in r['candidates']) > 0]
    if args.limit:
        rows = rows[:args.limit]
    ranker = Ranker(args, adapter=args.init_adapter, trainable=True) if args.init_adapter else Ranker(args, rank=args.rank)
    model = ranker.model
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=.01)
    updates = args.epochs * math.ceil(len(rows) / (args.rows_per_step * args.grad_accum))
    print(json.dumps({'stage': 'train_start', 'rows': len(rows), 'teacher_rows': sum(r['source'] != 'claude' for r in rows),
                      'group': args.group, 'updates': updates, 'model': ranker.path, 'init_adapter': args.init_adapter,
                      'trainable': sum(p.numel() for p in params)}), flush=True)
    step, seen, tokens, cut, start, log = 0, 0, 0, 0, time.monotonic(), []
    model.train()
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(rows))); rng.shuffle(order)
        batches = [order[s:s + args.rows_per_step] for s in range(0, len(order), args.rows_per_step)]
        for b, batch in enumerate(batches):
            sequences, spans, targets, weights = [], [], [], []
            for i in batch:
                group, target = make_group(rows[i], args.group, rng)
                first = len(sequences)
                for c in group:
                    ids, was_cut = ranker.encode_pair(rows[i]['context'], c['text'])
                    sequences.append(ids); cut += was_cut
                spans.append((first, len(sequences))); targets.append(target); weights.append(float(rows[i].get('weight', 1.)))
            scores = ranker.scores(sequences)
            loss = scores.new_zeros(())
            for (a, z), target, weight in zip(spans, targets, weights):
                logp = F.log_softmax(scores[a:z], dim=0)
                loss = loss - weight * (torch.tensor(target, dtype=torch.float32, device=scores.device) * logp).sum()
            loss = loss / len(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {step}')
            (loss / args.grad_accum).backward()
            seen += len(batch); tokens += sum(map(len, sequences)); log.append(loss.item())
            if (b + 1) % args.grad_accum == 0 or b + 1 == len(batches):
                lr = args.lr * min(1., (step + 1) / max(1, .05 * updates)) * .5 * (1 + math.cos(math.pi * step / max(1, updates)))
                for g in optimizer.param_groups:
                    g['lr'] = lr
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                if step % args.log_every == 0 or step == updates:
                    seconds = time.monotonic() - start
                    print(json.dumps({'stage': 'train', 'epoch': epoch + 1, 'step': step, 'of': updates, 'rows': seen,
                                      'loss': round(float(np.mean(log)), 4), 'lr': lr, 'seconds': round(seconds, 1),
                                      'rows_per_s': round(seen / max(seconds, 1e-9), 3),
                                      'tokens_per_s': round(tokens / max(seconds, 1e-9)), 'truncated': cut}), flush=True)
                    log = []
    if args.save_adapter:
        model.save_pretrained(out / 'adapter')
    for path in args.predict or []:
        predict_rows(ranker, read_jsonl(path, args.limit), args, out / f'predict_{Path(path).stem}')
    print('K12_POINTWISE_TRAIN_COMPLETE', flush=True)


# ---------------------------------------------------------------- predict
def predict_rows(ranker, rows, args, out):
    import torch
    from .core import label_space, map3, read_csv, save_bundle
    out.mkdir(parents=True, exist_ok=True)
    ranker.model.eval()
    classes = label_space(read_csv(Path(args.data) / 'taxonomy_v1.csv'))
    ci = {c: i for i, c in enumerate(classes)}
    jobs = []
    for r, row in enumerate(rows):
        for c in row['candidates']:
            ids, _ = ranker.encode_pair(row['context'], c['text'])
            jobs.append((len(ids), r, ci[c['code']], ids))
    jobs.sort(key=lambda j: -j[0])  # longest first: similar lengths share a batch and any memory error comes early
    raw = np.full((len(rows), len(classes)), -np.inf)
    start = time.monotonic()
    with torch.inference_mode():
        for s in range(0, len(jobs), args.score_batch):
            batch = jobs[s:s + args.score_batch]
            values = ranker.scores([j[3] for j in batch]).cpu().numpy()
            for (_, r, col, _), v in zip(batch, values):
                raw[r, col] = v
            if (s // args.score_batch) % args.log_every == 0:
                print(json.dumps({'stage': 'predict', 'pairs': min(len(jobs), s + args.score_batch), 'of': len(jobs),
                                  'seconds': round(time.monotonic() - start, 1)}), flush=True)
    seconds = time.monotonic() - start
    p = np.full(raw.shape, 1e-4)
    for r in range(len(rows)):  # softmax over the row's candidates; every other code keeps a small floor
        m = np.isfinite(raw[r])
        e = np.exp((raw[r, m] - raw[r, m].max()) / args.temperature)
        p[r, m] = e / e.sum()
    p /= p.sum(1, keepdims=True)
    metrics = {'rows': len(rows), 'pairs': len(jobs), 'seconds': round(seconds, 1),
               'pairs_per_s': round(len(jobs) / max(seconds, 1e-9), 2), 'model': ranker.path}
    if rows and all('truth' in r for r in rows):  # development reporting only; never an input
        truth = [r['truth'] for r in rows]
        metrics['map3'] = map3(truth, p, classes)
        metrics['top1'] = float(np.mean([classes[i] == t for i, t in zip(p.argmax(1), truth)]))
        metrics['truth_in_candidates'] = float(np.mean([np.isfinite(raw[i, ci[t]]) for i, t in enumerate(truth)]))
    np.save(out / 'raw_scores.npy', raw)
    save_bundle(out / 'predictions.npz', [r['trace_id'] for r in rows], classes, p, kind='pointwise_ranker',
                **{k: v for k, v in metrics.items() if not isinstance(v, str)})
    (out / 'predict_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({'stage': 'predict_done', **metrics}), flush=True)
    return p


def predict(args):
    ranker = Ranker(args, adapter=args.adapter)
    predict_rows(ranker, read_jsonl(args.input, args.limit), args, Path(args.output))
    print('K12_POINTWISE_PREDICT_COMPLETE', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--source', default='runs/deep/stageB/data', help='folder with cand_{train,dev,test}.jsonl')
    b.add_argument('--out', default='data/v7')
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--model', help='base model path; default on cpu: the cached tiny Qwen2 for smoke tests')
    shared.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    shared.add_argument('--load-4bit', action='store_true')
    shared.add_argument('--output', required=True)
    shared.add_argument('--data', default='data/raw', help='folder with taxonomy_v1.csv')
    shared.add_argument('--max-length', type=int, default=1024)
    shared.add_argument('--score-batch', type=int, default=32)
    shared.add_argument('--temperature', type=float, default=1.)
    shared.add_argument('--log-every', type=int, default=25)
    shared.add_argument('--limit', type=int, help='smoke test: first N rows of each input')
    shared.add_argument('--seed', type=int, default=2026)
    t = sub.add_parser('train', parents=[shared])
    t.add_argument('--train', required=True, help='pw_train.jsonl')
    t.add_argument('--predict', nargs='*', help='pointwise files to score after training')
    t.add_argument('--claude-rows', type=int, default=2800, help='Claude-labelled rows to sample (-1 = all)')
    t.add_argument('--group', type=int, default=6, help='candidates per row in one training group')
    t.add_argument('--rows-per-step', type=int, default=2); t.add_argument('--grad-accum', type=int, default=2)
    t.add_argument('--epochs', type=int, default=1); t.add_argument('--lr', type=float, default=1e-4)
    t.add_argument('--rank', type=int, default=32)
    t.add_argument('--init-adapter', help='continue training this saved adapter instead of starting a new one')
    t.add_argument('--save-adapter', action=argparse.BooleanOptionalAction, default=True)
    q = sub.add_parser('predict', parents=[shared])
    q.add_argument('--input', required=True); q.add_argument('--adapter', required=True)
    args = p.parse_args(argv)
    {'build': build, 'train': train, 'predict': predict}[args.command](args)


if __name__ == '__main__':
    main()
