"""Text-only root-error classifier over a transcript of the student's working.

With teacher transcripts this measures how well diagnosis works once reading is perfect
(a development diagnostic, never a test-time input). With machine transcripts it becomes
the second stage of a read-then-diagnose pipeline.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .baseline import HierarchicalPrior
from .core import label_space, make_folds, map3, read_csv, save_bundle
from .holdout_calibration import calibrated
from .vision import seed_all


def prompt(row, transcript, definitions, examples=''):
    return (f"{definitions}\n\n{examples}Grade: {row.grade}\nTopic: {row.topic_name}\nQuestion: {row.question_text}\n"
            f"Answer key: {row.correct_answer}\nStudent working (line by line):\n{transcript}\n\n"
            "Identify the root major error the rest of the working follows from, or NONE if there is no "
            "major error. ROOT:")


class TextSystem:
    def __init__(self, model_path, classes, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Left truncation drops the start of the code list, never the student's working at the end.
        self.tokenizer.padding_side = self.tokenizer.truncation_side = 'left'
        self.max_length = args.max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={'': 0},
                                                    attn_implementation='sdpa')
        self.model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=.05,
                                                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                                     'gate_proj', 'up_proj', 'down_proj']))
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.head = nn.Linear(base.config.hidden_size, len(classes), device='cuda')
        self.classes = classes

    def hidden(self, texts):
        batch = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                               max_length=self.max_length).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model.get_base_model().model(**batch, use_cache=False).last_hidden_state
        return out[:, -1].float()


def transcripts(frame, source):
    if source == 'gold':
        return frame.transcription_steps.tolist()
    bundle = json.loads(Path(source).read_text())
    return [bundle[t] for t in frame.trace_id]


class ExampleBank:
    """Labelled workings of the same question from the fitting rows, never the row itself."""

    def __init__(self, fit, count, chars=350, seed=2026):
        self.count, self.chars = count, chars
        self.by_question = {q: g.sample(frac=1, random_state=seed) for q, g in fit.groupby('question_id')}

    def __call__(self, row):
        if not self.count:
            return ''
        group = self.by_question.get(row.question_id)
        if group is None:
            return ''
        picked = group[group.trace_id != row.trace_id].head(self.count)
        if picked.empty:
            return ''
        lines = [f"Example (teacher root: {r.root_code}):\n{str(r.transcription_steps)[:self.chars]}"
                 for r in picked.itertuples(index=False)]
        return 'Other students on this question:\n' + '\n'.join(lines) + '\n\n'


@torch.inference_mode()
def score(system, frame, texts, definitions, batch_size, examples=lambda row: ''):
    system.model.eval(); system.head.eval()
    out = []
    for offset in range(0, len(frame), batch_size):
        batch = [prompt(frame.iloc[i], texts[i], definitions, examples(frame.iloc[i]))
                 for i in range(offset, min(offset + batch_size, len(frame)))]
        out.append(system.head(system.hidden(batch)).softmax(-1).cpu().numpy())
    return np.concatenate(out)


def run(args):
    seed_all(args.seed)
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train, taxonomy = read_csv(data / 'train.csv'), read_csv(data / 'taxonomy_v1.csv')
    classes = label_space(taxonomy)
    ci = {c: i for i, c in enumerate(classes)}
    definitions = 'Error codes:\n' + '\n'.join(
        f'{r.id}: {r.name}' + (f' - {r.description}' if args.descriptions else '') for r in taxonomy.itertuples(index=False))
    folds = make_folds(train, seed=args.split_seed)
    fit = train.copy() if args.full_train else train.loc[folds != args.fold].reset_index(drop=True)
    validation = train.loc[folds == args.fold].reset_index(drop=True)
    fit_text = transcripts(fit, args.fit_transcripts)
    examples = ExampleBank(fit, args.examples)
    sources = [] if args.full_train else args.validation_transcripts
    validation_text = {source: transcripts(validation, source) for source in sources}
    system = TextSystem(args.model, classes, args)
    system.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    system.model.enable_input_require_grads()
    params = [p for p in system.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': args.lr},
                                   {'params': system.head.parameters(), 'lr': args.head_lr}], weight_decay=.01)
    updates = args.epochs * math.ceil(len(fit) / args.batch_size)
    step, start, metrics = 0, time.monotonic(), []
    prior = HierarchicalPrior(classes).fit(fit).predict_proba(validation)
    for epoch in range(args.epochs):
        system.model.train(); system.head.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(fit))
        for offset in range(0, len(order), args.batch_size):
            idx = order[offset:offset + args.batch_size]
            texts = [prompt(fit.iloc[i], fit_text[i], definitions, examples(fit.iloc[i])) for i in idx]
            labels = torch.tensor([ci[fit.root_code.iloc[i]] for i in idx], device='cuda')
            loss = F.cross_entropy(system.head(system.hidden(texts)), labels, label_smoothing=.03)
            loss.backward()
            factor = min(1., (step + 1) / max(1, .05 * updates)) * .5 * (1 + math.cos(math.pi * step / max(1, updates)))
            optimizer.param_groups[0]['lr'] = args.lr * factor
            optimizer.param_groups[1]['lr'] = args.head_lr * factor
            torch.nn.utils.clip_grad_norm_(params + list(system.head.parameters()), 1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                print(json.dumps({'stage': 'train', 'epoch': epoch + 1, 'rows': offset + len(idx), 'loss': float(loss),
                                  'seconds': round(time.monotonic() - start)}), flush=True)
        for source, texts in validation_text.items():
            p = score(system, validation, texts, definitions, args.batch_size, examples)
            name = 'gold' if source == 'gold' else Path(source).stem
            save_bundle(out / f'validation_{name}_epoch_{epoch + 1}.npz', validation.trace_id.tolist(), classes, p,
                        kind='text_diagnosis_development', transcripts=str(source))
            metric = {'epoch': epoch + 1, 'rows': len(validation), 'transcripts': name,
                      'map3': map3(validation.root_code, p, classes),
                      'fixed_calibration_map3': map3(validation.root_code, calibrated(p, prior, .25, 2.), classes),
                      'seconds': round(time.monotonic() - start)}
            metrics.append(metric); print(json.dumps(metric), flush=True)
            (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    if args.test_transcripts:
        test = read_csv(data / 'test.csv')
        p = score(system, test, transcripts(test, args.test_transcripts), definitions, args.batch_size, examples)
        save_bundle(out / 'test.npz', test.trace_id.tolist(), classes, p, kind='text_diagnosis_test',
                    transcripts=str(args.test_transcripts))
        print(json.dumps({'stage': 'test_scored', 'rows': len(test)}), flush=True)
    print('K12_TEXT_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--model', required=True); p.add_argument('--output', required=True)
    p.add_argument('--fit-transcripts', default='gold', help="'gold' or a JSON {trace_id: transcript}")
    p.add_argument('--validation-transcripts', nargs='+', default=['gold'])
    p.add_argument('--test-transcripts', help='JSON {trace_id: transcript} for test.csv rows')
    p.add_argument('--full-train', action='store_true')
    p.add_argument('--examples', type=int, default=0, help='same-question labelled examples in the prompt')
    p.add_argument('--descriptions', action='store_true', help='include taxonomy descriptions')
    p.add_argument('--max-length', type=int, default=3072)
    p.add_argument('--seed', type=int, default=2026); p.add_argument('--split-seed', type=int, default=2026)
    p.add_argument('--fold', type=int, default=0); p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--rank', type=int, default=32); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=1e-3); p.add_argument('--batch-size', type=int, default=4)
    run(p.parse_args())
