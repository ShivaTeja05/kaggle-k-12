"""Two checks on the trained 2B adapter (Mac rehearsal):
1. the state the classification layer reads at the last prompt token is the same whether or not the target text follows
   (training reads it with the target appended, `head` without);
2. a few written answers generated greedily: is the JSON valid, do the codes exist, does first_code match the truth and the head?

Usage (repo root, PYTHONPATH=src): python runs/mac_2b/check_state_and_generate.py <model> <adapter> <device> <n_generate> <out.json>
"""
import argparse, json, sys, time
import torch
import k12.q38_finetune as Q
from k12.core import read_csv

model_dir, adapter, device, n_gen, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
a = argparse.Namespace(model=model_dir, device=device, dtype=None, image_root=None, gpu_memory='13GiB')
processor, base = Q.load(a)
from peft import PeftModel
Q.peft_ready(); model = PeftModel.from_pretrained(base, adapter); model.eval()
saved = torch.load(adapter + '/head.pt', map_location=Q.entry(base), weights_only=True)
head = Q.make_head(saved['width'], len(saved['classes']), Q.entry(base)); head.load_state_dict(saved['state']); head.eval()
classes = saved['classes']; report = {'state_check': [], 'generated': []}

rows, _ = Q.read_rows('data/v11/ft_train.jsonl', 3)
for r in rows:
    with torch.inference_mode():
        i1, at1 = Q.encode(processor, r); keys = Q.accepted_keys(base, i1)
        _, s1, _ = Q.forward(model, i1, keys, a); v1 = s1[0, at1 - 1].float().cpu()
        i2, at2 = Q.encode(processor, r, 900)
        _, s2, _ = Q.forward(model, i2, keys, a); v2 = s2[0, at2 - 1].float().cpu()
        p1 = torch.softmax(head(v1.unsqueeze(0).to(Q.entry(base)))[:, :-1], 1)[0].cpu(); p2 = torch.softmax(head(v2.unsqueeze(0).to(Q.entry(base)))[:, :-1], 1)[0].cpu()
    report['state_check'].append({'trace_id': r['trace_id'], 'prompt_tokens': at1, 'same_position': at1 == at2, 'tokens_with_target': int(i2['input_ids'].shape[1]),
                                  'max_abs_state_difference': float((v1 - v2).abs().max()), 'state_scale': float(v1.abs().mean()),
                                  'max_abs_probability_difference': float((p1 - p2).abs().max())})
    print(json.dumps(report['state_check'][-1]), flush=True)

train = read_csv('data/raw/train.csv'); truth = dict(zip(train.trace_id, train.root_code))
dev, _ = Q.read_rows('data/v11/ft_dev.jsonl', n_gen)
for r in dev:
    inputs, at = Q.encode(processor, r); keys = Q.accepted_keys(base, inputs)
    batch = {k: inputs[k].to(Q.entry(base)) for k in keys}; t = time.monotonic()
    with torch.inference_mode():
        got = model.generate(**batch, max_new_tokens=450, do_sample=False)
        _, s, _ = Q.forward(model, inputs, keys, a); p = torch.softmax(head(s[0, at - 1].float().unsqueeze(0))[:, :-1], 1)[0].cpu()
    text = processor.tokenizer.decode(got[0, at:], skip_special_tokens=True); item = {'trace_id': r['trace_id'], 'truth': truth[r['trace_id']],
            'head_top1': classes[int(p.argmax())], 'new_tokens': int(got.shape[1] - at), 'seconds': round(time.monotonic() - t, 1), 'text': text}
    try:
        j = json.loads(text); item['json_valid'] = True; item['keys'] = list(j)
        item['codes'] = [j.get('first_code'), j.get('second_code'), j.get('third_code')]; item['codes_in_taxonomy'] = all(c in classes for c in item['codes'])
    except Exception as exc:
        item['json_valid'] = False; item['error'] = str(exc)[:120]
    report['generated'].append(item); print(json.dumps(item)[:1500], flush=True)
open(out, 'w').write(json.dumps(report, indent=1) + '\n')
