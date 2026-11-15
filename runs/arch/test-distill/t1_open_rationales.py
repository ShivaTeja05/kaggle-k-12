"""T1. Does the rationale route survive when the rationale is written by an OPEN model instead of Claude?
(a) reproduce d4 with nested cross-fit (desc weight tuned inside folds too);
(b) control: Claude dev reasoning cut to its last 300 chars (the length of the stored open-model tails);
(c) the real test: last-300-char rationale tails that Qwen3-32B / Qwen3-235B wrote on the 301 dev rows from machine
    transcripts (runs/pod/probe235/*/responses.jsonl, zero-shot and few-shot, thinking mode), decoded by the same
    Claude-trained decoder and fused with the blend;
(d) the same open models' own parsed codes fused with the blend;
(e) G1(a)-style agreement with Claude's dev first code for existing open students."""
import json
import numpy as np
from common import *

res = {'blend': {'map3': round(float(sb.mean()), 4), 'bins': pb(sb), 'top1_agree_claude': round(float((blend.argmax(1) == claude_first).mean()), 4)}}
full = cached_decoder('full', U_txt, U_y)
tail_tr = [cl((U[t].get('reasoning') or '')[-300:]) for t in U_ids]
tail = cached_decoder('tail300', tail_tr, U_y)

r, cf_full = fuse_crossfit(full.lr_proba(dev_txt), desc_sim(dev_txt)); res['a_claude_full_reasoning'] = r
ct = [cl((D.get(t, {}).get('reasoning') or '')[-300:]) for t in ids]
res['b_claude_tail300_taildecoder'], _ = fuse_crossfit(tail.lr_proba(ct), desc_sim(ct))
res['b_claude_tail300_fulldecoder'], _ = fuse_crossfit(full.lr_proba(ct), desc_sim(ct))
for m in ['zeroshot32', 'fewshot32', 'zeroshot235', 'fewshot235']:
    R = {json.loads(l)['trace_id']: json.loads(l) for l in open(ROOT/f'runs/pod/probe235/{m}/responses.jsonl')}
    tx = [cl(R.get(t, {}).get('tail') or '') for t in ids]
    r, cf = fuse_crossfit(tail.lr_proba(tx), desc_sim(tx)); res[f'c_{m}_tail_taildecoder'] = r
    r2, _ = fuse_crossfit(full.lr_proba(tx), desc_sim(tx)); res[f'c_{m}_tail_fulldecoder'] = r2
    own = aligned_bundle(ROOT/f'runs/pod/probe235/{m}/validation.npz', ids, classes)
    r3, _ = fuse_crossfit(own, np.zeros_like(own), dgrid=[0]); r3['own_top1_agree_claude'] = round(float((own.argmax(1) == claude_first).mean()), 4)
    res[f'd_{m}_own_codes'] = r3
for name, b in [('scale_100_8b_tx', 'runs/pod/scaling/scale_100/validation_tx_dev_epoch_1.npz'), ('q8_e3_tx', 'runs/pod/tx/text_tx_dev_e3.npz'),
                ('q32_tx', 'runs/pod/text32/validation_tx_dev_epoch_1.npz'), ('vl_distill', 'runs/pod/dev-distill/validation_epoch_2.npz')]:
    p = aligned_bundle(ROOT/b, ids, classes)
    res.setdefault('e_open_students', {})[name] = {'map3': round(float(rs(p).mean()), 4), 'top1_agree_claude': round(float((p.argmax(1) == claude_first).mean()), 4),
        'top1_agree_claude_unseen106': round(float((p.argmax(1) == claude_first)[unseen].mean()), 4), 'unseen106_map3': round(float(rs(p)[unseen].mean()), 4)}
res['claude_top1_correct'] = round(float((claude_first == yi).mean()), 4)
np.save(OUT/'cf_claude_full.npy', cf_full)
print(json.dumps(res, indent=1)); json.dump(res, open(OUT/'t1_open_rationales.json', 'w'), indent=1)
