"""Realistic (offline) version of s6: mechanism text written by OPEN models already run on dev
(probe235 responses: 235B/32B zero/few-shot; only the final ~500-char 'tail' was saved), code ids and
code names stripped, mapped by the same question-blind TF-IDF mapper, fused into the blend (cross-fitted)."""
import json, re, numpy as np
src = open('runs/arch/design-structured/s6_fuse_mech.py').read()
exec(src.split("Xd = [strip((cl.get(t)")[0])
exec("L = lambda p: aligned_bundle(p, dev.trace_id" + src.split("L = lambda p: aligned_bundle(p, dev.trace_id", 1)[1].split("\nout = {}")[0])
out = {}
for name in ['zeroshot32', 'fewshot32', 'zeroshot235', 'fewshot235']:
    rs = {json.loads(l)['trace_id']: json.loads(l) for l in open(f'runs/pod/probe235/{name}/responses.jsonl')}
    Xd = [strip(re.sub(r'CODES:.*', ' ', (rs.get(t) or {}).get('tail', ''))) for t in dev.trace_id]
    P = run(Xtr, Ytr, W); sc, ins = crossfit(P); d = sc - ref
    out[name] = {'mapper_err_stats': stats(P), 'crossfit': round(float(sc.mean()), 4), 'delta': round(float(d.mean()), 4), 'ci': boot(d),
                 'best_in_sample': max(ins.items(), key=lambda kv: kv[1])}
    print(name, out[name])
json.dump(out, open('runs/arch/design-structured/s7_openmodel_mech.json', 'w'), indent=1)
