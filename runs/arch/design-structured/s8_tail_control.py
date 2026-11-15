"""Control for s7: the probe responses kept only a ~500-char tail. Truncate Claude's dev reasoning to its
last 500 chars (and first 300) and redo s6. If Claude's tail keeps most of the +0.061, the open models'
~0 in s7 reflects mechanism QUALITY, not the truncation."""
import json, re, numpy as np
src = open('runs/arch/design-structured/s6_fuse_mech.py').read()
exec(src.split("Xd = [strip((cl.get(t)")[0])
exec("L = lambda p: aligned_bundle(p, dev.trace_id" + src.split("L = lambda p: aligned_bundle(p, dev.trace_id", 1)[1].split("\nout = {}")[0])
out = {}
for name, f in [('claude_tail500', lambda s: s[-500:]), ('claude_head300', lambda s: s[:300])]:
    Xd = [strip(f(str((cl.get(t) or {}).get('reasoning', '')))) for t in dev.trace_id]
    P = run(Xtr, Ytr, W); sc, ins = crossfit(P); d = sc - ref
    out[name] = {'mapper_err_stats': stats(P), 'crossfit': round(float(sc.mean()), 4), 'delta': round(float(d.mean()), 4), 'ci': boot(d)}
    print(name, out[name])
out['median_reasoning_chars'] = float(np.median([len(str((cl.get(t) or {}).get('reasoning', ''))) for t in dev.trace_id]))
print(out['median_reasoning_chars'])
json.dump(out, open('runs/arch/design-structured/s8_tail_control.json', 'w'), indent=1)
