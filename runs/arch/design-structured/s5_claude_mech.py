"""Same cheap mapper as s4, applied to a MACHINE-written mechanism: Claude's dev reasoning with code ids
and code names stripped (diagnostic only; Claude is not usable at test). Also the annotator text with
code ids/names stripped, to check s4 was not reading code names."""
import json, re, numpy as np
exec(open('runs/arch/design-structured/s4_mechanism.py').read().split("out = {}")[0])
names = sorted(tax.name.str.lower().tolist(), key=len, reverse=True)
def strip(t):
    t = re.sub(r'\b[A-Z]{2,4}-\d{2}\b', ' ', str(t)); tl = t.lower()
    for n in names: tl = tl.replace(n, ' ')
    return tl
cl = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open('runs/claude/dev_opus.jsonl')}
Xtr = [strip(x) for x in Xf + Xc]; Ytr = yf + yc; W = [3.]*len(Xf) + [1.]*len(Xc)
out = {}
for name, Xd in [('annot_text_stripped', [strip(x) for x in Xd]),
                 ('claude_dev_reasoning_stripped', [strip((cl.get(t) or {}).get('reasoning', '')) for t in dev.trace_id])]:
    globals()['Xd'] = Xd
    out[name] = stats(run(Xtr, Ytr, W))
out['n_claude_reasoning_with_code_id'] = int(sum(bool(re.search(r'\b[A-Z]{2,4}-\d{2}\b', str((cl.get(t) or {}).get('reasoning', '')))) for t in dev.trace_id))
print(json.dumps(out, indent=1)); json.dump(out, open('runs/arch/design-structured/s5_claude_mech.json', 'w'), indent=1)
