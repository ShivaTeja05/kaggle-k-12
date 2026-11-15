"""Run the checker on one shard of the Claude-transcribed unlabelled photos (serial, resumable).
usage: run_unl_shard.py SHARD NSHARDS"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from run_check import work

k, n = int(sys.argv[1]), int(sys.argv[2])
items, seen = [], set()
for l in open(ROOT / 'data/claude_dataset/labels.jsonl'):
    r = json.loads(l)
    if r.get('split') == 'unlabelled' and r.get('transcription') and r['trace_id'] not in seen:
        seen.add(r['trace_id'])
        items.append((r['trace_id'], r['transcription'], r.get('question_text', ''), r.get('correct_answer', '')))
items = items[k::n]
out = HERE / f'out/unl_shard{k}.jsonl'
done = set()
if out.exists():
    done = {json.loads(l)['trace_id'] for l in open(out)}
cur = HERE / f'out/unl_shard{k}.current'
t0 = time.time()
with open(out, 'a') as f:
    for i, it in enumerate(items):
        if it[0] in done:
            continue
        cur.write_text(it[0])
        try:
            r = work(it)
        except BaseException as e:
            r = dict(trace_id=it[0], lines=[], fac=None, fac1=None, facall=None, error=repr(e)[:100])
        f.write(json.dumps(r) + '\n'); f.flush()
print('shard', k, len(items), f'{time.time() - t0:.0f}s')
