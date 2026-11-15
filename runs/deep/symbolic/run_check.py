"""Run the symbolic checker over gold train transcripts, machine dev/test transcripts and
(optionally) Claude transcripts of unlabelled photos. Writes runs/deep/symbolic/out/*.jsonl."""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from symengine import check_transcript, final_answer_flag

OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)


def work(args):
    tid, tx, q, a = args
    try:
        lines = check_transcript(tx, q, a)
    except Exception as e:
        lines = []
    try:
        fac = final_answer_flag(tx, a, q, k=2)
        fac1 = final_answer_flag(tx, a, q, k=1)
        facall = final_answer_flag(tx, a, q, k=99)
    except Exception:
        fac = fac1 = facall = None
    return dict(trace_id=tid, lines=lines, fac=fac, fac1=fac1, facall=facall)


def run(name, items):
    t0 = time.time()
    res = []
    with Pool(8, maxtasksperchild=500) as p:
        for i, r in enumerate(p.imap(work, items, chunksize=4)):
            res.append(r)
            if i % 2000 == 0:
                print(name, i, f'{time.time() - t0:.0f}s', flush=True)
    with open(OUT / f'{name}.jsonl', 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    print(name, len(res), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    which = sys.argv[1:] or ['gold', 'mdev']
    tr = pd.read_csv(ROOT / 'data/raw/train.csv', keep_default_na=False, dtype=str)
    if 'gold' in which:
        run('gold_train', [(r.trace_id, r.transcription_steps, r.question_text, r.correct_answer) for r in tr.itertuples()])
    if 'mdev' in which:
        tx = json.load(open(ROOT / 'runs/pod/tx/tx_dev.json'))
        d = tr[tr.trace_id.isin(tx)]
        run('machine_dev', [(r.trace_id, tx[r.trace_id], r.question_text, r.correct_answer) for r in d.itertuples()])
    if 'cdev' in which:   # Claude's own transcription of dev photos (reference reader)
        rows = [json.loads(l) for l in open(ROOT / 'runs/claude/dev_opus.jsonl')]
        m = {r['trace_id']: r['answer']['transcription'] for r in rows if r.get('answer')}
        d = tr[tr.trace_id.isin(m)]
        run('claude_dev', [(r.trace_id, m[r.trace_id], r.question_text, r.correct_answer) for r in d.itertuples()])
    if 'unl' in which:
        items = []
        for l in open(ROOT / 'data/claude_dataset/labels.jsonl'):
            r = json.loads(l)
            if r.get('split') == 'unlabelled' and r.get('transcription'):
                items.append((r['trace_id'], r['transcription'], r.get('question_text', ''), r.get('correct_answer', '')))
        seen = set(); items = [x for x in items if not (x[0] in seen or seen.add(x[0]))]
        run('claude_unl', items)
