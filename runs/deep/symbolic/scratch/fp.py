import json, pandas as pd, random, sys
tr=pd.read_csv('../../../data/raw/train.csv',keep_default_na=False,dtype=str)
g={r['trace_id']:r for r in map(json.loads,open('out/gold_train.jsonl'))}
none=tr[tr.root_code=='NONE']
fl=[(r, g[r.trace_id]) for r in none.itertuples() if any(x['invalid'] for x in g[r.trace_id]['lines'])]
print('NONE flagged', len(fl), '/', len(none))
err=tr[tr.root_code!='NONE']; print('ERR flagged', sum(any(x['invalid'] for x in g[t]['lines']) for t in err.trace_id), '/', len(err))
random.seed(int(sys.argv[1]) if len(sys.argv)>1 else 2)
for r,rec in random.sample(fl,25):
    bad=[x for x in rec['lines'] if x['invalid']][0]
    line=[l for l in r.transcription_steps.splitlines() if l.strip().startswith(f"L{bad['line']} ")]
    print(repr(line[0][:150]) if line else bad['line'], '|', bad['reasons'][:2])
