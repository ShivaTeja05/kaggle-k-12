import json, pandas as pd, re, collections
tr=pd.read_csv('../../../data/raw/train.csv',keep_default_na=False,dtype=str)
g={r['trace_id']:r for r in map(json.loads,open('out/gold_train.jsonl'))}
def typ(reason):
    if reason.startswith('cont:'): return 'cont'
    if 'contradicts' in reason: return 'bind'
    if re.match(r'^\w+: \[', reason): return 'eq'
    return 'arith'
def roots(m):
    for a in (json.loads(m) if m else []):
        if a.get('root'):
            n=[int(x) for x in re.findall(r'L(\d+)',a['step'])]
            return set(range(n[0],n[-1]+1))
    return None
st=collections.defaultdict(lambda: collections.Counter())
for r in tr.itertuples():
    rec=g[r.trace_id]; R=roots(r.misconceptions); isn=r.root_code=='NONE'
    for T in ['arith','eq','bind','cont']:
        lines=[x['line'] for x in rec['lines'] if any(typ(z)==T for z in x['reasons'])]
        if not lines: continue
        f=lines[0]
        if isn: st[T]['none_rows_flagged']+=1
        else:
            st[T]['err_rows_flagged']+=1
            if R: st[T]['first_hits_root']+=f in R; st[T]['with_root']+=1; st[T]['any_hits_root']+=any(l in R for l in lines)
for T,c in st.items():
    print(T, dict(c), 'prec_first=%.3f'%(c['first_hits_root']/max(1,c['with_root'])), 'none_rate=%.3f'%(c['none_rows_flagged']/506), 'err_rate=%.3f'%(c['err_rows_flagged']/994))
