"""W5 approach C feasibility: how many correct (NONE) gold transcripts exist as donors for synthetic error injection,
on the test questions the Claude pool never covers, and how many have a symbolically verified line a mal-rule could corrupt?"""
import json
import pandas as pd
ROOT='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'
tr=pd.read_csv(ROOT+'/data/raw/train.csv'); te=pd.read_csv(ROOT+'/data/raw/test.csv')
unlq=set(pd.read_csv(ROOT+'/data/semisupervised/competition_data/unlabelled.csv',usecols=['question_id']).question_id)
sym={}
for l in open(ROOT+'/runs/deep/symbolic/out/gold_train.jsonl'):
    r=json.loads(l); sym[r['trace_id']]=r
INJ={'pow':['EXP-01','EXP-02','EXP-03','EXP-04'],'div':['FRA-01','FRA-03','OPS-02','INT-03'],'mul':['OPS-02','INT-03','ALG-04'],
     'sub':['INT-02','INT-04','OOO-01'],'add':['INT-02','OOO-01','ALG-02']}
def donor_ok(t):
    r=sym.get(t)
    if not r: return False, set()
    codes=set()
    for ln in r['lines']:
        if ln['n_valid']>0:
            for o in ln['ops']: codes|=set(INJ.get(o,[]))
    return bool(codes), codes
out={}
for name,mask in [('all_test_q',te.question_id.isin(tr.question_id)),('uncovered_test_q',~te.question_id.isin(unlq))]:
    qs=set(te.question_id[mask]); rows=te[mask]
    don=tr[(tr.root_code=='NONE')&tr.question_id.isin(qs)]
    ok=[donor_ok(t) for t in don.trace_id]
    okq=set(q for q,(f,_) in zip(don.question_id,ok) if f)
    out[name]=dict(test_rows=int(len(rows)),test_questions=len(set(rows.question_id)),
                   gold_none_donor_rows=int(len(don)),donor_rows_with_verified_injectable_line=int(sum(f for f,_ in ok)),
                   test_questions_with_a_donor=len(set(don.question_id)),test_questions_with_injectable_donor=len(okq),
                   test_rows_on_questions_with_injectable_donor=int(rows.question_id.isin(okq).sum()),
                   gold_error_rows_on_these_q=int(((tr.root_code!='NONE')&tr.question_id.isin(qs)).sum()))
print(json.dumps(out,indent=1)); json.dump(out,open(ROOT+'/runs/deep/approach-W5/w5_inject_feasibility.json','w'),indent=1)
