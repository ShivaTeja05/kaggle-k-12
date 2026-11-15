"""W5: where does the code appear in v5 Stage B targets, and how is the v5 Claude subset distributed over questions/test coverage?"""
import json, re, collections
import numpy as np, pandas as pd
ROOT='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12'
codes=set(pd.read_csv(ROOT+'/data/raw/taxonomy_v1.csv').iloc[:,0].astype(str))
pat=re.compile(r'\b[A-Za-z]{2,3}-\d\d\b')
rows=[json.loads(l) for l in open(ROOT+'/data/v5/cand_train.jsonl')]
out={}
for src in ['claude','gold']:
    R=[r for r in rows if r['source']==src]
    named=0; first_sent=0; frac=[]; truth_named=0; other_named=0
    for r in R:
        why=r['target'].split('WHY:',1)[1].split('\nANSWER:')[0]
        m=[x for x in pat.finditer(why)]
        letter=r['target'].rsplit('ANSWER:',1)[1].strip()
        truth=r['codes'][r['letters'].index(letter)]
        if m:
            named+=1; frac.append(m[0].start()/max(1,len(why)))
            first_sent+= m[0].start() < (why.find('. ')+1 if '. ' in why else len(why))
            truth_named+= any(x.group(0)==truth for x in m)
            other_named+= any(x.group(0)!=truth for x in m)
    out[src]=dict(rows=len(R), why_names_a_code=round(named/len(R),3), code_in_first_sentence=round(first_sent/len(R),3),
                  median_rel_position_first_code=round(float(np.median(frac)),3) if frac else None,
                  why_names_truth=round(truth_named/len(R),3), why_names_other_code=round(other_named/len(R),3))
unl=pd.read_csv(ROOT+'/data/semisupervised/competition_data/unlabelled.csv').set_index('trace_id')
test=pd.read_csv(ROOT+'/data/raw/test.csv'); train=pd.read_csv(ROOT+'/data/raw/train.csv')
cl=[r for r in rows if r['source']=='claude']
qv5=pd.Series([unl.at[r['trace_id'],'question_id'] for r in cl]).value_counts()
tq=test.question_id.value_counts()
out['v5_claude_questions']=int(len(qv5)); out['v5_claude_rows_per_q_median']=float(qv5.median()); out['v5_claude_rows_per_q_p90']=float(qv5.quantile(.9))
cov=test.question_id.isin(unl.question_id.unique())
out['test_rows_uncovered']=int((~cov).sum()); out['test_q_uncovered']=int(test.question_id[~cov].nunique())
trq=train.question_id.value_counts()
unc=test[~cov]
out['uncovered_test_rows_whose_q_in_train']=round(float(unc.question_id.isin(trq.index).mean()),3)
out['uncovered_test_q_train_rows_median']=float(unc.drop_duplicates('question_id').question_id.map(trq).fillna(0).median())
out['gold_train_rows_on_uncovered_test_q']=int(train.question_id.isin(unc.question_id.unique()).sum())
out['gold_train_rows_total']=len(train)
# test-weighted mass: share of test rows per question vs share of v5 claude mass per question (TV distance over questions)
p_test=(tq/tq.sum()); p_v5=(qv5/qv5.sum()).reindex(p_test.index).fillna(0)
out['tv_distance_testq_vs_v5claudeq']=round(float(0.5*np.abs(p_test-p_v5).sum() + 0.5*(qv5[~qv5.index.isin(p_test.index)].sum()/qv5.sum())),3)
# topics
out['uncovered_test_topics']=int(unc.topic_name.nunique()); out['uncovered_test_topics_not_in_pool']=int((~unc.topic_name.drop_duplicates().isin(unl.topic_name.unique())).sum())
print(json.dumps(out,indent=1)); json.dump(out,open(ROOT+'/runs/deep/approach-W5/w5_targets.json','w'),indent=1)
