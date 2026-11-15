"""CPU probe: compare root-code usage of human gold (train) vs Claude teacher labels (unlabelled), per question."""
import json, ast, collections, csv
R='/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/'
train=list(csv.DictReader(open(R+'data/raw/train.csv')))
unl={r['trace_id']:r['question_id'] for r in csv.DictReader(open(R+'data/semisupervised/competition_data/unlabelled.csv'))}
g=collections.defaultdict(collections.Counter); t=collections.defaultdict(collections.Counter)
for r in train: g[r['question_id']][r['root_code']]+=1
for l in open(R+'runs/claude/unl_all_v4.jsonl'):
    r=json.loads(l); a=r['answer']; a=a if isinstance(a,dict) else ast.literal_eval(a)
    q=unl.get(r['trace_id']);
    if q: t[q][a['first_code']]+=1
# question-reweighted teacher distribution: weight each question by its train row count
G=collections.Counter(); T=collections.Counter()
for q,c in g.items():
    if q not in t: continue
    n=sum(c.values()); m=sum(t[q].values())
    for k,v in c.items(): G[k]+=v
    for k,v in t[q].items(): T[k]+=v*n/m
nG=sum(G.values()); nT=sum(T.values())
diff=sorted(set(G)|set(T),key=lambda k:-(abs(G[k]/nG-T[k]/nT)))
rows=[(k,round(G[k]/nG,3),round(T[k]/nT,3)) for k in diff[:15]]
tv=0.5*sum(abs(G[k]/nG-T[k]/nT) for k in set(G)|set(T))
# per-question modal error code agreement (questions with >=2 gold error rows)
agree=tot=0
for q,c in g.items():
    ce=collections.Counter({k:v for k,v in c.items() if k!='NONE'})
    te=collections.Counter({k:v for k,v in t[q].items() if k!='NONE'})
    if sum(ce.values())>=2 and te:
        tot+=1; agree+= ce.most_common(1)[0][0]==te.most_common(1)[0][0]
out={'question_matched_total_variation_gold_vs_teacher':round(tv,3),'largest_share_gaps(code,gold,teacher)':rows,
     'questions_with>=2_gold_errors':tot,'modal_error_code_agreement':round(agree/max(tot,1),3),
     'gold_NONE_share':round(G['NONE']/nG,3),'teacher_NONE_share_question_matched':round(T['NONE']/nT,3)}
print(json.dumps(out,indent=1)); json.dump(out,open(R+'runs/research/q5-datasets/label_shift_probe.json','w'),indent=1)
