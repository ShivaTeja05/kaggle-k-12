"""W3 test 1: is the transcriber's 'correction' pulled toward the prompt context (question text + answer key)?
CPU only. Compares gold vs machine transcripts on 301 dev rows, by casebook mechanism."""
import json, re, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/symbolic')
from common import load, ROOT, boot_ci
from symutil import root_lines, body, norm
tr, tax, classes, folds, dev, fit = load()
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json')); txt = json.load(open(ROOT/'runs/pod/tx/tx_test.json'))
lab = pd.read_csv(ROOT/'runs/deep/casebook/labels.csv', keep_default_na=False).set_index('trace_id')
NUM = re.compile(r'\d+(?:[.,]\d+)*')
def nums(s):
    s = str(s).replace('−', '-')
    out = []
    for m in NUM.findall(s):
        v = m.replace('.', '').replace(',', '.') if re.fullmatch(r'\d{1,3}(\.\d{3})+', m) else m.replace(',', '.')
        try: out.append(str(float(v)).rstrip('0').rstrip('.'))
        except ValueError: pass
    return out
def lines(t): return [l for l in str(t).strip().strip('"').splitlines() if l.strip()]
def tail_nums(t, k=2): return set(sum((nums(body(l)) for l in lines(t)[-k:]), []))
def key_hit(t, key):
    kn = set(nums(key)); return bool(kn) and bool(kn & tail_nums(t))
rows = []
for r in dev.itertuples(index=False):
    g, m = str(r.transcription_steps), tx.get(r.trace_id, '')
    ctx = Counter(nums(r.question_text) + nums(r.correct_answer))
    G, T = Counter(nums(g)), Counter(nums(m))
    ins, dele = T - G, G - T
    ins_ctx = sum(v for k, v in ins.items() if k in ctx); ins_out = sum(v for k, v in ins.items() if k not in ctx)
    del_stud = sum(v for k, v in dele.items() if k not in ctx); del_ctx = sum(v for k, v in dele.items() if k in ctx)
    gold_ctx_share = sum(v for k, v in G.items() if k in ctx) / max(1, sum(G.values()))
    rl, span = root_lines(r.misconceptions)
    span_nums = Counter(nums(span)) if span else Counter()
    span_lost = sum((span_nums - T).values()) if span else 0
    span_in = (norm(span) in norm(''.join(body(l) for l in lines(m)))) if span else None
    mech = lab.mech[r.trace_id] if r.trace_id in lab.index else ('OK' if True else '')
    rows.append(dict(trace_id=r.trace_id, q=r.question_id, truth=r.root_code, mech=mech,
                     gold_key=key_hit(g, r.correct_answer), tx_key=key_hit(m, r.correct_answer),
                     ins_ctx=ins_ctx, ins_out=ins_out, del_stud=del_stud, del_ctx=del_ctx, gold_ctx_share=gold_ctx_share,
                     n_gold=sum(G.values()), span_nums=sum(span_nums.values()), span_lost=span_lost, span_in=span_in,
                     sub_to_ctx=int(ins_ctx > 0 and del_stud > 0)))
d = pd.DataFrame(rows)
d['converge'] = d.tx_key & ~d.gold_key          # tx tail now matches the key, gold tail did not
d['diverge'] = ~d.tx_key & d.gold_key
d['grp'] = np.where(d.mech == 'M6', 'M6', np.where(d.mech == 'OK', 'OK(top1 right)', 'other miss'))
# 'OK' assignment: rows not in the casebook are top-1 hits
d.loc[~d.trace_id.isin(lab.index), 'grp'] = 'OK(top1 right)'
err = d.truth != 'NONE'
d['grp2'] = d.grp + np.where(err, ' / error', ' / NONE')
summ = d.groupby('grp2').agg(n=('trace_id', 'size'), converge=('converge', 'mean'), diverge=('diverge', 'mean'),
                             gold_key=('gold_key', 'mean'), tx_key=('tx_key', 'mean'), sub_to_ctx=('sub_to_ctx', 'mean'),
                             ins_ctx=('ins_ctx', 'mean'), ins_out=('ins_out', 'mean'), del_stud=('del_stud', 'mean'),
                             span_lost_rate=('span_lost', 'sum'), span_nums=('span_nums', 'sum'),
                             gold_ctx_share=('gold_ctx_share', 'mean')).round(3)
summ['span_lost_rate'] = (summ.span_lost_rate / summ.span_nums.clip(lower=1)).round(3)
print(summ.to_string())
# Share of inserted (hallucinated/misread) numbers that are context numbers, vs the base rate of context numbers in gold
tot_ins_ctx, tot_ins_out = d.ins_ctx.sum(), d.ins_out.sum()
base = (d.gold_ctx_share * d.n_gold).sum() / d.n_gold.sum()
print('inserted numbers: in prompt context', tot_ins_ctx, 'not', tot_ins_out, 'share', round(tot_ins_ctx / (tot_ins_ctx + tot_ins_out), 3), 'base rate of ctx numbers in gold', round(base, 3))
def share(sel):
    a, b = d.ins_ctx[sel].sum(), d.ins_out[sel].sum(); return round(a / max(1, a + b), 3), int(a + b)
print('ctx share of insertions: M6', share(d.grp == 'M6'), 'other miss', share(d.grp == 'other miss'), 'OK', share(d.grp == 'OK(top1 right)'))
# error rows only: P(converge | error row whose gold tail does not match key)
e = err & ~d.gold_key
print('error rows w/ gold tail != key:', int(e.sum()), 'tx converges to key:', int(d.converge[e].sum()), round(d.converge[e].mean(), 3))
for g_ in ['M6', 'other miss', 'OK(top1 right)']:
    s = e & (d.grp == g_); print('  ', g_, int(s.sum()), 'converge', int(d.converge[s].sum()))
# Label-free test monitor: tx tail matches key rate on test vs dev
te = pd.read_csv(ROOT/'data/raw/test.csv', keep_default_na=False)
tk = np.mean([key_hit(txt.get(r.trace_id, ''), r.correct_answer) for r in te.itertuples(index=False)])
gk_tr = np.mean([key_hit(r.transcription_steps, r.correct_answer) for r in tr.itertuples(index=False)])
print('tail==key rate: dev tx', round(d.tx_key.mean(), 3), 'dev gold', round(d.gold_key.mean(), 3), 'train gold (1500)', round(gk_tr, 3), 'test tx', round(tk, 3))
# final_answer_correct column agreement
fac = dev.final_answer_correct.astype(str).str.lower().eq('true').to_numpy()
print('final_answer_correct=False rows:', int((~fac).sum()), 'gold tail key-hit among them', round(d.gold_key[~fac].mean(), 3), 'tx key-hit', round(d.tx_key[~fac].mean(), 3))
d.to_csv(ROOT/'runs/deep/approach-W3/w1_rows.csv', index=False)
json.dump(dict(summary=summ.reset_index().to_dict('records')), open(ROOT/'runs/deep/approach-W3/w1_context_pull.json', 'w'), indent=1, default=str)
