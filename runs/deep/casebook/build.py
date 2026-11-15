"""Build the failure casebook table (CPU only). Writes casebook.csv, cases_*.txt."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from common import load, blend, ROOT
from k12.core import topk, row_scores
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_claude import claude_scores

OUT = ROOT/'runs/deep/casebook'
tr, tax, classes, folds, dev, fit = load()
ids = dev.trace_id.tolist(); truth = dev.root_code.to_numpy()
P = blend(ids, classes)
C, _ = claude_scores(ROOT/'runs/claude/dev_opus.jsonl', ids, classes)
rb, rc = topk(P, classes), topk(C, classes)
sb, sc = row_scores(truth, rb), row_scores(truth, rc)
print('blend', sb.mean().round(4), 'claude', sc.mean().round(4))
ans = {json.loads(l)['trace_id']: json.loads(l)['answer'] for l in open(ROOT/'runs/claude/dev_opus.jsonl')}
tx = json.loads((ROOT/'runs/pod/tx/tx_dev.json').read_text())
name = dict(zip(tax.id, tax.name)); name['NONE'] = 'no misconception'
sec = dict(zip(tax.id, tax.section)); sec['NONE'] = 'NONE'
nature = dict(zip(tax.id, tax.nature)); nature['NONE'] = 'none'
qcodes = defaultdict(Counter)
for q, c in zip(fit.question_id, fit.root_code): qcodes[q][c] += 1
ci = {c: i for i, c in enumerate(classes)}
rows = []
for i, r in dev.iterrows():
    t = truth[i]; q = r.question_id
    seen_q = q in qcodes and sum(qcodes[q].values()) > 0; seen_pair = seen_q and qcodes[q][t] > 0
    if t == 'NONE': slice_ = 'none'
    elif not seen_q: slice_ = 'unseen_q'
    elif not seen_pair: slice_ = 'seen_q_new_code'
    else: slice_ = 'seen_pair'
    mis = json.loads(r.misconceptions) if isinstance(r.misconceptions, str) and r.misconceptions.strip() else []
    root = [m for m in mis if m.get('root')]
    root = root[0] if root else {}
    a = ans.get(r.trace_id, {})
    order = np.argsort(-P[i])
    rank_truth = int(np.where(order == ci[t])[0][0]) + 1
    top1 = rb[i][0]
    rows.append(dict(trace_id=r.trace_id, question_id=q, grade=r.grade, topic=r.topic_name, slice=slice_,
        truth=t, truth_name=name.get(t), truth_section=sec.get(t), truth_nature=nature.get(t),
        final_answer_correct=r.final_answer_correct,
        blend_top3=' '.join(rb[i]), blend_p3=' '.join(f'{P[i,ci[c]]:.2f}' for c in rb[i]),
        p_truth=round(float(P[i, ci[t]]), 4), rank_truth=rank_truth,
        blend_top1_in_q_fit_codes=bool(qcodes.get(q, {}).get(top1, 0) > 0) if seen_q else None,
        q_fit_codes=' '.join(f'{c}:{n}' for c, n in qcodes[q].most_common()) if q in qcodes else '',
        blend_top1_section_match=sec.get(top1) == sec.get(t), blend_top1_family_match=top1.split('-')[0] == t.split('-')[0],
        claude_top3=' '.join(str(a.get(k)) for k in ('first_code', 'second_code', 'third_code')),
        claude_root_line=str(a.get('root_line', ''))[:80], s_blend=sb[i], s_claude=sc[i],
        gold_root_step=root.get('step'), gold_root_span=root.get('span'), gold_root_cat=root.get('category'),
        n_misc=len(mis), other_cats=' '.join(sorted({m['category'] for m in mis if not m.get('root')})),
        top1_is_nonroot_annot=top1 in {m['category'] for m in mis if not m.get('root')}))
df = pd.DataFrame(rows)
df.to_csv(OUT/'all_dev_rows.csv', index=False)
err = df[df.truth != df.blend_top3.str.split().str[0]]
print('blend top1 wrong rows', len(err), err.slice.value_counts().to_dict())
print(df.groupby('slice')[['s_blend', 's_claude']].agg(['mean', 'size']).round(3))
err.to_csv(OUT/'casebook.csv', index=False)
# Readable dumps for manual reading
def dump(sub, path):
    with open(path, 'w') as f:
        for _, x in sub.iterrows():
            r = dev.loc[dev.trace_id == x.trace_id].iloc[0]; a = ans.get(x.trace_id, {})
            mis = json.loads(r.misconceptions) if isinstance(r.misconceptions, str) and r.misconceptions.strip() else []
            f.write(f"=== {x.trace_id} [{x.slice}] g{x.grade} {x.topic} | final_correct={x.final_answer_correct}\n")
            f.write(f"Q: {r.question_text}\nKEY: {r.correct_answer}\n")
            f.write(f"GOLD TX:\n{r.transcription_steps}\n")
            f.write(f"MACHINE TX:\n{tx.get(x.trace_id, '')}\n")
            for m in mis:
                f.write(f"  MISC {'ROOT' if m.get('root') else 'sub '} {m.get('step')} [{m.get('category')} {name.get(m.get('category'), '')}] span={m.get('span')!r} :: {m.get('text')}\n")
            f.write(f"TRUTH {x.truth} ({x.truth_name}; {x.truth_nature}) | q_fit_codes: {x.q_fit_codes}\n")
            f.write(f"BLEND {x.blend_top3} p={x.blend_p3} | {' / '.join(name.get(c, c) for c in x.blend_top3.split())} | truth rank {x.rank_truth} p={x.p_truth}\n")
            f.write(f"CLAUDE {x.claude_top3} root_line={a.get('root_line')} s={x.s_claude}\n  REASON: {a.get('reasoning')}\n\n")
order = {'seen_q_new_code': 0, 'unseen_q': 1, 'seen_pair': 2, 'none': 3}
err = err.assign(o=err.slice.map(order)).sort_values(['o', 's_claude'], ascending=[True, False])
for s in order:
    dump(err[err.slice == s], OUT/f'cases_{s}.txt')
# rows where we beat Claude
win = df[df.s_blend > df.s_claude].sort_values('s_claude')
print('blend beats claude rows', len(win), win.slice.value_counts().to_dict())
dump(win, OUT/'cases_we_win.txt')
win.to_csv(OUT/'we_win.csv', index=False)
