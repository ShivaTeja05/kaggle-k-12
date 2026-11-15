"""Dump Claude-better and blend-better dev rows for qualitative reading (dev rows only)."""
import json, sys
import pandas as pd
from k12.core import read_csv
d = pd.read_csv('runs/arch/teacher-gap/rows.csv', keep_default_na=False, dtype={'trace_id': str, 'question_id': str})
tr = read_csv('data/raw/train.csv').set_index('trace_id')
tax = read_csv('data/raw/taxonomy_v1.csv'); name = dict(zip(tax.id, tax.name)) | {'NONE': 'no major error'}
which = sys.argv[1]
out = []
for r in d[d.outcome == which].itertuples():
    t = tr.loc[r.trace_id]
    ann = [a for a in json.loads(t.misconceptions or '[]') if a['root']]
    out.append(f"=== {r.trace_id} q={r.question_id} topic={r.topic} | truth {r.truth} ({name[r.truth]}) | blend {r.blend} [{r.stage_blend}] | claude {r.claude} [{r.stage_claude}] | s_b={r.s_blend:.2f} s_c={r.s_claude:.2f} root_loc={r.root_loc} tx_sim={r.tx_sim:.2f}\n"
               f"Q: {t.question_text[:300]}\nKEY: {t.correct_answer[:120]}\n"
               f"GOLD ROOT: {ann[0]['step'] + ' ' + ann[0]['span'] + ' :: ' + ann[0]['text'][:300] if ann else 'NONE'}\n"
               f"BLEND TOP1 NAME: {name[r.blend.split()[0]]}\nCLAUDE root_line: {r.claude_root_line}\nCLAUDE: {r.claude_reason[:700]}\n")
open(f'runs/arch/teacher-gap/cases_{which}.txt', 'w').write('\n'.join(out))
print(len(out))
