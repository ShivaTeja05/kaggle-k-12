"""W3 test 5: the 'appended verdict' is a learned annotation convention. How often does gold contain
'[student answered ... correctly]', what label follows it, and how often does the machine reader emit it on dev/test?"""
import json, re
import numpy as np, pandas as pd
from common import load, ROOT, blend
from k12.core import topk, row_scores
tr, tax, classes, folds, dev, fit = load()
PH = re.compile(r'\[[^\]]*student[^\]]*answer[^\]]*correct[^\]]*\]', re.I)
tr['ph'] = tr.transcription_steps.astype(str).str.contains(PH)
print('gold rows with verdict phrase:', int(tr.ph.sum()), 'of', len(tr), '| root NONE among them:', round((tr.root_code[tr.ph] == 'NONE').mean(), 3),
      '| final_answer_correct True among them:', round(tr.final_answer_correct.astype(str).str.lower().eq('true')[tr.ph].mean(), 3))
print('gold NONE rows with phrase:', round(tr.ph[tr.root_code == 'NONE'].mean(), 3), '| gold error rows with phrase:', round(tr.ph[tr.root_code != 'NONE'].mean(), 3))
print('by fold:', tr.groupby(folds).ph.mean().round(3).to_dict())
tx = json.load(open(ROOT/'runs/pod/tx/tx_dev.json')); txt = json.load(open(ROOT/'runs/pod/tx/tx_test.json'))
dev['g_ph'] = dev.transcription_steps.astype(str).str.contains(PH)
dev['m_ph'] = dev.trace_id.map(lambda t: bool(PH.search(tx.get(t, ''))))
P = blend(dev.trace_id.tolist(), classes); y = dev.root_code.to_numpy(); s = row_scores(y, topk(P, classes))
dev['score'] = s; dev['top1'] = [t[0] for t in topk(P, classes)]
print(pd.crosstab(dev.g_ph, dev.m_ph, rownames=['gold has phrase'], colnames=['machine has phrase']))
for name, m in [('machine phrase', dev.m_ph), ('machine phrase, gold not', dev.m_ph & ~dev.g_ph), ('gold phrase, machine not', dev.g_ph & ~dev.m_ph)]:
    sub = dev[m]
    print(f'{name}: n={len(sub)} truth NONE={np.mean(sub.root_code == "NONE"):.2f} blend top1 NONE={np.mean(sub.top1 == "NONE"):.2f} blend MAP={sub.score.mean():.3f}',
          sub[['trace_id', 'root_code', 'top1']].values.tolist() if len(sub) < 12 else '')
te_ph = sum(bool(PH.search(t)) for t in txt.values())
print('test machine transcripts with phrase:', te_ph, 'of', len(txt), f'({te_ph/len(txt):.3f}) vs dev machine {dev.m_ph.mean():.3f}, train gold {tr.ph.mean():.3f}')
json.dump(dict(gold_rows=int(tr.ph.sum()), gold_none_share=float((tr.root_code[tr.ph] == 'NONE').mean()), dev_machine=int(dev.m_ph.sum()),
               dev_gold=int(dev.g_ph.sum()), test_machine=int(te_ph)), open(ROOT/'runs/deep/approach-W3/w5_verdict_leak.json', 'w'), indent=1)
