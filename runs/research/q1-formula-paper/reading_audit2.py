import json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from reading_audit import tx, train, strip_ids, norm, mtoks, lev  # reruns audit (fine, cheap)
lr, f1s, ex_lines, highcer_lenratio = [], [], [], []
crossed = 0
for tid, text in tx.items():
    g, m = strip_ids(train.loc[tid, 'transcription_steps']), strip_ids(text)
    gl, ml = sum(len(norm(x)) for x in g), sum(len(norm(x)) for x in m)
    lr.append(ml / max(1, gl))
    gc, mc = Counter(mtoks(' '.join(g))), Counter(mtoks(' '.join(m)))
    inter = sum((gc & mc).values()); p = inter / max(1, sum(mc.values())); r = inter / max(1, sum(gc.values()))
    f1s.append(2 * p * r / max(1e-9, p + r))
    gs = set(norm(x) for x in g); ex_lines.append(sum(norm(x) in gs for x in m) / max(1, len(g)))
    if re.search(r'coret|cross|~~|\[', str(train.loc[tid, 'transcription_steps']), re.I): crossed += 1
q = lambda a: dict(mean=round(float(np.mean(a)), 4), median=round(float(np.median(a)), 4), p10=round(float(np.percentile(a, 10)), 4), p90=round(float(np.percentile(a, 90)), 4))
out = dict(length_ratio_machine_over_gold=q(lr), math_token_bag_f1_order_free=q(f1s),
           exact_line_match_fraction_of_gold_lines=q(ex_lines), gold_rows_with_markup_like_crossouts=crossed)
print(json.dumps(out, indent=1))
Path(__file__).with_name('reading_audit2.json').write_text(json.dumps(out, indent=1))
