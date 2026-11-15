"""Throwaway: tokens the student must generate after the '\\nROOT_LINE:' seed to reach 'ANSWER:' (train targets as proxy)."""
import types, numpy as np
from k12.candidate_student import Student, read_jsonl, SEED, ANSWER
st = Student(types.SimpleNamespace(model=None, device='cpu', load_4bit=False, max_length=2048))
rows = read_jsonl('data/v5/cand_train.jsonl'); seed = len(st.encode(SEED))
n = np.array([len(st.encode('\n' + r['target'][:r['target'].rindex(ANSWER) + len(ANSWER)])) - seed for r in rows])
src = np.array([r['source'] for r in rows])
for s in ('claude', 'gold'):
    m = n[src == s]; print(s, 'p50', int(np.median(m)), 'p95', int(np.percentile(m, 95)), 'p99', int(np.percentile(m, 99)), 'max', int(m.max()), 'share>160', round(float((m > 160).mean()), 4), 'share>140', round(float((m > 140).mean()), 4))
# phase-b effective lr: cosine x warmup x .3 over phase-b steps of the real 1-epoch run (bs4, accum1)
a, b = (src == 'claude').sum(), (src == 'gold').sum(); U = int(np.ceil(a / 4) + np.ceil(b / 4)); A = int(np.ceil(a / 4))
f = lambda s: min(1, (s + 1) / max(1, .05 * U)) * .5 * (1 + np.cos(np.pi * s / U))
print('updates', U, 'phase-a steps', A, 'mean lr factor phase a', round(np.mean([f(s) for s in range(A)]), 4), 'phase b (x.3)', round(.3 * np.mean([f(s) for s in range(A, U)]), 4), 'phase b first step', round(.3 * f(A), 4))
