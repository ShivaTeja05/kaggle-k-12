import sys; sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/reading-impact')
from common import *
for k in P: print(k, round(rs(P[k]).mean(), 4))
for e in ('8_tx_e2', '8_tx_e3', 's100_tx'):
    print('blend', e, round(rs(geo([('32_tx', .35), (e, .30), ('vl', .35)])).mean(), 4))
