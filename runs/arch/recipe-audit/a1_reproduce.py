import json, sys
sys.path.insert(0, '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/arch/recipe-audit')
from common import *
res = {}
for n in BUNDLES:
    try: res[n] = round(map3(truth, load(n), classes), 4)
    except Exception as e: res[n] = str(e)
for e in ('q8_tx_e2', 'q8_tx_e3'):
    res['blend_' + e] = round(map3(truth, geo([(load('q32_tx'), .35), (load(e), .30), (load('vl_distill'), .35)]), classes), 4)
print(json.dumps(res, indent=1))
