import sys, json, numpy as np
sys.path.insert(0, 'runs/arch/label-structure')
from common import *
tr, tax, classes, folds, dev, fit = load(); te = read_csv(ROOT/'data/raw/test.csv')
ci = {c: i for i, c in enumerate(classes)}
def bal(P, codes, it):
    pi = np.bincount(codes.map(ci), minlength=len(classes)) + .5; pi /= pi.sum(); b = np.zeros(len(classes))
    for _ in range(it): b += np.log(pi / normalize(P * np.exp(b)).mean(0))
    return normalize(P * np.exp(.5 * b))
Q = bal(blend(dev.trace_id.tolist(), classes), fit.root_code, 30)
n = ci['NONE']
print('dev NONE outside top10', int(sum(n not in np.argsort(-q, kind='stable')[:10] for q in Q)), 'of', len(Q))
QT = aligned_bundle(ROOT/'runs/arch/label-structure/test_blend_balanced_l0.5.npz', te.trace_id.tolist(), classes)
print('test NONE outside top10', int(sum(n not in np.argsort(-q, kind='stable')[:10] for q in QT)), 'of', len(QT))
D = [json.loads(x) for x in open(ROOT/'data/v5/cand_dev.jsonl')]; X = [json.loads(x) for x in open(ROOT/'data/v5/cand_test.jsonl')]
print('dev rows NONE last in order1', sum(d['codes_per_prompt'][0][-1] == 'NONE' for d in D), 'test', sum(d['codes_per_prompt'][0][-1] == 'NONE' for d in X))
