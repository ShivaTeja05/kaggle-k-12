"""Candidate-level stackers (LR, HistGradientBoosting) with prob/rank/entropy/agreement/prior features.
Cross-fitted: repeated (5x) question-grouped 5-fold inside the 301 dev rows. Reference = .35/.30/.35 blend."""
import json, sys, warnings
from common import *
from priors import *
from cf import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

names = sys.argv[1].split(',') if len(sys.argv) > 1 else ['t32', 't8e3', 'vl']
tag = sys.argv[2] if len(sys.argv) > 2 else 'base3'
M = {n: load(n) for n in names}
L = {n: np.log(np.maximum(p, 1e-12)) for n, p in M.items()}
ref = geo([load('t32'), load('t8e3'), load('vl')], [.35, .30, .35]); refp = pts(ref); lref = np.log(ref)
eq = geo(list(M.values()), [1/len(M)]*len(M)); leq = np.log(eq)
gp, cp = gold_prior(), claude_prior(a=5)
qcount = dev.question_id.map(fit.question_id.value_counts()).fillna(0).to_numpy()
ranks = {n: np.argsort(np.argsort(-p, 1), 1) + 1 for n, p in M.items()}
ent = {n: -(p*np.log(np.maximum(p, 1e-12))).sum(1) for n, p in M.items()}
eqrank = np.argsort(np.argsort(-eq, 1), 1) + 1
TOP = 8
pos, feats = [], []
for i in range(len(dev)):
    pool = list(dict.fromkeys([j for n in names for j in np.argsort(-M[n][i])[:TOP]] + list(np.argsort(-gp[i])[:3])))
    for j in pool:
        f = []
        for n in names:
            f += [L[n][i, j], L[n][i, j] - L[n][i].max(), 1/ranks[n][i, j], ranks[n][i, j] == 1, ranks[n][i, j] <= 3, ent[n][i]]
        f += [leq[i, j], leq[i, j] - leq[i].max(), 1/eqrank[i, j],
              sum(ranks[n][i, j] == 1 for n in names), sum(ranks[n][i, j] <= 3 for n in names),
              np.log(gp[i, j]), np.log(cp[i, j]), np.log1p(qcount[i]), np.log(gp[i, j])*np.log1p(qcount[i]),
              j == ci['NONE']]
        f += [L[n][i, j]*(j == ci['NONE']) for n in names]
        feats.append(f); pos.append((i, j))
X = np.asarray(feats, float); pos = np.asarray(pos); tgt = (pos[:, 1] == y[pos[:, 0]]).astype(int)
cover = np.mean([y[i] in set(pos[pos[:, 0] == i, 1]) for i in range(len(dev))])
print(tag, 'models', names, 'candidates', len(X), 'coverage', round(cover, 4), flush=True)

def rowpts(conf, mask_rows):
    S = 1e-6*eq.copy(); S[pos[:, 0], pos[:, 1]] = conf + 1e-3
    return pts(S[mask_rows], y[mask_rows])

def run(make, cols=None):
    cols = slice(None) if cols is None else cols
    def fa(tr, he):
        a = tr[pos[:, 0]]; b = he[pos[:, 0]]
        conf = np.zeros(len(X)); conf[b] = make().fit(X[a][:, cols], tgt[a]).predict_proba(X[b][:, cols])[:, 1]
        return rowpts(conf, he)
    return crossfit_model(fa, n_rep=5)

report = {}
def show(name, r):
    d, c, pg = boot_ci(r, refp); report[name] = {'map3': float(r.mean()), 'delta': d, 'ci95': c, 'p_gt0': pg}
    print(f'{tag}:{name:40s} {r.mean():.4f} d={d:+.4f} CI[{c[0]:+.4f},{c[1]:+.4f}] P>0={pg:.2f}', flush=True)

show('equal_geo_blend(fixed)', pts(eq))
nprior = 6 + len(names)  # prior/NONE columns at the end
base_cols = np.r_[0:X.shape[1]]
noprior = np.array([k for k in range(X.shape[1]) if k not in range(6*len(names)+5, 6*len(names)+9)])
for C in [.01, .03, .1, .3, 1]:
    show(f'LR_C{C}', run(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))))
show('LR_C0.1_noprior', run(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000)), noprior))
for lr, it, leaf in [(.05, 100, 15), (.03, 200, 31), (.05, 300, 7)]:
    show(f'HGB_lr{lr}_it{it}_leaf{leaf}', run(lambda: HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaf, min_samples_leaf=20, l2_regularization=1.0)))
json.dump(report, open(R+f'runs/arch/combiner-headroom/s3_stacker_{tag}.json', 'w'), indent=1)
