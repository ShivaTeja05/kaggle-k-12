"""Fit one configuration of the content-bearing candidate proxy and save its dev letter probabilities.

score(i, k) = theta . [prior feats, example sim](i, k) + W[code_ik] . phi_i ,  phi_i = [working TF-IDF, question TF-IDF, 1]
Loss: weighted soft cross-entropy over the row's 10 candidates (mode 'ce'), or the A3 factorised loss (mode 'poe'):
2-way NONE-vs-error CE with no bias + code-given-error soft CE with a fixed lam * centred log question prior offset
(offset dropped at inference). A1 weights: error rows with LOO-unseen truth x K, error-weight share rescaled to the
original 0.664, mean weight 1. Full-batch L-BFGS (deterministic).
usage: w2_fit.py MODE K LAM L2   e.g. w2_fit.py ce 3 0 1e-4
"""
import pickle, sys, time, json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

OUT = '/Users/gaddamshivateja/Documents/ChatGPT/kaggle-K-12/runs/deep/test-W2'
mode, K, lam, l2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
C = pickle.load(open(f'{OUT}/cache.pkl', 'rb'))
tr, dv = C['tr'], C['dv']; NC = len(C['classes'])


def weights(K):
    w0, e, n = tr['w0'], tr['err'], tr['newp']
    w = w0 * np.where(e & n, K, 1.)
    tgt = w0[e].sum() / w0.sum()
    a = tgt * w[~e].sum() / ((1 - tgt) * w[e].sum()); w = np.where(e, w * a, w)
    return w / w.mean()


def design(d):
    Phi = sp.hstack([d['W'], d['Q'], np.ones((d['W'].shape[0], 1))]).tocsr()
    Xc = np.concatenate([d['Xp'], d['sim'][..., None]], 2)
    return Phi, Xc


Phi, Xc = design(tr); Phd, Xcd = design(dv)
N, F = Phi.shape; P_ = Xc.shape[2]
w = weights(K); Y = tr['y']; cidx = tr['cidx']; rows = np.arange(N)[:, None]
isN = (cidx == C['classes'].index('NONE'))
lp = tr['lp']
reg = np.ones((F, 1)); reg[-1] = 0.  # no penalty on the code bias column


def scores(th, Ph, X, cid):
    t, Wm = th[:P_], th[P_:].reshape(F, NC)
    return X @ t + (Ph @ Wm)[np.arange(len(cid))[:, None], cid]


def lse(z):
    m = z.max(1, keepdims=True); return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def f(th):
    Wm = th[P_:].reshape(F, NC)
    z = scores(th, Phi, Xc, cidx)
    if mode == 'ce':
        Pz = np.exp(z - lse(z)[:, None])
        loss = w @ (lse(z) - (Y * z).sum(1)) / N
        G = w[:, None] * (Pz - Y) / N
    else:  # factorised PoE
        mN = (Y * isN).sum(1); mE = 1 - mN
        zc = np.where(isN, -np.inf, z); lc = lse(zc); zN = z[isN]
        two = np.stack([zN, lc], 1); l2w = lse(two)
        sN = np.exp(zN - l2w); sE = 1 - sN
        zb = np.where(isN, -np.inf, z + lam * lp); lb = lse(zb); q = np.exp(zb - lb[:, None])
        tE = np.where(isN, 0., Y); tE = tE / np.clip(tE.sum(1, keepdims=True), 1e-12, None)
        ce_code = np.where(mE > 0, lb - (tE * np.where(isN, 0, z + lam * lp)).sum(1), 0.)
        loss = w @ (l2w - mN * zN - mE * lc + mE * ce_code) / N
        pc = np.exp(zc - lc[:, None])  # softmax over codes (unbiased), 0 on NONE
        G = (sE - mE)[:, None] * pc + mE[:, None] * (q - tE)
        G[isN] = (sN - mN)
        G = w[:, None] * G / N
    loss += .5 * l2 * (reg * Wm ** 2).sum()
    gt = np.einsum('nk,nkp->p', G, Xc)
    R = np.zeros((N, NC)); np.add.at(R, (np.broadcast_to(rows, cidx.shape), cidx), G)
    gW = Phi.T @ R + l2 * reg * Wm
    return loss, np.concatenate([gt, np.asarray(gW).ravel()])


t0 = time.time()
r = minimize(f, np.zeros(P_ + F * NC), jac=True, method='L-BFGS-B', options=dict(maxiter=400))
th = r.x
zd = scores(th, Phd, Xcd, dv['cidx']); Pd = np.exp(zd - lse(zd)[:, None])
zt = scores(th, Phi, Xc, cidx)
tag = f'{mode}_K{K:g}_lam{lam:g}_l2{l2:g}'
ess = w[tr['err']].sum() ** 2 / (w[tr['err']] ** 2).sum()
info = dict(tag=tag, iters=int(r.nit), loss=float(r.fun), secs=round(time.time() - t0, 1), theta=th[:P_].round(4).tolist(),
            shortcut_at_n1=float(th[0] + th[1] * np.log(2)), ess_err=float(ess),
            train_top1_acc=float((zt.argmax(1) == Y.argmax(1)).mean()))
np.savez(f'{OUT}/fits/{tag}.npz', Pd=Pd, theta=th[:P_])
json.dump(info, open(f'{OUT}/fits/{tag}.json', 'w'))
print(json.dumps(info))
