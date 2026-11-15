"""Memoriser x reasoner fusion with a gate: the three modes of ARCHITECTURE_REFORM_2026.md section 4.2 (numpy only).

Notation (section 3). M is the memoriser (blend x ranker^0.5, rows sum to 1), sub its Sinkhorn-balanced form (the
submitted system), S the reasoner's rank shares, R = norm(S + eps), sent the rows whose reasoner answer is used
(submitted confidence below the threshold and an answer exists), g in [0, 1] the gate score, defined on sent rows.

  product  norm(M * R^(v*s)), the formula of scripts/fuse_vlm.py; balance afterwards with sinkhorn.
  soft     norm(M^(1 - beta*g*s) * R^(v*s)); beta = 0 is product.
  hard     sent rows with g >= 0.5 are ranked by the reasoner alone, every other row is sub unchanged.

A row that was not sent equals the memoriser in every mode. Nothing here reads a label except crossfit_gate, which
is given one to fit, and never fits a row's gate on rows of its own group (question).
"""
import numpy as np

FEATURES = ('confidence', 'memoriser_entropy', 'top_codes_agree', 'reasoner_top_new_error_code', 'question_seen',
            'memoriser_top_none', 'line_flagged_unequal')


def norm(P):
    """Rescale every row to sum to 1 (the same arithmetic and checks as k12.core.normalize)."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Expected a finite nonnegative probability matrix')
    if (P.sum(axis=1) <= 0).any():
        raise ValueError('Empty probability row')
    return P / P.sum(axis=1, keepdims=True)


def sinkhorn(P, target, lam, iters):
    """Balance the mean prediction toward target, then apply the fitted class offsets at strength lam.

    Identical maths to scripts/fuse_candidate.py::sinkhorn (a test compares the two on random input).
    """
    b = np.zeros(P.shape[1])
    for _ in range(iters):
        b += np.log(target / norm(P * np.exp(b)).mean(0))
    return norm(P * np.exp(lam * b))


def reasoner_probs(S, eps):
    """R = norm(S + eps): rank shares with a floor, so codes the reasoner did not name are not ruled out."""
    return norm(np.asarray(S, float) + eps)


def _flags(sent, n):
    sent = np.asarray(sent, bool)
    if sent.shape != (n,):
        raise ValueError('sent must be one flag per row')
    return sent


def product_fusion(M, R, sent, v, R0=None, v0=0.0):
    """norm(M * R0^(v0*s) * R^(v*s)). Rows with sent False are norm(M) exactly, whatever R holds there."""
    out = norm(M); sent = _flags(sent, len(out))
    if sent.any():
        F = np.asarray(M, float)[sent] * np.asarray(R, float)[sent] ** v
        if R0 is not None and v0:
            F = F * np.asarray(R0, float)[sent] ** v0
        out[sent] = norm(F)
    return out


def soft_fusion(M, R, sent, v, g, beta):
    """norm(M^(1 - beta*g*s) * R^(v*s)): the gate turns the memoriser down, row by row. g counts only where sent.

    beta = 0 is product_fusion; beta = 1 with g = 1 leaves the reasoner alone on that row.
    """
    out = norm(M); sent = _flags(sent, len(out))
    g = np.where(sent, np.asarray(g, float), 0.)
    if ((g < 0) | (g > 1)).any() or not 0 <= beta <= 1:
        raise ValueError('gate scores and beta must lie in [0, 1]')
    if sent.any():
        a = (1. - beta * g[sent])[:, None]
        out[sent] = norm(np.asarray(M, float)[sent] ** a * np.asarray(R, float)[sent] ** v)
    return out


def hard_route(sub, R, sent, g, tie_break=None, delta=1e-3):
    """One branch per row: sent rows with g >= 0.5 take the reasoner alone, every other row is sub exactly.

    On a routed row the order is the reasoner's; codes it scores equally (everything outside its three) are ordered
    by tie_break (the memoriser M or sub; default sub). This is done by multiplying R by 1 + delta * u, with u in
    (0, 1] falling with the code's rank under tie_break, so any gap in R wider than a factor 1 + delta is kept.
    No balancing is applied to routed rows: sub is already balanced, and R is used as it is.
    """
    if sub is None or g is None:
        raise ValueError('hard routing needs sub (the balanced memoriser) and a gate score per row')
    out = np.array(sub, dtype=np.float64); sent = _flags(sent, len(out))
    T = out if tie_break is None else np.asarray(tie_break, float)
    routed = sent & (np.where(sent, np.asarray(g, float), 0.) >= .5)
    if routed.any():
        k = out.shape[1]
        order = np.argsort(-T[routed], axis=1, kind='stable')  # best code first, class order on ties, as topk does
        rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(k)[None, :].repeat(len(order), 0), axis=1)
        out[routed] = norm(np.asarray(R, float)[routed] * (1. + delta * (k - rank) / k))
    return out


def fuse(M, R, sent, mode='product', v=1., lam=0., target=None, iters=30, g=None, beta=0., sub=None, tie_break=None):
    """The three modes behind one call. product and soft are balanced over all rows when lam > 0; hard is not."""
    if mode == 'hard':
        return hard_route(sub, R, sent, g, tie_break)
    if mode not in ('product', 'soft'):
        raise ValueError(f'Unknown mode {mode!r}')
    F = product_fusion(M, R, sent, v) if mode == 'product' else soft_fusion(M, R, sent, v, g, beta)
    return sinkhorn(F, target, lam, iters) if lam else F


def code_sets(questions, codes):
    """question -> set of codes seen on it (C_fit(q) when given the fit rows)."""
    out = {}
    for q, c in zip(questions, codes):
        out.setdefault(q, set()).add(c)
    return out


def gate_features(sub, M, S, rows, qid, fit_codes, classes, flagged=None):
    """The seven gate features of section 4.2.2 for the given rows (a boolean mask or row numbers), in row order.

    confidence c = max sub; entropy of M; the two top codes agree; the reasoner's top code is an error code not seen
    on this question in the fit rows; the question is in the fit rows; the memoriser's top code is NONE; a line was
    flagged unequal (flagged, one value per row of sub; zeros when the reasoner does not report it). The memoriser's
    top code is the top code of sub, the answer the memoriser branch would give. All are test-time facts.
    The gate exists only for rows with a reasoner answer, so a row whose shares are all zero raises ValueError.
    """
    sub, M, S = np.asarray(sub, float), norm(M), np.asarray(S, float)
    rows = np.asarray(rows)
    idx = np.flatnonzero(rows) if rows.dtype == bool else rows.astype(int)
    if (S[idx].sum(1) <= 0).any():
        raise ValueError('gate features are defined only for rows with a reasoner answer')
    classes = np.asarray(classes); none = int(np.flatnonzero(classes == 'NONE')[0])
    top_m, top_r = sub[idx].argmax(1), S[idx].argmax(1)
    q = np.asarray(qid)[idx]
    Mi = M[idx]
    entropy = -(Mi * np.log(np.where(Mi > 0, Mi, 1.))).sum(1)
    new_code = np.array([k != none and classes[k] not in fit_codes.get(x, ()) for k, x in zip(top_r, q)], bool)
    seen_q = np.array([x in fit_codes for x in q], bool)
    flag = np.zeros(len(sub)) if flagged is None else np.asarray(flagged, float)
    return np.column_stack([sub[idx].max(1), entropy, top_m == top_r, new_code, seen_q, top_m == none, flag[idx]]).astype(float)


def _sigmoid(a):
    return 1. / (1. + np.exp(-np.clip(a, -35., 35.)))


def fit_logistic(Z, label, l2=1.0, iters=100):
    """L2 logistic regression by Newton steps on features standardised with this sample's mean and deviation.

    Minimises the summed log loss + l2/2 * |weights|^2 (the intercept is not penalised). Returns (mean, deviation,
    theta) for predict_logistic. A feature that is constant here gets weight exactly 0. If only one class is present
    the model is the smoothed base rate.
    """
    Z, t = np.asarray(Z, float), np.asarray(label, float)
    mu, sd = Z.mean(0), Z.std(0); sd = np.where(sd > 0, sd, 1.)
    X = np.column_stack([np.ones(len(Z)), (Z - mu) / sd])
    pen = np.full(X.shape[1], float(l2)); pen[0] = 0.
    theta = np.zeros(X.shape[1])
    if t.min() == t.max():
        p = (t.sum() + .5) / (len(t) + 1.); theta[0] = np.log(p / (1. - p))
        return mu, sd, theta
    loss = lambda th: float(np.logaddexp(0., X @ th).sum() - t @ (X @ th) + .5 * (pen * th * th).sum())
    for _ in range(iters):
        p = _sigmoid(X @ theta)
        grad = X.T @ (p - t) + pen * theta
        H = (X * (p * (1. - p))[:, None]).T @ X + np.diag(pen + 1e-10)
        step = np.linalg.solve(H, grad)
        scale, base = 1., loss(theta)
        while loss(theta - scale * step) > base + 1e-12 and scale > 1e-4:  # damp a step that overshoots
            scale /= 2.
        theta = theta - scale * step
        if np.abs(scale * step).max() < 1e-10:
            break
    return mu, sd, theta


def predict_logistic(model, Z):
    mu, sd, theta = model
    return _sigmoid(theta[0] + ((np.asarray(Z, float) - mu) / sd) @ theta[1:])


def group_folds(groups, n_folds=5, seed=7):
    """Fold number per row with whole groups kept together: the distinct groups, in order of first appearance, are
    shuffled with the seed and dealt round-robin. On the dev question ids this is the split scripts/fuse_vlm.py uses."""
    distinct = np.array(list(dict.fromkeys(np.asarray(groups).tolist())), dtype=object)
    np.random.default_rng(seed).shuffle(distinct)
    fold = {q: i % n_folds for i, q in enumerate(distinct)}
    return np.array([fold[q] for q in np.asarray(groups).tolist()], int)


def crossfit_gate(Z, label, groups, n_folds=5, seed=7, l2=1.0, folds=None):
    """Out-of-fold gate scores: each row is scored by a logistic regression fitted on the other folds only.

    Folds keep groups (questions) whole, so no row's gate is fitted on its own question; features are standardised
    inside each training fold. folds (one number per row) replaces the built-in split, for example to reuse the
    fusion's question folds; it must not split a group.
    """
    Z, t, groups = np.asarray(Z, float), np.asarray(label, float), np.asarray(groups)
    fold = group_folds(groups, n_folds, seed) if folds is None else np.asarray(folds, int)
    if len({(a, b) for a, b in zip(groups.tolist(), fold.tolist())}) != len(set(groups.tolist())):
        raise ValueError('a group is split across folds')
    out = np.full(len(Z), np.nan)
    for f in np.unique(fold):
        held = fold == f
        if held.all():
            raise ValueError('every row is in one fold; nothing to fit on')
        out[held] = predict_logistic(fit_logistic(Z[~held], t[~held], l2), Z[held])
    return out
