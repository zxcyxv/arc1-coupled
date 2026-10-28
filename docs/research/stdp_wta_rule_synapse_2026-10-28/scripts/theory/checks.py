"""Numerical checks of the fixed-point propositions for the rule synapse (float64, numpy only).

Notation as in arc/model.py: phi [n, p] unit rows (demo output cells), targets Y [n, C],
G = phi^T phi / n  (p x p),  S = Y^T phi / n  (C x p),  one ISTA step  theta <- soft(theta - (theta G - S), lam).
"""
import numpy as np

rng = np.random.default_rng(0)
soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def lasso(G, S, lam, w=None, iters=200000, tol=1e-14):
    """FISTA to high precision; w: per-feature threshold weights (lam_j = lam * w_j)."""
    L = np.linalg.eigvalsh(G).max()
    th = np.zeros_like(S); z = th.copy(); t = 1.0
    lj = lam * (np.ones(G.shape[0]) if w is None else w)
    for _ in range(iters):
        new = soft(z - (z @ G - S) / L, lj / L)
        t2 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new + (t - 1) / t2 * (new - th)
        if np.abs(new - th).max() < tol:
            th = new; break
        th, t = new, t2
    return th


def kkt(th, G, S, lam):
    g = S - th @ G
    on = th != 0
    r_on = np.abs(g[on] - lam * np.sign(th[on])).max() if on.any() else 0.0
    r_off = max(0.0, (np.abs(g[~on]) - lam).max()) if (~on).any() else 0.0
    return max(r_on, r_off)


def problem(n=40, p=24, C=5, s=3, noiseless=True):
    phi = unit(rng.standard_normal((n, p)))
    th_true = np.zeros((C, p))
    for c in range(C):
        th_true[c, rng.choice(p, s, replace=False)] = rng.standard_normal(s) * 2
    Y = phi @ th_true.T
    if not noiseless:
        Y = Y + 0.05 * rng.standard_normal(Y.shape)
    return phi, Y, th_true


def stats(phi, Y):
    n = phi.shape[0]
    return phi.T @ phi / n, Y.T @ phi / n


out = {}

# 1. unit-step ISTA == forward-Euler LCA with dt / tau = 1 -------------------------------------------------------
phi, Y, _ = problem()
G, S = stats(phi, Y)
lam = 1e-3
assert np.linalg.eigvalsh(G).max() <= 1 + 1e-12            # lambda_max(G) <= tr G = 1
th = np.zeros_like(S); u = np.zeros_like(S); thl = soft(u, lam)
diff = 0.0
for t in range(500):
    th = soft(th - (th @ G - S), lam)                       # the rule head's step
    u = u + 1.0 * (S - u - thl @ (G - np.eye(len(G))))     # LCA, Euler, dt/tau = 1
    thl = soft(u, lam)
    diff = max(diff, np.abs(th - thl).max())
out['1_ista_equals_lca_dt1_maxdiff'] = diff

# 2. leaky LCA (alpha < 1) converges to the same lasso minimiser --------------------------------------------------
ref = lasso(G, S, lam)
res = {}
for alpha in (0.2, 0.5, 1.0):
    u = np.zeros_like(S)
    for t in range(200000):
        a = soft(u, lam)
        u_new = u + alpha * (S - u - a @ (G - np.eye(len(G))))
        if np.abs(u_new - u).max() < 1e-15:
            break
        u = u_new
    a = soft(u, lam)
    res[alpha] = (np.abs(a - ref).max(), kkt(a, G, S, lam), t)
out['2_leaky_lca_vs_lasso'] = {k: dict(maxdiff=v[0], kkt=v[1], iters=v[2]) for k, v in res.items()}
out['2_ref_kkt'] = kkt(ref, G, S, lam)


# 3. two-sided kernels on the Hebbian and anti-Hebbian terms, perception converging geometrically ---------------
def kernel_run(kh, ka, T=6000, rho=0.97, lam=1e-3, phi_star=None, Y=None):
    """kh, ka: dicts {0: k0, +tau: K+(tau) (pre leads), -tau: K-(tau) (post leads)}.
    Hebbian term  sum_tau kh(tau) * post_y(t or t-tau) pre(t-tau or t);  post_y = Y constant (teacher).
    anti-Hebbian  sum_tau ka(tau) * rhat(.) pre(.),  rhat_t = theta_t phi_t (the synapse's own read)."""
    n, p = phi_star.shape
    drift = unit(rng.standard_normal((n, p)))
    phis, ths = [], []
    th = np.zeros((Y.shape[1], p))
    lags = max(abs(k) for k in list(kh) + list(ka))
    for t in range(T):
        ph = unit(phi_star + rho ** t * drift)            # perception converging to phi_star
        phis.append(ph); ths.append(th)
        H = np.zeros_like(th); A = np.zeros_like(th)
        for tau, k in kh.items():
            if t - abs(tau) < 0:
                continue
            pre = phis[t - tau] if tau > 0 else ph       # tau > 0: pre at t - tau, post (Y) at t
            H += k * (Y.T @ pre) / n                     # (teacher constant: post time does not matter)
        for tau, k in ka.items():
            if t - abs(tau) < 0:
                continue
            if tau > 0:   # pre leads: post rhat at t, pre at t - tau
                A += k * ((ph @ th.T).T @ phis[t - tau]) / n
            elif tau < 0:  # post leads: post rhat at t - |tau|, pre at t
                A += k * ((phis[t + tau] @ ths[t + tau].T).T @ ph) / n
            else:
                A += k * ((ph @ th.T).T @ ph) / n
        th = soft(th + H - A, lam)
    return th, phis[-1]


phi_s, Y, _ = problem(n=40, p=24, C=5)
kh = {0: 0.2, 1: 0.5, 2: 0.25, 3: 0.1, -1: -0.15, -2: 0.1}          # odd part present, DC = 1.0
ka = {0: 0.3, 1: 0.1, 2: -0.05, -1: 0.4, -2: 0.15, -3: -0.1}        # different shape, DC = 0.8
kbh, kba = sum(kh.values()), sum(ka.values())
th_k, ph_end = kernel_run(kh, ka, phi_star=phi_s, Y=Y)
G, S = stats(phi_s, Y)
lam = 1e-3
pred = (kbh / kba) * lasso(G, S, lam / kbh)
out['3_kernel_fixed_point'] = dict(DC_hebb=kbh, DC_anti=kba, maxdiff_vs_scaled_lasso=np.abs(th_k - pred).max(),
                                   maxdiff_vs_plain_lasso=np.abs(th_k - lasso(G, S, lam)).max(),
                                   argmax_agree_on_cells=float(np.mean((phi_s @ th_k.T).argmax(1)
                                                                       == (phi_s @ lasso(G, S, lam).T).argmax(1))))

# 3b. same, transient differs: distance to the fixed point after 30 steps, delta kernel vs the two-sided kernel
def dist_after(kh, ka, T):
    th, _ = kernel_run(kh, ka, T=T, phi_star=phi_s, Y=Y)
    return np.abs(th - (sum(kh.values()) / sum(ka.values())) * lasso(G, S, 1e-3 / sum(kh.values()))).max()
out['3b_transient_T30'] = dict(delta_kernel=dist_after({0: 1.0}, {0: 1.0}, 30), two_sided=dist_after(kh, ka, 30))

# 4. hard threshold, noiseless demos: many fixed points (selection moves into the dynamics) ---------------------
phi, Y, th_true = problem(n=16, p=40, C=1, s=3)
G, S = stats(phi, Y)
mu = 1e-3
fps = []
for init in range(6):
    th = rng.standard_normal(S.shape) * 2 if init else np.zeros_like(S)
    for t in range(200000):
        new = np.where(np.abs(th - (th @ G - S)) > mu, th - (th @ G - S), 0.0)
        if np.abs(new - th).max() < 1e-14:
            break
        th = new
    fps.append(th)
hard = [dict(nnz=int((f != 0).sum()), fit_resid=float(np.abs(phi @ f.T - Y).max()),
             is_fixed=bool(np.abs(np.where(np.abs(f - (f @ G - S)) > mu, f - (f @ G - S), 0) - f).max() < 1e-10),
             dist_true=float(np.abs(f - th_true).max())) for f in fps]
softs = []
for init in range(4):
    th = rng.standard_normal(S.shape) * 2 if init else np.zeros_like(S)
    for t in range(400000):
        new = soft(th - (th @ G - S), mu)
        if np.abs(new - th).max() < 1e-15:
            break
        th = new
    softs.append(th)
out['4_hard_threshold_fixed_points'] = hard
out['4_soft_threshold_spread_over_inits'] = float(max(np.abs(a - softs[0]).max() for a in softs))
out['4_soft_dist_true'] = float(np.abs(softs[0] - th_true).max())

# 5. k-WTA (keep top-s |u|) = IHT: recovery of the s-sparse rule from theta = 0, against lasso(lam -> 0) ---------
def iht_topk(G, S, s, iters=5000):
    L = np.linalg.eigvalsh(G).max()
    th = np.zeros_like(S)
    for _ in range(iters):
        u = th - (th @ G - S) / L
        keep = np.argsort(-np.abs(u), axis=1)[:, :s]
        new = np.zeros_like(u); np.put_along_axis(new, keep, np.take_along_axis(u, keep, 1), 1)
        th = new
    return th

rec = {}
for n in (10, 14, 20, 30):
    ok_i = ok_l = 0
    trials = 40
    for _ in range(trials):
        phi, Y, th_true = problem(n=n, p=64, C=1, s=4)
        G, S = stats(phi, Y)
        ok_i += np.abs(iht_topk(G, S, 4) - th_true).max() < 1e-6
        ok_l += np.abs(lasso(G, S, 1e-7, iters=20000) - th_true).max() < 1e-3
    rec[n] = dict(iht_topk=ok_i / trials, lasso_lam0=ok_l / trials)
out['5_recovery_p64_s4'] = rec

# 6. teacher onset with a constant pre: the post-leading lobe's first moment produces the delta rule ------------
p, C = 8, 3
ph = unit(rng.standard_normal(p))
rhat, y = rng.standard_normal(C), np.eye(C)[0]
Kp = {1: 0.6, 2: 0.3, 3: 0.1}; Km = {1: -0.5, 2: -0.3, 3: -0.2}; K0 = 0.05
Tn, t0 = 40, 20
post = [rhat if t < t0 else y for t in range(Tn)]
def S_at(t, post):
    s = K0 * np.outer(post[t], ph)
    for tau, k in Kp.items():
        if t - tau >= 0: s += k * np.outer(post[t], ph)
    for tau, k in Km.items():
        if t - tau >= 0: s += k * np.outer(post[t - tau], ph)
    return s
win = range(t0, t0 + 3)                               # the transition blocks
Kbar = K0 + sum(Kp.values()) + sum(Km.values())
M1m = sum(tau * k for tau, k in Km.items())
trans = sum(S_at(t, post) for t in win) - len(win) * Kbar * np.outer(y, ph)
out['6_onset_transition_minus_pred'] = float(np.abs(trans - (-M1m) * np.outer(y - rhat, ph)).max())
post2 = [rhat if (t < t0 or t >= t0 + 10) else y for t in range(Tn)]   # clamp, then release
tot = sum(S_at(t, post2) for t in range(3, Tn)) - Kbar * sum(np.outer(post2[t], ph) for t in range(3, Tn))
out['6_clamp_release_cycle_residual'] = float(np.abs(tot).max())

# 7. per-feature pre-side DC gains d_j on both terms -> weighted lasso lam_j = lam / d_j -------------------------
phi, Y, _ = problem(n=40, p=24, C=5)
G, S = stats(phi, Y)
d = rng.uniform(0.3, 1.0, 24)
th = np.zeros_like(S)
for t in range(300000):
    new = soft(th - (th @ G - S) * d, 1e-3)
    if np.abs(new - th).max() < 1e-15:
        break
    th = new
out['7_weighted_lasso_maxdiff'] = float(np.abs(th - lasso(G, S, 1e-3, w=1 / d)).max())

# 8. post-side DC gains: scalar keeps argmax, colour-dependent gains change it ----------------------------------
phi, Y, _ = problem(n=60, p=24, C=5)
G, S = stats(phi, Y)
base = lasso(G, S, 1e-4)
def post_gain_fp(dy, dr, lam=1e-4):
    th = np.zeros_like(S)
    for t in range(300000):
        new = soft(th + dy[:, None] * S - dr[:, None] * (th @ G), lam)
        if np.abs(new - th).max() < 1e-15:
            break
        th = new
    return th
q = unit(rng.standard_normal((500, 24)))
agree = lambda a: float(np.mean((q @ a.T).argmax(1) == (q @ base.T).argmax(1)))
out['8_argmax_agree_scalar_gain'] = agree(post_gain_fp(np.full(5, 0.7), np.full(5, 0.9)))
out['8_argmax_agree_colour_gains'] = agree(post_gain_fp(np.array([0.3, 1, 0.5, 0.9, 0.6]), np.full(5, 0.9)))

# 9. perceptual gating: mixture of sparse experts == one lasso on the product code g (x) phi ---------------------
M = 3
phi, Y, _ = problem(n=50, p=12, C=4)
g = np.exp(rng.standard_normal((50, M))); g /= g.sum(1, keepdims=True)
psi = (g[:, :, None] * phi[:, None, :]).reshape(50, M * 12)
assert (np.linalg.norm(psi, axis=1) <= 1 + 1e-12).all()
Gp, Sp = stats(psi, Y)
Th = lasso(Gp, Sp, 1e-3)
r_lift = psi @ Th.T
r_mix = sum(g[:, m:m + 1] * (phi @ Th[:, m * 12:(m + 1) * 12].T) for m in range(M))
out['9_lifted_equals_mixture'] = float(np.abs(r_lift - r_mix).max())

# 10. two rectified populations with uniform depression (Dale split) == lasso ------------------------------------
phi, Y, _ = problem(n=40, p=24, C=5)
G, S = stats(phi, Y)
tp = np.zeros_like(S); tm = np.zeros_like(S)
for t in range(300000):
    g_ = (tp - tm) @ G - S
    ntp = np.maximum(tp - g_ - 1e-3, 0); ntm = np.maximum(tm + g_ - 1e-3, 0)
    if max(np.abs(ntp - tp).max(), np.abs(ntm - tm).max()) < 1e-15:
        break
    tp, tm = ntp, ntm
out['10_dale_split_maxdiff'] = float(np.abs((tp - tm) - lasso(G, S, 1e-3)).max())
out['10_dale_overlap'] = float((np.minimum(tp, tm) > 0).sum())

# 11. a cycling perception orbit: the odd lobe adds a skew-symmetric coupling; the limit is not the lasso
#     of the time-averaged statistics (two-timescale: small step on theta) --------------------------------------
n, p, C = 30, 10, 3
base_phi = rng.standard_normal((n, p))
Rplane = rng.standard_normal((n, 2))
def phi_cyc(t, period=12):
    a = 2 * np.pi * t / period
    x = base_phi.copy(); x[:, 0] += 2 * Rplane[:, 0] * np.cos(a); x[:, 1] += 2 * Rplane[:, 1] * np.sin(a)
    return unit(x)
Yc = rng.standard_normal((n, C))
Kodd = {0: 0.5, 1: 0.6, 2: 0.2, -1: -0.2, -2: -0.1}           # pre-leading > post-leading, DC = 1.0
Keven = {0: 0.5, 1: 0.2, 2: 0.05, -1: 0.2, -2: 0.05}           # its even part (same DC)
P = 12
def run_cycle(K, T=60000, eta=0.005, lam=1e-3):
    th = np.zeros((C, p)); hp, ht = [], []
    acc = np.zeros_like(th); cnt = 0
    for t in range(T):
        ph = phi_cyc(t); hp.append(ph); ht.append(th)
        H = np.zeros_like(th); A = np.zeros_like(th)
        for tau, k in K.items():
            if t - abs(tau) < 0: continue
            if tau > 0:
                H += k * (Yc.T @ hp[t - tau]) / n; A += k * (th @ ph.T @ hp[t - tau]) / n
            elif tau < 0:
                H += k * (Yc.T @ ph) / n; A += k * (ht[t + tau] @ hp[t + tau].T @ ph) / n
            else:
                H += k * (Yc.T @ ph) / n; A += k * (th @ ph.T @ ph) / n
        th = soft(th + eta * (H - A), eta * lam)
        if t >= T - 10 * P: acc += th; cnt += 1
    return acc / cnt
Clag = lambda tau: sum(phi_cyc(t).T @ phi_cyc(t - tau) for t in range(P)) / (P * n)   # E[phi_t^T phi_{t-tau}] / n
Sbar = sum(Yc.T @ phi_cyc(t) for t in range(P)) / (P * n)
def GK(K):
    return sum(k * (Clag(tau) if tau >= 0 else Clag(-tau).T) for tau, k in K.items())
def vi_solve(Gk, Sk, lam=1e-3, eta=0.2, iters=400000):
    th = np.zeros((C, p))
    for _ in range(iters):
        new = soft(th - eta * (th @ Gk - Sk), eta * lam)
        if np.abs(new - th).max() < 1e-14: break
        th = new
    return th
res11 = {}
for name, K in (('odd', Kodd), ('even', Keven)):
    Gk = GK(K)
    lim = run_cycle(K)
    res11[name] = dict(skew_norm=float(np.abs(Gk - Gk.T).max()),
                       limit_vs_averaged_VI=float(np.abs(lim - vi_solve(Gk, sum(K.values()) * Sbar)).max()),
                       limit_norm=float(np.abs(lim).max()))
    res11[name + '_lim'] = lim
res11['odd_vs_even_limits'] = float(np.abs(res11['odd_lim'] - res11['even_lim']).max())
del res11['odd_lim'], res11['even_lim']
out['11_cycling_orbit'] = res11

import json
print(json.dumps(out, indent=1, default=float))
