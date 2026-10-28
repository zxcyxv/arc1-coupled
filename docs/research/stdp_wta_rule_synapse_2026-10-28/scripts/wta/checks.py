"""float64 checks for the plateau-gated soft-WTA rule synapse (numpy only)."""
import numpy as np
rng = np.random.default_rng(0)
K = 11                                   # valid colour units (tokens 2..12)

def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def softmax(a):
    a = a - a.max(-1, keepdims=True); e = np.exp(a); return e / e.sum(-1, keepdims=True)

def problem(n=300, p=64, nsit=12, noise=0.35, nq=200):
    proto = unit(rng.standard_normal((nsit, p)))
    rule = rng.integers(K, size=nsit)
    s = rng.integers(nsit, size=n); sq = rng.integers(nsit, size=nq)
    phi = unit(proto[s] + noise * unit(rng.standard_normal((n, p))))
    phq = unit(proto[sq] + noise * unit(rng.standard_normal((nq, p))))
    Y = np.eye(K)[rule[s]]; return phi, Y, phq, rule[sq]

def H_unbalanced(th, phi, Y, Ap, Am, project=True):
    q = softmax(phi @ th.T)
    Hm = (Ap * Y - Am * q).T @ phi / len(phi)
    if project: Hm = Hm - Hm.mean(0, keepdims=True)          # heterosynaptic: colour-common part removed
    return Hm

def run(phi, Y, rho=0.9, eta=1.0, kappa=0.0, T=8, restart=True, mode='ce'):
    p = phi.shape[1]; th = np.zeros((K, p)); E = np.zeros_like(th); Cprev = -np.inf; n = len(phi)
    Yt = rho * Y + (1 - rho) / K
    for t in range(T):
        if mode == 'lasso':                                  # current head, lam = 0, step 1 (one-hot regression)
            G = phi.T @ phi / n; S = Y.T @ phi / n
            th = th - (th @ G - S); continue
        q = softmax(phi @ th.T)
        C = (Yt * np.log(q)).sum(-1).mean()                   # demo consistency (smoothed log-likelihood)
        g = 1.0 if (C >= Cprev or not restart) else 0.0       # dopamine gate (TD error >= 0)
        Cprev = C
        H = (Yt - q).T @ phi / n
        E = H + g * kappa * E
        th = th + eta * E
    return th

out = {}
# 1. unbalanced lobes (A+ < A-) + colour-common projection == smoothed-label CE fixed point
phi, Y, phq, yq = problem()
Ap, Am = 0.9, 1.0
th = np.zeros((K, phi.shape[1]))
for _ in range(40000): th = th + 1.0 * H_unbalanced(th, phi, Y, Ap, Am)
th2 = run(phi, Y, rho=Ap / Am, T=40000)
out['1_unbalanced_lobes_eq_label_smoothing_maxdiff'] = float(np.abs(th - th2).max())
q = softmax(phi @ th.T); Yt = 0.9 * Y + 0.1 / K
out['1_moment_residual'] = float(np.abs((Yt - q).T @ phi).max() / len(phi))
out['1_max_logit_margin'] = float(np.max(np.log(q.max(-1) / np.sort(q, -1)[:, -2])))
out['1_predicted_ceiling'] = float(np.log(1 + 0.9 * K / 0.1))
# 1b. without the projection the colour-common part drifts linearly
th3 = np.zeros_like(th)
for _ in range(2000): th3 = th3 + H_unbalanced(th3, phi, Y, Ap, Am, project=False)
out['1b_colour_common_norm_after_2000_unprojected'] = float(np.linalg.norm(th3.mean(0)))
# 2. Boehning curvature bound: L <= 1/2 lambda_max(G) <= 1/2
G = phi.T @ phi / len(phi)
out['2_lambda_max_G'] = float(np.linalg.eigvalsh(G).max())
# 3. momentum with dopamine restart: same fixed point, faster transient
ref = run(phi, Y, T=40000)
for kap in (0.0, 0.5, 0.8):
    a = run(phi, Y, kappa=kap, T=20000)
    out[f'3_kappa{kap}_fixed_point_diff'] = float(np.abs(a - ref).max())
    for T in (8, 30):
        b = run(phi, Y, kappa=kap, T=T)
        out[f'3_kappa{kap}_T{T}_dist'] = float(np.linalg.norm(b - ref) / np.linalg.norm(ref))
        out[f'3_kappa{kap}_T{T}_demo_acc'] = float((np.argmax(phi @ b.T, -1) == Y.argmax(-1)).mean())
        out[f'3_kappa{kap}_T{T}_query_acc'] = float((np.argmax(phq @ b.T, -1) == yq).mean())
# 3b. lasso head (lam 0, step 1) for comparison
for T in (8, 30):
    b = run(phi, Y, T=T, mode='lasso')
    out[f'3_lasso_T{T}_demo_acc'] = float((np.argmax(phi @ b.T, -1) == Y.argmax(-1)).mean())
    out[f'3_lasso_T{T}_query_acc'] = float((np.argmax(phq @ b.T, -1) == yq).mean())
# 4. cone condition: any consistent theta answers right on a conic combination of same-answer demo cells
ok = 0; tot = 0
for trial in range(200):
    th_r = rng.standard_normal((K, phi.shape[1]))
    pred = np.argmax(phi @ th_r.T, -1)
    for c in np.unique(pred)[:3]:
        idx = np.flatnonzero(pred == c)
        x = rng.random(len(idx)) @ phi[idx]                  # conic combination of cells this theta labels c
        tot += 1; ok += int(np.argmax(th_r @ x) == c)
out['4_cone_correct_fraction'] = ok / tot
for k, v in out.items(): print(f'{k:48s} {v:.6g}')
