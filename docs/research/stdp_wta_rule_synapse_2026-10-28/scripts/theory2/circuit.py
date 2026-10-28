"""Float64 checks of the competitive rule-synapse circuit (guarantee-first design).

theta [C,p]; drive e = S - theta G (delta rule: teacher x pre minus own read x pre); base map T(theta) = soft(theta + c0 e, c0 lam).
Circuit: theta+ = soft(theta + c0 (e + lobes), c0 (lam + omega R)),
  lobes = sum_r Re(b_r (Xi_r - gbar_r e)),  Xi_r <- z_r Xi_r + gamma_r e   (zero-area: vanish when e is constant)
  R[c] = max_j |T(theta)_cj - theta_cj|   (row residual of the base map: E%-max surplus that vanishes at fixed points)
"""
import numpy as np
rng = np.random.default_rng(0)
soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)

def lasso(G, S, lam, iters=400000, tol=1e-15):
    L = np.linalg.eigvalsh(G).max(); th = np.zeros_like(S); z = th.copy(); t = 1.0
    for _ in range(iters):
        new = soft(z - (z @ G - S) / L, lam / L); t2 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new + (t - 1) / t2 * (new - th)
        if np.abs(new - th).max() < tol: return new
        th, t = new, t2
    return th

def kkt(th, G, S, lam):
    g = S - th @ G; on = th != 0
    a = np.abs(g[on] - lam * np.sign(th[on])).max() if on.any() else 0.0
    b = max(0.0, (np.abs(g[~on]) - lam).max()) if (~on).any() else 0.0
    return max(a, b)

def circuit(G, S, lam, T, c0=1.0, omega=0.0, bands=(), hist=False):
    """bands: list of (z complex, b complex). Returns theta after T steps (and the trajectory if hist)."""
    th = np.zeros_like(S); Xi = [np.zeros(S.shape, complex) for _ in bands]; out = []
    for t in range(T):
        e = S - th @ G
        lob = np.zeros_like(S)
        for i, (z, b) in enumerate(bands):
            g = np.sqrt(1 - abs(z) ** 2)
            Xi[i] = z * Xi[i] + g * e
            lob += (b * (Xi[i] - g / (1 - z) * e)).real
        if omega:
            R = np.abs(soft(th + c0 * e, c0 * lam) - th).max(1, keepdims=True)
        else:
            R = 0.0
        th = soft(th + c0 * (e + lob), c0 * (lam + omega * R))
        if hist: out.append(th.copy())
    return (th, out) if hist else th

def fista(G, S, lam, T):
    th = np.zeros_like(S); z = th.copy(); t = 1.0
    for _ in range(T):
        new = soft(z - (z @ G - S), lam); t2 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new + (t - 1) / t2 * (new - th); th, t = new, t2
    return th

def iht(G, S, s, T, c0=1.0):
    th = np.zeros_like(S)
    for _ in range(T):
        u = th + c0 * (S - th @ G)
        keep = np.argsort(-np.abs(u), axis=1)[:, :s]
        new = np.zeros_like(u); np.put_along_axis(new, keep, np.take_along_axis(u, keep, 1), 1); th = new
    return th

def problem(n, p, C, s, nq=200):
    phi = unit(rng.standard_normal((n, p))); th = np.zeros((C, p))
    for c in range(C): th[c, rng.choice(p, s, replace=False)] = rng.standard_normal(s) * 2
    q = unit(rng.standard_normal((nq, p)))
    return phi, phi @ th.T, th, q

res = {}
# 1. fixed-point location: random zero-area lobes and omega in [0,1) all converge to lasso(lam)
P = []
for trial in range(10):
    phi, Y, tt, q = problem(40, 64, 13, 4)
    n = len(phi); G, S = phi.T @ phi / n, Y.T @ phi / n
    lam = 1e-3; ref = lasso(G, S, lam)
    bands = [(0.6 * np.exp(1j * rng.uniform(0, np.pi)), 0.3 * (rng.standard_normal() + 1j * rng.standard_normal())) for _ in range(2)]
    for omega in (0.0, 0.5, 0.9):
        for c0 in (1.0, 1.8):
            th = circuit(G, S, lam, 60000, c0=c0, omega=omega, bands=bands)
            P.append((np.abs(th - ref).max(), kkt(th, G, S, lam)))
P = np.array(P)
res['1_location_maxdiff_vs_lasso'] = float(P[:, 0].max()); res['1_location_max_kkt'] = float(P[:, 1].max())

# 2. transient: distance to the lasso solution and query argmax agreement with the truth after T steps
rows = {}
methods = dict(ista=dict(), ista_c0_1p8=dict(c0=1.8), wta05=dict(omega=0.5), wta08=dict(omega=0.8),
               nest_lobe=dict(bands=[(0.3 + 0j, -2.0 + 0j)]), wta05_c18=dict(omega=0.5, c0=1.8))
Ts = (8, 16, 32, 128)
acc = {m: {T: [] for T in Ts} for m in list(methods) + ['fista', 'iht', 'lasso']}
dist = {m: {T: [] for T in Ts} for m in list(methods) + ['fista']}
nnz8 = {m: [] for m in methods}
for trial in range(30):
    phi, Y, tt, q = problem(40, 64, 13, 4)
    n = len(phi); G, S = phi.T @ phi / n, Y.T @ phi / n
    lam = 1e-4; ref = lasso(G, S, lam)
    truth = (q @ tt.T).argmax(1)
    for m, kw in methods.items():
        th, h = circuit(G, S, lam, max(Ts), hist=True, **kw)
        nnz8[m].append((h[7] != 0).sum(1).mean())
        for T in Ts:
            dist[m][T].append(np.abs(h[T - 1] - ref).max() / np.abs(ref).max())
            acc[m][T].append(((q @ h[T - 1].T).argmax(1) == truth).mean())
    for T in Ts:
        f = fista(G, S, lam, T); dist['fista'][T].append(np.abs(f - ref).max() / np.abs(ref).max())
        acc['fista'][T].append(((q @ f.T).argmax(1) == truth).mean())
        i = iht(G, S, 4, T); acc['iht'][T].append(((q @ i.T).argmax(1) == truth).mean())
        acc['lasso'][T].append(((q @ ref.T).argmax(1) == truth).mean())
res['2_rel_dist_to_lasso'] = {m: {T: round(float(np.median(v)), 3) for T, v in d.items()} for m, d in dist.items()}
res['2_query_argmax_acc'] = {m: {T: round(float(np.mean(v)), 3) for T, v in d.items()} for m, d in acc.items()}
res['2_active_per_row_at_T8'] = {m: round(float(np.mean(v)), 1) for m, v in nnz8.items()}
import json; print(json.dumps(res, indent=1))
