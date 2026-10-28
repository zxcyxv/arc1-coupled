"""Float64: the proposal's circuit (window recursion, closed-form Kbar clamped at 0.25, E%-max omega) on a random lasso
problem. Checks the fixed point for Kbar_true >= 0.25, Kbar_true in (0, 0.25) and Kbar_true < 0."""
import numpy as np
rng = np.random.default_rng(1)
soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)
def lasso(G, S, lam, iters=200000):
    L = np.linalg.eigvalsh(G).max(); th = np.zeros_like(S); z = th.copy(); t = 1.0
    for _ in range(iters):
        new = soft(z - (z @ G - S) / L, lam / L); t2 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = new + (t - 1) / t2 * (new - th); th, t = new, t2
    return th
def circuit(G, S, lam, T, z, b, omega=0.5, kmin=0.25):
    g = np.sqrt(1 - abs(z) ** 2); Kbar_true = 1 + (b * g * z / (1 - z)).real; Kb = max(Kbar_true, kmin)
    th = np.zeros_like(S); E = np.zeros(S.shape, complex)
    for t in range(T):
        e = S - th @ G
        D = z * E; E = D + g * e
        Delta = e + (b * D).real
        u = th + Delta
        R = np.abs(soft(u, Kb * lam) - th).max(1, keepdims=True) / Kb
        th = soft(u, Kb * (lam + omega * R))
        if not np.isfinite(th).all(): return th, Kbar_true, Kb, 'diverged at %d' % t
    return th, Kbar_true, Kb, 'ok'
n, p, C = 40, 64, 13
phi = rng.standard_normal((n, p)); phi /= np.linalg.norm(phi, axis=1, keepdims=True)
tt = np.zeros((C, p))
for c in range(C): tt[c, rng.choice(p, 4, replace=False)] = rng.standard_normal(4)
Y = phi @ tt.T; G, S = phi.T @ phi / n, Y.T @ phi / n; lam = 1e-3
ref = lasso(G, S, lam)
beta = 0.85; gb = np.sqrt(1 - beta ** 2)
cases = {'default b=1/gamma (Kbar 6.67)': (beta + 0j, 1 / gb + 0j),
         'b scaled so Kbar_true=0.1': (beta + 0j, (0.1 - 1) / (gb * beta / (1 - beta)) + 0j),
         'b scaled so Kbar_true=-0.5': (beta + 0j, (-0.5 - 1) / (gb * beta / (1 - beta)) + 0j)}
for name, (z, b) in cases.items():
    th, kt, kb, st = circuit(G, S, lam, 20000, z, b)
    msg = st
    if st == 'ok':
        for lam_eff in (lam, lam * kb / kt if kt > 0 else np.nan):
            if np.isfinite(lam_eff):
                r2 = lasso(G, S, lam_eff, 50000); msg += f' | max|th - lasso({lam_eff:.2e})| {np.abs(th - r2).max():.2e}'
    print(f'{name}: Kbar_true {kt:.3f} used {kb:.3f}: {msg} | |th|max {np.nanmax(np.abs(th)):.3g}')
