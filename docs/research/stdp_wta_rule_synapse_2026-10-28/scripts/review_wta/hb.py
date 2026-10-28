import numpy as np
rng = np.random.default_rng(1)
K, rho = 11, 0.9
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def sm(a): a = a - a.max(-1, keepdims=True); e = np.exp(a); return e / e.sum(-1, keepdims=True)
def lse(a): m = a.max(1); return m + np.log(np.exp(a - m[:, None]).sum(1))
out = {0.0: [], 0.5: []}; nonmono = 0; blocks8 = {0.0: [], 0.5: []}
for trial in range(100):
    n, p = 60, 64
    base = unit(rng.standard_normal(p))
    A = unit(base + rng.uniform(0, 0.3) * rng.standard_normal((n, p)))
    lab = rng.integers(2, size=n) if trial % 2 else rng.integers(K, size=n)
    yt = rho * np.eye(K)[lab] + (1 - rho) / K
    J = lambda th: np.mean(lse(A @ th.T) - (yt * (A @ th.T)).sum(1))
    for kap in (0.0, 0.5):
        th = np.zeros((K, p)); E = np.zeros_like(th); Jp = J(th); Js = []
        for t in range(3000):
            H = (yt - sm(A @ th.T)).T @ A / n
            Jc = J(th); g = 1.0 if Jc <= Jp else 0.0; Jp = Jc; Js.append(Jc)
            E = H + g * kap * E; th = th + 2.0 * E
            if t == 7: blocks8[kap].append(Jc)
        out[kap].append(np.abs((yt - sm(A @ th.T)).T @ A / n).max())
        if kap == 0.0: nonmono += np.any(np.diff(Js) > 1e-12)
for k in out: print(f'kappa {k}: median residual {np.median(out[k]):.2e} max {np.max(out[k]):.2e} finite {np.all(np.isfinite(out[k]))}; J at block 8 median {np.median(blocks8[k]):.4f}')
print('plain GD (kappa 0) non-monotone cases:', nonmono)
