import numpy as np
from checks import unit, softmax, K
from checks2 import problem
import checks2
rng = np.random.default_rng(2); checks2.rng = rng

def head(phi, Y, T, mode, beta=1.0, kappa=0.0, rho=0.9, adaptive=False):
    n, p = phi.shape; th = np.zeros((K, p)); E = np.zeros_like(th); Cprev = -np.inf
    G = phi.T @ phi / n; S = Y.T @ phi / n; Yt = rho * Y + (1 - rho) / K
    lmax = np.linalg.eigvalsh(G).max() if adaptive else 1.0          # 1.0 = the tr G bound
    for t in range(T):
        if mode == 'lasso':
            th = th - (th @ G - S) / lmax; continue
        q = softmax(beta * phi @ th.T); C = (Yt * np.log(q)).sum(-1).mean()
        g = 1.0 if C >= Cprev else 0.0; Cprev = C
        H = (Yt - q).T @ phi / n
        E = H + g * kappa * E
        th = th + (2.0 / (beta * lmax)) * E                         # eta = 1 / L, L = beta lambda_max(G) / 2
    return th

cfgs = [('lasso', dict()), ('lasso', dict(adaptive=True))]
for beta in (1.0, 4.0, 11.0):
    for kap in (0.0, 0.5):
        cfgs.append(('wta', dict(beta=beta, kappa=kap)))
        cfgs.append(('wta', dict(beta=beta, kappa=kap, adaptive=True)))
for (n, p, nsit, noise) in [(120, 64, 40, 0.6), (300, 64, 60, 0.8)]:
    data = [problem(n, p, nsit, noise) for _ in range(20)]
    print(f'n={n} situations={nsit} noise={noise}')
    for m, kw in cfgs:
        row = []
        for T in (8, 30):
            a = []; d = []
            for phi, Y, phq, yq in data:
                th = head(phi, Y, T, m, **kw)
                b = kw.get('beta', 1.0)
                a.append((np.argmax(phq @ th.T, -1) == yq).mean()); d.append((np.argmax(phi @ th.T, -1) == Y.argmax(-1)).mean())
            row.append(f'T{T}: demo {np.mean(d):.3f} query {np.mean(a):.3f}')
        print(f'  {m:5s} {str(kw):55s} ' + '  '.join(row))
