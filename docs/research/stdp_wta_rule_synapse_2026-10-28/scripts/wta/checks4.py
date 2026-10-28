import numpy as np
from checks import unit, softmax, K
rng = np.random.default_rng(3)
n, p, rho = 200, 64, 0.9
phi = unit(rng.standard_normal((n, p))); Y = np.eye(K)[rng.integers(K, size=n)]
G = phi.T @ phi / n; S = Y.T @ phi / n; P = np.eye(K) - 1.0 / K
for scale in (1e-1, 1e-2, 1e-3):
    th = scale * rng.standard_normal((K, p))
    q = softmax(phi @ th.T); H = (rho * Y + (1 - rho) / K - q).T @ phi / n
    lin = P @ (rho * K * S - th @ G) / K
    print(f'drive scale {scale:g}: |H - (1/K) P (rho K S - theta G)| / |H| = {np.linalg.norm(H - lin) / np.linalg.norm(H):.2e}')
