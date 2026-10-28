import numpy as np
rng = np.random.default_rng(0)
K, rho = 11, 0.9
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def sm(a): a = a - a.max(-1, keepdims=True); e = np.exp(a); return e / e.sum(-1, keepdims=True)
P = np.eye(K) - 1.0 / K

# 1. Bohning: diag q - q q^T <= (1/2) P on random and adversarial q
worst = 0
for _ in range(20000):
    q = rng.dirichlet(rng.uniform(0.05, 2, K) * np.ones(K))
    M = 0.5 * P - (np.diag(q) - np.outer(q, q))
    worst = min(worst, np.linalg.eigvalsh(M).min())
q = np.zeros(K); q[:2] = 0.5
print('1. Bohning min eig of (1/2)P - Hess over 20k q:', worst, '| two-way tie eig:', np.linalg.eigvalsh(0.5 * P - (np.diag(q) - np.outer(q, q))).min())

# 2. small-signal WTA step (eta=2) vs the lambda=0 lasso step: coefficients and fixed points
n, p = 300, 64
phi = unit(rng.standard_normal((n, p))); lab = rng.integers(K, size=n); Y = np.eye(K)[lab]
G = phi.T @ phi / n; S = Y.T @ phi / n
th0 = 1e-3 * rng.standard_normal((K, p)); th0 = P @ th0
yt = rho * Y + (1 - rho) / K
H = (yt - sm(phi @ th0.T)).T @ phi / n
wta_step = 2 * H; lasso_step = S - th0 @ G
print('2. |2H - (S - theta G)|/|S - theta G| =', np.linalg.norm(wta_step - lasso_step) / np.linalg.norm(lasso_step),
      '| best scalar fit c:', (wta_step * lasso_step).sum() / (lasso_step ** 2).sum())
# linear-regime fixed point: theta G = rho K P S  -> rho K times the centred lasso(lam=0) solution
th_l = np.linalg.solve(G.T, S.T).T
th_w = th0.copy()
for t in range(20000):
    th_w = th_w + 2 * ((yt - sm(phi @ th_w.T)).T @ phi / n)
r_l = phi @ th_l.T; r_w = phi @ th_w.T
print('   converged WTA read margin (mean max-minus-2nd on demo cells):', np.mean(np.sort(r_w, 1)[:, -1] - np.sort(r_w, 1)[:, -2]),
      '| lasso read margin:', np.mean(np.sort(r_l, 1)[:, -1] - np.sort(r_l, 1)[:, -2]))
print('   demo argmax agreement WTA vs lasso:', np.mean(r_w.argmax(1) == r_l.argmax(1)), ' (WTA train acc', np.mean(r_w.argmax(1)==lab), ', lasso', np.mean(r_l.argmax(1)==lab),')')

# 3. underdetermined regime: WTA fixed point = min-norm interpolant of the smoothed logit targets (implicit L2), no max-ent selection over theta
n2, p2 = 40, 200
A = unit(rng.standard_normal((n2, p2))); lab2 = rng.integers(K, size=n2); Y2 = np.eye(K)[lab2]; yt2 = rho * Y2 + (1 - rho) / K
th = np.zeros((K, p2))
L = 0.5 * np.linalg.eigvalsh(A.T @ A / n2).max()
for t in range(200000):
    g = (yt2 - sm(A @ th.T)).T @ A / n2
    th = th + (1 / L) * g
    if np.abs(g).max() < 1e-12: break
tgt = np.log(yt2); tgt = tgt - tgt.mean(1, keepdims=True)
th_mn = (np.linalg.pinv(A) @ tgt).T
print('3. underdetermined: |theta_GD - theta_minnorm(log yt)|/|theta_mn| =', np.linalg.norm(th - th_mn) / np.linalg.norm(th_mn), 'iters', t)

# 4. heavy-ball eta=2 (=1/L bound), kappa=0.5, gate = 1[J did not increase]; hardest case: near-collinear phi, tied labels
fails = 0; worst_res = 0
for trial in range(200):
    n3, p3 = 60, 64
    base = unit(rng.standard_normal(p3))
    A3 = unit(base + rng.uniform(0, 0.3) * rng.standard_normal((n3, p3)))   # lambda_max(G) close to tr G = 1
    lab3 = rng.integers(2, size=n3) if trial % 2 else rng.integers(K, size=n3)
    Y3 = np.eye(K)[lab3]; yt3 = rho * Y3 + (1 - rho) / K
    J = lambda th: np.mean(np.log(np.exp(A3 @ th.T - (A3 @ th.T).max(1, keepdims=True)).sum(1)) + (A3 @ th.T).max(1) - (yt3 * (A3 @ th.T)).sum(1))
    th = np.zeros((K, p3)); E = np.zeros_like(th); Jp = J(th)
    for t in range(3000):
        Hh = (yt3 - sm(A3 @ th.T)).T @ A3 / n3
        Jc = J(th); g = 1.0 if Jc <= Jp + 1e-15 else 0.0; Jp = Jc
        E = Hh + g * 0.5 * E; th = th + 2.0 * E
    res = np.abs((yt3 - sm(A3 @ th.T)).T @ A3 / n3).max()
    worst_res = max(worst_res, res); fails += (not np.isfinite(res)) or res > 1e-6
print('4. heavy-ball + restart, 200 adversarial cases: fails', fails, 'worst moment residual', worst_res)
