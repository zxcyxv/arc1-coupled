"""Noiseless demos, finite lambda: closed form of the lasso fixed point under sign consistency
(theta_S = theta*_S - lam sign(theta*_S) G_SS^{-1}, zero off S) and the resulting query-read error."""
import numpy as np
exec(open('checks.py').read().split("out = {}")[0])
rng = np.random.default_rng(1)
rows = []
for trial in range(20):
    phi, Y, th_true = problem(n=60, p=24, C=1, s=3)
    G, S = stats(phi, Y)
    Sup = np.flatnonzero(th_true[0]); sg = np.sign(th_true[0, Sup])
    GSS_inv = np.linalg.inv(G[np.ix_(Sup, Sup)])
    irr = np.abs(G[np.ix_(np.setdiff1d(np.arange(24), Sup), Sup)] @ GSS_inv @ sg).max()
    lam = 1e-4
    closed = np.zeros(24); closed[Sup] = th_true[0, Sup] - lam * GSS_inv @ sg
    fp = lasso(G, S, lam)[0]
    rows.append((irr, np.abs(fp - closed).max(), np.abs(fp - th_true[0]).max()))
rows = np.array(rows)
ok = rows[:, 0] < 1
print('irrepresentable holds in', ok.sum(), 'of', len(rows))
print('max |lasso - closed form| where it holds:', rows[ok, 1].max())
print('max |lasso - closed form| where it fails:', rows[~ok, 1].max() if (~ok).any() else None)
print('max |lasso - truth| (shrinkage, lam=1e-4):', rows[:, 2].max())
