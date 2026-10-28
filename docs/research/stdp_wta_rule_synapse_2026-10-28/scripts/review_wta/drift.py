"""(1) finite-T certificate identity: Theta_T = Theta* (I - (I - eta G)^T) when S = Theta* G.
(2) warm start with drifting codes: Theta keeps components in the null space of the current G (they never decay),
so the limit is not the min-norm fit of the final code and the span certificate |(I - P_D) psi*| no longer bounds the error.
(3) STDP reading with constant pre: sum_t sum_tau K(tau) post(t) pre(t - tau) over an alternating predict/observe train."""
import numpy as np
rng = np.random.default_rng(0)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
n, p, V = 30, 64, 13
# (1)
X = unit(rng.standard_normal((n, p))); Ts = rng.standard_normal((V, p)); G = X.T @ X / n; S = Ts @ G
eta = 0.9 / np.linalg.eigvalsh(G).max(); th = np.zeros((V, p))
for _ in range(17): th = th - eta * (th @ G - S)
pred = Ts @ (np.eye(p) - np.linalg.matrix_power(np.eye(p) - eta * G, 17))
print('(1) |Theta_T - Theta*(I-(I-eta G)^T)| =', np.abs(th - pred).max())
# (2) codes drift: block t uses X_t = unit(X0 + drift_t), Theta* fixed; warm start over 8 blocks x 4 steps
X0 = rng.standard_normal((n, p)); D = rng.standard_normal((n, p)); th = np.zeros((V, p))
for t in range(8):
    Xt = unit(X0 + 1.5 * (1 - t / 7) * D)              # early blocks far from the final code
    Gt = Xt.T @ Xt / n; St = Ts @ Gt; e = 0.9 / np.linalg.eigvalsh(Gt).max()
    for _ in range(4): th = th - e * (th @ Gt - St)
U, s, Vt = np.linalg.svd(Xt, full_matrices=False); P = Vt.T @ Vt
mn = Ts @ P
qc = rng.standard_normal(p); qc /= np.linalg.norm(qc)
print('(2) |Theta (I-P_final)| =', round(np.linalg.norm(th @ (np.eye(p) - P)), 3), ' |minnorm (I-P)| =', round(np.linalg.norm(mn @ (np.eye(p) - P)), 12))
q_in = P @ qc
print('    in-span query: error warm-start', round(np.linalg.norm(th @ q_in - Ts @ q_in), 3), ' vs fresh Landweber of same length from 0:', end=' ')
th0 = np.zeros((V, p)); e = 0.9 / np.linalg.eigvalsh(Gt).max()
for _ in range(32): th0 = th0 - e * (th0 @ Gt - St)
print(round(np.linalg.norm(th0 @ q_in - Ts @ q_in), 3), '; out-of-span component of query read (warm):', round(np.linalg.norm(th @ (np.eye(p) - P) @ qc), 3))
# (3) alternating post rhat, y, rhat, y, ... with constant pre and an odd kernel K(tau) = sign(tau) exp(-|tau|/2)
rhat, y = 0.2, 1.0
post = np.array([rhat, y] * 8); pre = np.ones_like(post)
taus = np.arange(-6, 7); K = np.sign(taus) * np.exp(-np.abs(taus) / 2)
def pairing(post, pre):
    tot = 0.0
    for t in range(len(post)):
        for tau, k in zip(taus, K):
            if 0 <= t - tau < len(pre): tot += k * post[t] * pre[t - tau]
    return tot
one = pairing(np.array([rhat, y]), np.ones(2))
print('(3) odd-kernel pairing: one predict->observe pair', round(one, 4), ' 8 alternating pairs', round(pairing(post, pre), 4),
      ' (8 x single =', round(8 * one, 4), ')')
