"""Float64 checks for the competing delta-rule head (no threshold) on the conjunctive code psi = pi (x) phi."""
import numpy as np
rng = np.random.default_rng(1)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

n, p, M, Cy, dv = 40, 16, 4, 13, 8   # n < M p: underdetermined, so selection matters
phi = unit(rng.standard_normal((n, p)))
pi_soft = softmax(2.0 * rng.standard_normal((n, M)))
psi = (pi_soft[:, :, None] * phi[:, None, :]).reshape(n, M * p)
print('max |psi|^2 =', (psi ** 2).sum(1).max(), '(<= 1 so tr G <= 1)')
Th_true = rng.standard_normal((Cy + dv, M * p))
T = psi @ Th_true.T
G = psi.T @ psi / n; S = T.T @ psi / n
lmax = np.linalg.eigvalsh(G).max()

# 1. Landweber from 0, step 1/lmax -> min-norm LS; iterate stays in the row space of the demo codes
th = np.zeros_like(S); eta = 1 / lmax
for _ in range(200000):
    th = th - eta * (th @ G - S)
U, sv, Vt = np.linalg.svd(psi, full_matrices=False); r = (sv > 1e-10 * sv[0]).sum()
mn = T.T @ np.linalg.pinv(psi.T, rcond=1e-10)        # min-norm interpolant
print('rank of demo codes', r, 'of', M * p, '; smallest nonzero eig of G', (sv[r-1] ** 2) / n)
P = Vt[:r].T @ Vt[:r]             # projector on span of demo codes
print('1. |Landweber - minnorm| =', np.abs(th - mn).max(), ' |th (I-P)| =', np.abs(th @ (np.eye(M*p) - P)).max())

# 2. span certificate: query error = Th_true (I - P) psi*
q_in = (P @ rng.standard_normal(M * p)); q_out = rng.standard_normal(M * p)
for name, q in (('in-span', q_in), ('generic', q_out)):
    err = mn @ q - Th_true @ q
    pred = -Th_true @ (np.eye(M*p) - P) @ q
    print(f'2. {name}: |err| = {np.abs(err).max():.2e}, matches -Th (I-P) q: {np.abs(err - pred).max():.2e}, residual |(I-P)q| = {np.linalg.norm((np.eye(M*p)-P)@q):.2e}')

# 3. hard gates -> block-diagonal G, per-head independent regressions
hard = np.eye(M)[rng.integers(0, M, n)]
psi_h = (hard[:, :, None] * phi[:, None, :]).reshape(n, M * p)
Gh = psi_h.T @ psi_h / n; Sh = (psi_h @ Th_true.T).T @ psi_h / n
off = sum(np.abs(Gh[m*p:(m+1)*p, k*p:(k+1)*p]).max() for m in range(M) for k in range(M) if m != k)
joint = Sh @ np.linalg.pinv(Gh)
per = np.concatenate([ (Sh[:, m*p:(m+1)*p]) @ np.linalg.pinv(Gh[m*p:(m+1)*p, m*p:(m+1)*p]) for m in range(M)], 1)
print('3. hard gates: off-diagonal blocks', off, ' |joint - per-head| =', np.abs(joint - per).max())
Gs_off = np.abs(G[0:p, p:2*p]).max()
print('   soft gates: largest cross-head block entry', Gs_off, '(heads inhibit each other through -Theta G)')

# 4. fixed point invariance: DC gains k_h on S, k_a on G -> (k_h/k_a) min-norm; argmax of colour rows unchanged
kh, ka = 0.7, 1.3
th = np.zeros_like(S); eta = 1 / (ka * lmax)
for _ in range(2000000):
    th = th - eta * (th @ (ka * G) - kh * S)
print('4. |limit - (kh/ka) minnorm| =', np.abs(th - kh / ka * mn).max(),
      ' argmax agreement on demo cells', ( (psi @ th.T)[:, :Cy].argmax(1) == (psi @ mn.T)[:, :Cy].argmax(1)).mean())

# 5. finite T from 0 = spectral filter (1 - (1 - eta s)^T) / s : early stopping as the noise threshold
w, V = np.linalg.eigh(G); Tsteps = 8; eta = 1 / lmax
th = np.zeros_like(S)
for _ in range(Tsteps):
    th = th - eta * (th @ G - S)
filt = np.where(w > 1e-12, (1 - (1 - eta * w) ** Tsteps) / np.maximum(w, 1e-300), 0.0)
pred = S @ V @ np.diag(filt) @ V.T
print('5. |Landweber_T - filter| =', np.abs(th - pred).max(), ' filter factor at s = lmax/100:', 1 - (1 - 0.01) ** Tsteps)
print('   norm bound |Theta_T| <= T eta |S|:', np.linalg.norm(th, 2), '<=', Tsteps * eta * np.linalg.norm(S, 2))

# 6. step 1 (the tr G bound) vs 1/lmax: eigenvalue spread of G at p = 64, M = 4 with unit phi
n2, p2 = 121, 64
phi2 = unit(rng.standard_normal((n2, p2))); pi2 = softmax(3.0 * rng.standard_normal((n2, M)))
psi2 = (pi2[:, :, None] * phi2[:, None, :]).reshape(n2, M * p2)
ev = np.linalg.eigvalsh(psi2.T @ psi2 / n2)
print('6. lmax(G_psi) =', ev.max(), ' tr =', ev.sum(), ' -> step 1/lmax is', 1 / ev.max(), 'x the tr bound')
