import numpy as np
from checks import unit, softmax, K
rng = np.random.default_rng(1)

def problem(n, p, nsit, noise, nq=400):
    proto = unit(rng.standard_normal((nsit, p)))
    rule = rng.integers(K, size=nsit)
    s = rng.integers(nsit, size=n); sq = rng.integers(nsit, size=nq)
    phi = unit(proto[s] + noise * unit(rng.standard_normal((n, p))))
    phq = unit(proto[sq] + noise * unit(rng.standard_normal((nq, p))))
    return phi, np.eye(K)[rule[s]], phq, rule[sq]

def head(phi, Y, T, mode, eta=2.0, kappa=0.5, rho=0.9):
    n, p = phi.shape; th = np.zeros((K, p)); E = np.zeros_like(th); Cprev = -np.inf
    G = phi.T @ phi / n; S = Y.T @ phi / n; Yt = rho * Y + (1 - rho) / K
    for t in range(T):
        if mode == 'lasso':
            th = th - (th @ G - S); continue
        q = softmax(phi @ th.T); C = (Yt * np.log(q)).sum(-1).mean()
        g = 1.0 if C >= Cprev else 0.0; Cprev = C
        H = (Yt - q).T @ phi / n
        E = H + (g * kappa if mode == 'wta_trace' else 0.0) * E
        th = th + eta * E
    return th

def newton_ref(phi, Y, rho=0.9, iters=200):
    """smoothed-CE minimiser by many large safe steps + momentum restart (reference)."""
    return head(phi, Y, 200000, 'wta_trace', eta=2.0 / max(np.linalg.eigvalsh(phi.T @ phi / len(phi)).max(), 1e-9) * 0.5, kappa=0.9, rho=rho)

res = {}
for (n, p, nsit, noise) in [(120, 64, 40, 0.6), (300, 64, 60, 0.8), (60, 64, 30, 0.5)]:
    acc = {m: {T: [] for T in (8, 30)} for m in ('lasso', 'wta', 'wta_trace')}
    dacc = {m: {T: [] for T in (8, 30)} for m in ('lasso', 'wta', 'wta_trace')}
    for trial in range(20):
        phi, Y, phq, yq = problem(n, p, nsit, noise)
        for m in acc:
            for T in (8, 30):
                th = head(phi, Y, T, m)
                acc[m][T].append((np.argmax(phq @ th.T, -1) == yq).mean())
                dacc[m][T].append((np.argmax(phi @ th.T, -1) == Y.argmax(-1)).mean())
    print(f'n={n} p={p} situations={nsit} noise={noise}')
    for m in acc:
        print(f'  {m:10s} ' + '  '.join(f'T{T}: demo {np.mean(dacc[m][T]):.3f} query {np.mean(acc[m][T]):.3f}' for T in (8, 30)))
# fixed point invariance with a converged reference
phi, Y, phq, yq = problem(120, 64, 40, 0.6)
ref = newton_ref(phi, Y)
for m, kap in (('wta', 0.0), ('wta_trace', 0.5), ('wta_trace', 0.8)):
    th = head(phi, Y, 100000, m, kappa=kap)
    print(f'fixed point {m} kappa {kap}: rel diff to reference {np.linalg.norm(th - ref) / np.linalg.norm(ref):.2e}')
q = softmax(phi @ ref.T); print('reference moment residual', np.abs((0.9 * Y + 0.01 / 1.1 * 1.0 - q).T @ phi).max())
