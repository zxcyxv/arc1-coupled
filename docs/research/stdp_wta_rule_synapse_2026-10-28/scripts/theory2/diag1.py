exec(open('circuit.py').read().split("res = {}")[0])
rng2 = np.random.default_rng(5)
for trial in range(3):
    phi, Y, tt, q = problem(40, 64, 13, 4)
    n = len(phi); G, S = phi.T @ phi / n, Y.T @ phi / n
    lam = 1e-3; ref = lasso(G, S, lam)
    bands = [(0.6 * np.exp(1j * rng2.uniform(0, np.pi)), 0.3 * (rng2.standard_normal() + 1j * rng2.standard_normal())) for _ in range(2)]
    for omega in (0.0, 0.5, 0.9):
        for c0 in (1.0, 1.8):
            for bb in ((), bands):
                th, h = circuit(G, S, lam, 60000, c0=c0, omega=omega, bands=bb, hist=True)
                d = [np.abs(h[T] - ref).max() for T in (999, 9999, 59999)]
                print(trial, omega, c0, 'lobes' if bb else 'none', ['%.2e' % x for x in d], 'kkt %.1e' % kkt(th, G, S, lam), 'finite', np.isfinite(th).all())
