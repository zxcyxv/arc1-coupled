"""float64, rule-head-like problems: the window form theta+ = soft(theta + e + sum_r Re(b_r D_r), Kbar (lam + omega R)),
D_r = z_r E_r (past drives), E_r <- D_r + g_r e (bands' D_V / E_V recursion), Kbar = 1 + sum_r Re(b_r g_r z_r / (1 - z_r)),
R = row max |soft(theta + e + lobe, Kbar lam) - theta| / Kbar."""
exec(open('circuit.py').read().split("res = {}")[0])
def win(G, S, lam, T, omega=0.5, bands=((0.9 + 0j, None),), hist=False):
    th = np.zeros_like(S); E = [np.zeros(S.shape, complex) for _ in bands]; out = []
    gs = [np.sqrt(1 - abs(z) ** 2) for z, _ in bands]
    bs = [(b if b is not None else 1.0 / g) for (z, b), g in zip(bands, gs)]        # default: lobe beta^tau (heavy ball)
    Kb = 1 + sum((b * g * z / (1 - z)).real for (z, _), b, g in zip(bands, bs, gs))
    for t in range(T):
        e = S - th @ G; lob = np.zeros_like(S)
        for i, ((z, _), b, g) in enumerate(zip(bands, bs, gs)):
            D = z * E[i]; lob += (b * D).real; E[i] = D + g * e
        u = th + e + lob
        R = np.abs(soft(u, Kb * lam) - th).max(1, keepdims=True) / Kb if omega else 0.0
        th = soft(u, Kb * (lam + omega * R))
        if hist: out.append(th.copy())
    return (th, out, Kb) if hist else th
res = {}
for n, oh in ((40, False), (150, False), (40, True), (150, True)):
    Ts = (4, 8, 16, 32); A = {m: {T: [] for T in Ts} for m in ('ista', 'fista', 'wta', 'wta+lobe', 'lobe_only', 'fixed')}
    for trial in range(30):
        phi, Y, tt, q = problem(n, 64, 13, 4); truth = (q @ tt.T).argmax(1)
        if oh: Y = np.eye(13)[(phi @ tt.T).argmax(1)]
        G, S = phi.T @ phi / n, Y.T @ phi / n; lam = 1e-4; ref = lasso(G, S, lam, iters=100000)
        for T in Ts:
            for m, th in (('ista', circuit(G, S, lam, T)), ('fista', fista(G, S, lam, T)), ('wta', circuit(G, S, lam, T, omega=0.5)),
                          ('wta+lobe', win(G, S, lam, T)), ('lobe_only', win(G, S, lam, T, omega=0.0)), ('fixed', ref)):
                A[m][T].append(((q @ th.T).argmax(1) == truth).mean())
    res[f'n{n}_onehot{oh}'] = {m: [round(float(np.mean(A[m][T])), 3) for T in Ts] for m in A}
    print(f'n={n} onehot={oh} T={Ts}', res[f"n{n}_onehot{oh}"], flush=True)
# location + stability with complex poles
loc = []
for trial in range(8):
    phi, Y, tt, q = problem(40, 64, 13, 4); G, S = phi.T @ phi / 40, Y.T @ phi / 40; lam = 1e-3; ref = lasso(G, S, lam)
    for bands in (((0.9 + 0j, None),), ((0.8 * np.exp(0.5j), 0.6 - 0.4j), (0.5 + 0j, 0.8 + 0j)), ((0.9 * np.exp(1.2j), 1.5 + 1.0j),)):
        th, h, Kb = win(G, S, lam, 30000, bands=bands, hist=True)
        loc.append((round(float(Kb), 3), float(np.abs(th - ref).max()), float(kkt(th, G, S, lam)), bool(np.isfinite(th).all())))
print('location (Kbar, max|theta - lasso|, kkt, finite):'); [print('  ', x) for x in loc]
