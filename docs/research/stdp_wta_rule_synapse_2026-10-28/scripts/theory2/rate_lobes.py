"""Rate-form window: D_t = e_t + sum_r Re(a_r Xp_r + b_r Xm_r),
   Xp_r <- z_r Xp_r + g_r * sum_o eps_t(o) (phi_t(o) - phi_{t-1}(o))^T      (post x d pre: pre-leading lobe)
   Xm_r <- z_r Xm_r + g_r * sum_o (eps_t(o) - eps_{t-1}(o)) phi_t(o)^T      (d post x pre: post-leading lobe)
   eps_t(o) = y(o) - theta_t phi_t(o): the error of THIS lane's synapse (eps_{t-1} uses theta_{t-1}, phi_{t-1}).
   theta+ = soft(theta + D, lam + omega R), R = row max |soft(theta + e, lam) - theta|.  Perception phi_t -> phi* geometrically."""
exec(open('circuit.py').read().split("res = {}")[0])
def run(phis, Y, lam, T, omega=0.0, bands=()):
    n = Y.shape[0]; C, p = Y.shape[1], phis[0].shape[1]
    th = np.zeros((C, p)); th_prev = th.copy(); Xp = [np.zeros((C, p), complex) for _ in bands]; Xm = [np.zeros((C, p), complex) for _ in bands]
    out = []
    for t in range(T):
        ph = phis[min(t, len(phis) - 1)]; pp = phis[max(min(t, len(phis) - 1) - 1, 0)]
        eps = Y - ph @ th.T; eps_prev = Y - pp @ th_prev.T
        e = eps.T @ ph / n
        lob = np.zeros_like(th)
        for i, (z, a, b) in enumerate(bands):
            g = np.sqrt(1 - abs(z) ** 2)
            Xp[i] = z * Xp[i] + g * (eps.T @ (ph - pp)) / n
            Xm[i] = z * Xm[i] + g * ((eps - eps_prev).T @ ph) / n
            lob += (a * Xp[i] + b * Xm[i]).real
        R = np.abs(soft(th + e, lam) - th).max(1, keepdims=True) if omega else 0.0
        th_prev, th = th, soft(th + e + lob, lam + omega * R)
        out.append(th)
    return out
res = {}
Tmax = 20000
cfgs = dict(none=(), post_m=[(0.5 + 0j, 0.0, -0.8)], post_p=[(0.5 + 0j, 0.0, 0.8)], pre_m=[(0.5 + 0j, -0.8, 0.0)], pre_p=[(0.5 + 0j, 0.8, 0.0)],
            both=[(0.5 * np.exp(0.7j), 0.5 - 0.3j, -0.5 + 0.4j), (0.2 + 0j, -0.4, 0.3)])
acc = {(c, w): {T: [] for T in (8, 16, 32)} for c in cfgs for w in (0.0, 0.5)}; loc = []
for trial in range(12):
    phs, Y, tt, q = problem(40, 64, 13, 4)
    xi = unit(rng.standard_normal(phs.shape))
    phis = [unit(phs + 0.8 ** t * 2 * xi) for t in range(400)]          # perception settling over recursion
    Ys = phs @ tt.T                                                      # teacher defined on the settled features
    G, S = phs.T @ phs / 40, Ys.T @ phs / 40; ref = lasso(G, S, 1e-4)
    truth = (q @ tt.T).argmax(1)
    for c, bands in cfgs.items():
        for w in (0.0, 0.5):
            h = run(phis, Ys, 1e-4, Tmax if trial < 3 else 32, omega=w, bands=bands)
            for T in (8, 16, 32): acc[(c, w)][T].append(((q @ h[T - 1].T).argmax(1) == truth).mean())
            if trial < 3: loc.append((c, w, float(np.abs(h[-1] - ref).max())))
for k, v in acc.items(): print(k, {T: round(float(np.mean(x)), 3) for T, x in v.items()})
print('location: max |theta_20000 - lasso(phi*)| over configs/trials:', max(x[2] for x in loc))
print([x for x in loc if x[2] > 1e-6])
