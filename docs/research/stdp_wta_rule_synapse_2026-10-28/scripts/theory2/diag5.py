exec(open('circuit.py').read().split("res = {}")[0])
probs = []
for n, oh in ((40, False), (150, False), (40, True), (150, True)):
    for trial in range(4):
        phi, Y, tt, q = problem(n, 64, 13, 4)
        if oh: Y = np.eye(13)[(phi @ tt.T).argmax(1)]
        probs.append(((n, oh), phi.T @ phi / n, Y.T @ phi / n))
for omega, c0 in ((0.0, 1.0), (0.0, 1.9), (0.5, 1.0), (0.6, 1.9), (0.9, 1.0)):
    out = []
    for (key, G, S) in probs:
        th, h = circuit(G, S, 1e-3, 100000, c0=c0, omega=omega, hist=True)
        ks = [kkt(h[T], G, S, 1e-3) for T in (999, 9999, 99999)]
        out.append(ks)
    out = np.array(out)
    print(f'omega {omega} c0 {c0}: median kkt at 1e3/1e4/1e5 steps', np.median(out, 0), 'max at 1e5', out[:, 2].max(), flush=True)
