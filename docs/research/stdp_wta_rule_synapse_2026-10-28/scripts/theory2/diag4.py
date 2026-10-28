exec(open('circuit.py').read().split("res = {}")[0])
# (a) is the failure a cycle? c0=1.4, omega=0.7
phi, Y, tt, q = problem(40, 64, 13, 4); n = 40
G, S = phi.T @ phi / n, Y.T @ phi / n
th, h = circuit(G, S, 1e-3, 40000, c0=1.4, omega=0.7, hist=True)
for per in (1, 2, 3, 4, 6):
    print('period', per, 'max|th_T - th_{T-per}|', np.abs(h[-1] - h[-1 - per]).max())
# (b) safe region on harder mixes, 20000 steps
for omega in (0.4, 0.5, 0.6):
    for c0 in (1.0, 1.5, 1.9):
        bad = 0; tot = 0
        for n, oh in ((40, False), (150, False), (40, True), (150, True)):
            for trial in range(4):
                phi, Y, tt, q = problem(n, 64, 13, 4)
                if oh: Y = np.eye(13)[(phi @ tt.T).argmax(1)]
                G, S = phi.T @ phi / n, Y.T @ phi / n
                th = circuit(G, S, 1e-3, 20000, c0=c0, omega=omega)
                bad += kkt(th, G, S, 1e-3) > 1e-8; tot += 1
        print(f'omega {omega} c0 {c0}: non-converged {bad}/{tot}', flush=True)
