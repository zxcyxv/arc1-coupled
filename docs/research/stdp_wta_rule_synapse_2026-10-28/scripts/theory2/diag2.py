exec(open('circuit.py').read().split("res = {}")[0])
# convergence region over (c0, omega), 6 problems each, 40000 steps
for c0 in (1.0, 1.4, 1.8):
    row = []
    for omega in (0.5, 0.7, 0.8, 0.9, 0.95):
        bad = 0
        for trial in range(6):
            phi, Y, tt, q = problem(40, 64, 13, 4)
            n = len(phi); G, S = phi.T @ phi / n, Y.T @ phi / n
            th = circuit(G, S, 1e-3, 40000, c0=c0, omega=omega)
            bad += kkt(th, G, S, 1e-3) > 1e-8
        row.append(f'w{omega}:{bad}/6')
    print('c0', c0, ' '.join(row), flush=True)
