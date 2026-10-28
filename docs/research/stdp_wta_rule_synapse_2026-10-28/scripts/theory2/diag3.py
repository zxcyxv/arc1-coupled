exec(open('circuit.py').read().split("res = {}")[0])
import json
def run(n, onehot, trials=30, lam=1e-4):
    Ts = (4, 8, 16, 32, 128)
    names = ['ista', 'w0.3', 'w0.5', 'w0.7', 'w0.9', 'fista', 'iht', 'fixed']
    acc = {m: {T: [] for T in Ts} for m in names}
    for trial in range(trials):
        phi, Y, tt, q = problem(n, 64, 13, 4)
        truth = (q @ tt.T).argmax(1)
        if onehot:
            Y = np.eye(13)[(phi @ tt.T).argmax(1)]
        G, S = phi.T @ phi / n, Y.T @ phi / n
        ref = lasso(G, S, lam, iters=100000)
        for m in names:
            for T in Ts:
                if m == 'ista': th = circuit(G, S, lam, T)
                elif m.startswith('w'): th = circuit(G, S, lam, T, omega=float(m[1:]))
                elif m == 'fista': th = fista(G, S, lam, T)
                elif m == 'iht': th = iht(G, S, 4, T)
                else: th = ref
                acc[m][T].append(((q @ th.T).argmax(1) == truth).mean())
    return {m: [round(float(np.mean(acc[m][T])), 3) for T in Ts] for m in names}
for n in (40, 150):
    for oh in (False, True):
        print(f'n={n} onehot={oh} (T = 4, 8, 16, 32, 128)', flush=True)
        for m, v in run(n, oh).items(): print('  ', m, v, flush=True)
