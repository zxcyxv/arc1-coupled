"""6x6 Sudoku (2x3 boxes) for lt/: random solved grids, clues removed in random order while the solution stays unique
(minimal puzzles). Writes data/sudoku6_lt.npz in the sudoku_lt_1k layout (raw digits, 0 = blank).
--hard keeps only puzzles that naked + hidden singles cannot finish (some step needs a guess or a stronger rule).
    python data/make_sudoku6.py [--train 1000] [--test 2048] [--seed 0] [--hard --out data/sudoku6_hard_lt.npz]"""
import argparse, numpy as np
from multiprocessing import Pool

N, BR, BC = 6, 2, 3
CELLS = [(r, c) for r in range(N) for c in range(N)]
PEERS = []
for r, c in CELLS:
    p = {(r, j) for j in range(N)} | {(i, c) for i in range(N)}
    br, bc = r // BR * BR, c // BC * BC
    p |= {(br + i, bc + j) for i in range(BR) for j in range(BC)}
    p.discard((r, c))
    PEERS.append([i * N + j for i, j in p])


def count(g, limit=2, rng=None):
    """Number of solutions of the flat grid g (0 = blank), stopping at limit; fills g with the first if rng given."""
    empty = [i for i in range(36) if g[i] == 0]
    if not empty:
        return 1
    best, bopts = None, None
    for i in empty:
        opts = set(range(1, 7)) - {g[j] for j in PEERS[i]}
        if best is None or len(opts) < len(bopts):
            best, bopts = i, opts
            if len(opts) <= 1:
                break
    opts = list(bopts)
    if rng is not None:
        rng.shuffle(opts)
    n = 0
    for v in opts:
        g[best] = v
        n += count(g, limit - n, rng)
        if n >= limit:
            if rng is None:
                g[best] = 0
            return n
        if rng is None:
            g[best] = 0
    if rng is not None and n == 0:
        g[best] = 0
    return n


UNITS = [[r * N + c for c in range(N)] for r in range(N)] + [[r * N + c for r in range(N)] for c in range(N)]
UNITS += [[(br + i) * N + bc + j for i in range(BR) for j in range(BC)] for br in range(0, N, BR) for bc in range(0, N, BC)]


def singles_solve(g):
    """True if naked + hidden singles alone fill the grid (flat list, 0 = blank)."""
    g = list(g)
    while 0 in g:
        cand = {i: set(range(1, 7)) - {g[j] for j in PEERS[i]} for i in range(36) if g[i] == 0}
        placed = False
        for i, v in cand.items():
            if len(v) == 1:
                g[i] = next(iter(v)); placed = True
        if not placed:
            for u in UNITS:
                for d in range(1, 7):
                    where = [i for i in u if i in cand and d in cand[i]]
                    if len(where) == 1 and g[where[0]] == 0:
                        g[where[0]] = d; placed = True
        if not placed:
            return False
    return True


def make(seed):
    rng = np.random.default_rng(seed)
    sol = [0] * 36
    count(sol, 1, rng)
    puz = sol[:]
    for i in rng.permutation(36):
        v, puz[i] = puz[i], 0
        if count(puz[:], 2) != 1:
            puz[i] = v
    return np.array(puz, np.uint8).reshape(N, N), np.array(sol, np.uint8).reshape(N, N), singles_solve(puz)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=1000)
    ap.add_argument('--test', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/sudoku6_lt.npz')
    ap.add_argument('--hard', action='store_true')
    a = ap.parse_args()
    seen, xs, ys, s, tried = set(), [], [], a.seed * 10**7, 0
    with Pool() as pool:
        while len(xs) < a.train + a.test:
            for x, y, easy in pool.map(make, range(s, s + 2000)):
                tried += 1
                if a.hard and easy:
                    continue
                if x.tobytes() not in seen and len(xs) < a.train + a.test:
                    seen.add(x.tobytes()); xs.append(x); ys.append(y)
            s += 2000
    print(f'kept {len(xs)} of {tried} generated puzzles')
    x, y = np.stack(xs), np.stack(ys)
    clues = (x > 0).sum((1, 2))
    print('clues mean %.2f min %d max %d' % (clues.mean(), clues.min(), clues.max()))
    np.savez_compressed(a.out, train_inputs=x[:a.train], train_labels=y[:a.train],
                        test_inputs=x[a.train:], test_labels=y[a.train:])
    print('wrote', a.out, x.shape)
