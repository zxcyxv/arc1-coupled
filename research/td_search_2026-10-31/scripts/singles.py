"""Fraction of puzzles solved by naked + hidden singles alone (no guessing), 6x6 (2x3 boxes) or 9x9."""
import sys, numpy as np
d = np.load(sys.argv[1])
x = d['test_inputs']
N = x.shape[-1]; BR, BC = (2, 3) if N == 6 else (3, 3)
units = [[(r, c) for c in range(N)] for r in range(N)] + [[(r, c) for r in range(N)] for c in range(N)]
units += [[(br + i, bc + j) for i in range(BR) for j in range(BC)] for br in range(0, N, BR) for bc in range(0, N, BC)]
peers = {(r, c): set() for r in range(N) for c in range(N)}
for u in units:
    for p in u:
        peers[p] |= set(u) - {p}
def solve(g):
    g = g.copy(); steps = 0
    while True:
        cand = {(r, c): set(range(1, N + 1)) - {g[p] for p in peers[(r, c)]} for r in range(N) for c in range(N) if g[r, c] == 0}
        if not cand: return True, steps
        if any(len(v) == 0 for v in cand.values()): return False, steps
        placed = False
        for p, v in cand.items():
            if len(v) == 1: g[p] = v.pop(); placed = True
        if not placed:
            for u in units:
                for dgt in range(1, N + 1):
                    where = [p for p in u if p in cand and dgt in cand[p]]
                    if len(where) == 1 and g[where[0]] == 0: g[where[0]] = dgt; placed = True
        if not placed: return False, steps
        steps += 1
res = [solve(p) for p in x]
ok = np.array([r[0] for r in res]); st = np.array([r[1] for r in res])
print(sys.argv[1], 'singles-only solved %d/%d (%.1f%%), rounds median %d max %d' % (ok.sum(), len(ok), 100 * ok.mean(), np.median(st[ok]) if ok.any() else -1, st[ok].max() if ok.any() else -1))
