"""Branch points a DFS needs when every node first applies naked + hidden singles (6x6, 2x3 boxes), and whether
naked pairs / pointing (a stronger rule, still no guessing) would finish the singles-stuck puzzles."""
import sys, numpy as np
N, BR, BC = 6, 2, 3
UNITS = [[r * N + c for c in range(N)] for r in range(N)] + [[r * N + c for r in range(N)] for c in range(N)]
UNITS += [[(br + i) * N + bc + j for i in range(BR) for j in range(BC)] for br in range(0, N, BR) for bc in range(0, N, BC)]
PEERS = [set() for _ in range(36)]
for u in UNITS:
    for i in u:
        PEERS[i] |= set(u) - {i}

def propagate(cand, pairs=False):
    """cand: list of sets; returns False on contradiction. Naked + hidden singles (and optionally naked pairs and
    box-line pointing) to a fixed point."""
    changed = True
    while changed:
        changed = False
        for i in range(36):
            if len(cand[i]) == 0: return False
            if len(cand[i]) == 1:
                v = next(iter(cand[i]))
                for j in PEERS[i]:
                    if v in cand[j]:
                        cand[j] = cand[j] - {v}; changed = True
                        if not cand[j]: return False
        for u in UNITS:
            for d in range(1, 7):
                where = [i for i in u if d in cand[i]]
                if not where: return False
                if len(where) == 1 and len(cand[where[0]]) > 1:
                    cand[where[0]] = {d}; changed = True
        if pairs and not changed:
            for u in UNITS:
                two = [i for i in u if len(cand[i]) == 2]
                for a in range(len(two)):
                    for b in range(a + 1, len(two)):
                        if cand[two[a]] == cand[two[b]]:
                            for j in u:
                                if j not in (two[a], two[b]) and cand[j] & cand[two[a]]:
                                    cand[j] = cand[j] - cand[two[a]]; changed = True
            for bi, u in enumerate(UNITS[2 * N:]):            # pointing: digit confined to one row/col inside a box
                for d in range(1, 7):
                    where = [i for i in u if d in cand[i]]
                    rows, cols = {i // N for i in where}, {i % N for i in where}
                    lines = []
                    if len(rows) == 1: lines.append(UNITS[next(iter(rows))])
                    if len(cols) == 1: lines.append(UNITS[N + next(iter(cols))])
                    for line in lines:
                        for j in line:
                            if j not in u and d in cand[j]:
                                cand[j] = cand[j] - {d}; changed = True
    return True

def dfs(cand, stats):
    if not propagate(cand): return False
    if all(len(c) == 1 for c in cand): return True
    i = min((k for k in range(36) if len(cand[k]) > 1), key=lambda k: len(cand[k]))
    stats['branch'] += 1
    for v in sorted(cand[i]):
        c2 = [set(c) for c in cand]; c2[i] = {v}
        if dfs(c2, stats): return True
        stats['wrong'] += 1
    return False

d = np.load(sys.argv[1]); x = d['test_inputs']
branches, wrongs, pairs_ok = [], [], 0
for p in x:
    g = p.flatten()
    cand = [({int(g[i])} if g[i] else set(range(1, 7))) for i in range(36)]
    st = dict(branch=0, wrong=0)
    dfs([set(c) for c in cand], st)
    branches.append(st['branch']); wrongs.append(st['wrong'])
    c3 = [set(c) for c in cand]
    if propagate(c3, pairs=True) and all(len(c) == 1 for c in c3): pairs_ok += 1
b, w = np.array(branches), np.array(wrongs)
print(sys.argv[1], f'n={len(x)}: needs >=1 branch {np.mean(b > 0) * 100:.1f}%, branches mean {b.mean():.2f} max {b.max()}; '
      f'wrong guesses (backtracks) >=1 {np.mean(w > 0) * 100:.1f}%, mean {w.mean():.2f}; '
      f'solved by singles+pairs+pointing (no guess) {pairs_ok / len(x) * 100:.1f}%')
