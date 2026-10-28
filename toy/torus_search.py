"""Iterative rule search on the 4x4 torus toy, stage 1: does a three-factor rule (eligibility trace x internal
dopamine) with competition improve the search trajectory? The representation is exact (designed group elements), so
only the dynamics differ between conditions.

Each instance (data/gtorus_instances.npz, copied from zxcyxv/something) has 3 demonstrations and a query whose output
the demonstrations determine. State: a transformation t (0..127), starting at the identity. A step applies one move
(stay, rot, rot^-1, flip, down, up, right, left) after t. Internal reward r(t): the fraction of demonstration cells
(3 x 16) explained by t under the best colour permutation, computed from the demonstrations alone; r = 1 is the
'eureka' state, and its prediction of the query (t and that colour permutation) is then right (determinacy).

Conditions (every searcher chooses a move by softmax over move scores, beta the inverse temperature):
  exhaustive   score all 128 x 24 (t, c) at once (upper bound, no steps)
  random       uniform moves
  greedy       score = r(next) - r(t): no memory
  hebb         greedy + a fast state value U += eta * trace (visited states attract: same-time Hebbian on visits)
  dopamine     greedy + U += eta * delta * trace, delta = r(t_n) - r(t_n-1) - b (three-factor rule: stagnation is a
               negative prediction error, so visited states are depressed; a jump to r = 1 is a large positive one)
  dopamine0    dopamine without the trace (kappa 0: only the current state is updated)
Trace: e <- kappa e, e[t_n] += 1. Fast state values start at 0 for every instance.

    python -m toy.torus_search            # fit the few scalars on train instances, evaluate all instances
"""
import argparse
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np

from . import torus_group as G

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'gtorus_instances.npz'
GROUPS = ('train', 'heldout_combo', 'heldout_t')


def _power(t, k):
    out = G.IDENTITY
    for _ in range(k):
        out = G.compose(t, out)
    return out


MOVES = dict(stay=G.IDENTITY, rot=G.GENERATORS['rot'], rot_inv=_power(G.GENERATORS['rot'], 3), flip=G.GENERATORS['flip'],
             down=G.GENERATORS['down'], up=_power(G.GENERATORS['down'], 3), right=G.GENERATORS['right'],
             left=_power(G.GENERATORS['right'], 3))
NEXT = np.array([[G.compose(m, t) for m in MOVES.values()] for t in range(G.NUM_T)])           # [128, 8]


def consistency(demo_in, demo_out, chunk=64):
    """r [n, 128] (fraction of demonstration cells explained by t under its best colour permutation) and that
    permutation c [n, 128]."""
    n = len(demo_in)
    r, best = np.zeros((n, G.NUM_T)), np.zeros((n, G.NUM_T), dtype=np.int64)
    Y = demo_out.reshape(n, -1).astype(np.int64)
    for i in range(0, n, chunk):
        X = np.stack([G.apply_t(demo_in[i:i + chunk], t) for t in range(G.NUM_T)], 1)          # [b, 128, K, 4, 4]
        X = X.reshape(*X.shape[:2], -1).astype(np.int64)                                       # [b, 128, K*16]
        match = (G.CMAPS[:, X] == Y[i:i + chunk][None, :, None]).sum(-1)                       # [24, b, 128]
        r[i:i + chunk] = match.max(0) / Y.shape[1]
        best[i:i + chunk] = match.argmax(0)
    return r, best


def shortest(r):
    """Fewest moves from the identity to a state with r = 1 (breadth-first search on the move graph)."""
    dist = np.full(G.NUM_T, -1)
    dist[G.IDENTITY] = 0
    q = deque([G.IDENTITY])
    while q:
        t = q.popleft()
        if r[t] >= 1:
            return dist[t]
        for t2 in NEXT[t]:
            if dist[t2] < 0:
                dist[t2] = dist[t] + 1
                q.append(t2)
    return -1


def search(r, kind, rng, max_steps=256, post=32, beta=20.0, eta=1.0, kappa=0.9, b=0.05):
    """One instance. Returns steps to r = 1 (max_steps + 1 if not reached), the fraction of steps that land on an
    already visited state, and the fraction of `post` further steps that stay at r = 1."""
    t, U, e = G.IDENTITY, np.zeros(G.NUM_T), np.zeros(G.NUM_T)
    seen = np.zeros(G.NUM_T, bool)
    seen[t] = True
    revisits, found, stay = 0, None, 0
    for n in range(max_steps + post):
        if found is None and r[t] >= 1:
            found = n
        if found is not None and n - found >= post:
            break
        nxt = NEXT[t]
        if kind == 'random':
            m = rng.integers(len(nxt))
        else:
            score = r[nxt] - r[t]
            if kind != 'greedy':
                score = score + U[nxt]
            z = beta * score + rng.gumbel(size=len(nxt))                 # softmax sample (Gumbel-max)
            m = int(np.argmax(z))
        t_prev, t = t, int(nxt[m])
        if found is None:
            revisits += bool(seen[t])
            seen[t] = True
        else:
            stay += r[t] >= 1
        if kind in ('hebb', 'dopamine', 'dopamine0'):
            e *= 0.0 if kind == 'dopamine0' else kappa
            e[t] += 1.0
            if kind == 'hebb':
                U += eta * e
            else:
                U += eta * (r[t] - r[t_prev] - b) * e
    steps = found if found is not None else max_steps + 1
    return steps, revisits / max(min(steps, max_steps), 1), (stay / post if found is not None else np.nan)


def run(rs, kind, seed=0, **kw):
    rng = np.random.default_rng(seed)
    out = np.array([search(r, kind, rng, **kw) for r in rs])
    return out[:, 0], out[:, 1], out[:, 2]


def summary(steps, rev, stay, max_steps):
    ok = steps <= max_steps
    return dict(success={k: float((steps <= k).mean()) for k in (16, 32, 64, 128, max_steps)},
                median_steps=float(np.median(np.minimum(steps, max_steps + 1))), mean_steps=float(np.minimum(steps, max_steps + 1).mean()),
                revisit=float(rev.mean()), relapse_free=float(np.nanmean(stay)) if ok.any() else float('nan'))


GRIDS = dict(
    greedy=dict(beta=[5.0, 20.0, 50.0]),
    hebb=dict(beta=[5.0, 20.0, 50.0], eta=[0.01, 0.1, 1.0], kappa=[0.5, 0.9]),
    dopamine=dict(beta=[5.0, 20.0, 50.0], eta=[0.3, 1.0, 3.0], kappa=[0.5, 0.8, 0.95], b=[0.01, 0.05, 0.2]),
    dopamine0=dict(beta=[5.0, 20.0, 50.0], eta=[0.3, 1.0, 3.0], b=[0.01, 0.05, 0.2]),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-steps', type=int, default=256)
    ap.add_argument('--fit-n', type=int, default=400, help='train instances used to fit the scalars')
    ap.add_argument('--out', default=str(ROOT / 'runs' / 'toy' / 'torus_search.json'))
    args = ap.parse_args()
    z = np.load(DATA)
    r, best = consistency(z['demo_in'], z['demo_out'])
    t_true, c_true = z['task'] // G.NUM_C, z['task'] % G.NUM_C
    # determinacy check: every r = 1 state predicts the query right
    for i in range(len(r)):
        for t in np.flatnonzero(r[i] >= 1):
            assert (G.apply_task(z['test_in'][i], t, best[i, t]) == z['test_out'][i]).all(), (i, t)
    assert (r[np.arange(len(r)), t_true] >= 1).all()
    group = z['group']
    opt = np.array([shortest(x) for x in r])
    print(f'instances {len(r)}; consistent t per instance: mean {(r >= 1).sum(1).mean():.2f}; '
          f'fewest moves to r = 1: mean {opt.mean():.2f}, max {opt.max()}; '
          f'r of a wrong t: mean {r[r < 1].mean():.3f}, 99th percentile {np.quantile(r[r < 1], 0.99):.3f}')
    fit = np.flatnonzero(group == 0)[:args.fit_n]
    res = dict(meta=dict(instances=len(r), max_steps=args.max_steps, fit_n=len(fit), optimal_mean=float(opt.mean())))
    for kind in ('random', 'greedy', 'hebb', 'dopamine', 'dopamine0'):
        params = {}
        if kind in GRIDS:
            grid = GRIDS[kind]
            best_score = None
            for vals in itertools.product(*grid.values()):
                kw = dict(zip(grid, vals))
                s, _, _ = run(r[fit], kind, seed=1, max_steps=args.max_steps, **kw)
                score = np.minimum(s, args.max_steps + 1).mean()
                if best_score is None or score < best_score:
                    best_score, params = score, kw
        s, rev, stay = run(r, kind, seed=2, max_steps=args.max_steps, **params)
        res[kind] = dict(params=params, all=summary(s, rev, stay, args.max_steps),
                         **{GROUPS[g]: summary(s[group == g], rev[group == g], stay[group == g], args.max_steps)
                            for g in range(3)})
        a = res[kind]['all']
        print(f"{kind:10s} {params} | success@32 {a['success'][32]:.3f} @64 {a['success'][64]:.3f} "
              f"@{args.max_steps} {a['success'][args.max_steps]:.3f} | median steps {a['median_steps']:.0f} "
              f"mean {a['mean_steps']:.1f} | revisit {a['revisit']:.3f} | relapse-free {a['relapse_free']:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
