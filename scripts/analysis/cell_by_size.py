"""Cell accuracy of the test queries against the size of the true output grid, version 0 (no augmentation), for a
checkpoint of arc.train: per size bin the number of test pairs, exact pairs, mean cell accuracy and the median number of
wrong cells, plus the rank correlation of cell accuracy with grid size. Then the accuracy without the easy cells: the
predicted shape, and on the pairs whose output has the input's shape, the accuracy on the cells the rule changes
(output != input) and on those it keeps, with the copy-the-input baseline's cell accuracy for reference.

    python scripts/analysis/cell_by_size.py checkpoints/run_X/step_N.pt [...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arc.data import Store  # noqa: E402
from arc.evaluate import load_weights, run, version_set  # noqa: E402
from arc.train import build_model, load_config  # noqa: E402

BINS = [(1, 25), (26, 100), (101, 225), (226, 400), (401, 625), (626, 900)]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    dev = torch.device('cuda')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cfg = load_config(ck['config'])
        store = Store(cfg['data'])
        model = build_model(cfg, dev)
        load_weights(model, ck, 'ema')
        model.layer_fn = torch.compile(model.layer, dynamic=False)
        _, recs, grids = run(model, store, version_set(store, 'identity'), 96, cfg['loops'], device=dev)
        n = np.array([r[7] for r in recs])                          # cells of the true grid
        acc = np.array([r[6] for r in recs]) / np.maximum(n, 1)
        wrong = n - np.array([r[6] for r in recs])
        print(f"== {path} ({Path(cfg['data']).stem}, step {ck['step']}): {len(recs)} test pairs, "
              f"spearman(cell acc, grid cells) {spearman(acc, n):+.3f}")
        print(f"   {'cells':>9} {'pairs':>5} {'exact':>5} {'cell acc':>8} {'med wrong':>9} {'acc>=.95':>8}")
        for a, b in BINS:
            s = (n >= a) & (n <= b)
            if s.any():
                print(f"   {a:>4}-{b:<4} {s.sum():>5} {(wrong[s] == 0).sum():>5} {acc[s].mean():>8.3f} "
                      f"{int(np.median(wrong[s])):>9} {(acc[s] >= .95).sum():>8}")
        shape_ok, same, chg, keep, copy, nchg = [], [], [], [], [], []
        for r in recs:
            t, j = r[0], r[1]
            x, y = store.grids(int(store.z['task_test_start'][t]) + int(j))
            g = grids[r[3]]
            shape_ok.append(g.shape == y.shape)
            if x.shape != y.shape:
                continue
            c = y != x
            same.append(len(same))
            copy.append(float((~c).mean()))
            nchg.append(int(c.sum()))
            ok = (g == y) if g.shape == y.shape else np.zeros(y.shape, bool)
            chg.append(float(ok[c].mean()) if c.any() else np.nan)
            keep.append(float(ok[~c].mean()) if (~c).any() else np.nan)
        chg, keep, copy = map(np.array, (chg, keep, copy))
        print(f"   predicted shape right: {np.mean(shape_ok):.3f}; output shape = input shape: {len(same)} pairs, "
              f"changed cells median {int(np.median(nchg))} ({np.median(1 - copy):.1%} of the grid)")
        print(f"   on those: changed-cell acc {np.nanmean(chg):.3f} (median {np.nanmedian(chg):.3f}), kept-cell acc "
              f"{np.nanmean(keep):.3f}, copy-the-input baseline cell acc {copy.mean():.3f}")
        del model
        torch.cuda.empty_cache()
        torch._dynamo.reset()


if __name__ == '__main__':
    main()
