"""How a model's RoPE sees space on the <= 10 lane (input canvas stacked over the output canvas, C = 11): the angles
theta[h, pair, (row, col)] of a checkpoint (EMA) against their initial draw (same seed) and against TRM's fixed 1D
frequencies. Per head, the positional kernel K_h(dr, dc) = mean over pairs cos(theta_row dr + theta_col dc) is the
q.k factor RoPE contributes between two tokens whose (unit, RoPE-free) q and k are aligned: 1 at no offset; how fast it
falls off, how much it rings, and what it gives at the offsets the task needs (the output cell over its own input cell,
dr = C; the four neighbours) show the spatial code. 1D positions are t = row * C + col, so (dr, dc) -> 11 dr + dc.

    python scripts/analysis/rope_space.py RUN_DIR [RUN_DIR ...]"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
from arc.model import Coupled, ModelConfig  # noqa: E402

C = 11
DR, DC = np.meshgrid(np.arange(-2 * C + 1, 2 * C), np.arange(-C + 1, C), indexing='ij')   # every offset on the lane


def theta_of(run):
    ck = torch.load(Path(run) / 'last.pt', map_location='cpu', weights_only=False)
    mc = ck['meta']['model']
    w = ck['ema'].get('block.rope_theta')
    if w is None:                                    # fixed angles: a buffer in the model state
        w = ck['model']['block.rope_theta']
    torch.manual_seed(ck['meta']['seed'])
    init = Coupled(ModelConfig(**mc)).block.rope_theta.detach()
    return w.double().numpy(), init.double().numpy(), mc, ck['step']


def kernel(th, one_d):
    """[H, len(DR), len(DC)] mean_pairs cos(theta . offset)."""
    if one_d:
        off = (DR * C + DC)[None, None]                                   # sequence-index offset
        ang = th[:, :, 0, None, None] * off
    else:
        ang = th[:, :, 0, None, None] * DR[None, None] + th[:, :, 1, None, None] * DC[None, None]
    return np.cos(ang).mean(1)


def describe(name, th, one_d):
    K = kernel(th, one_d)
    i0, j0 = 2 * C - 1, C - 1                                             # index of offset (0, 0)
    far = np.abs(DR) + np.abs(DC) >= 3
    mag = np.hypot(th[..., 0], th[..., 1]) if not one_d else np.abs(th[..., 0]) * np.hypot(C, 1)
    print(f'  {name}')
    print(f'    pair frequency |theta| (rad/cell): median {np.median(mag):.3f}, >1 rad {np.mean(mag > 1):.2f}, '
          f'< 0.05 rad {np.mean(mag < 0.05):.2f}')
    for label, (dr, dc) in (('right neighbour', (0, 1)), ('below neighbour', (1, 0)), ('output over input', (C, 0)),
                            ('output over input + 1 col', (C, 1))):
        v = K[:, i0 + dr, j0 + dc]
        print(f'    K{(dr, dc)} {label:26s}: heads {np.array2string(v, precision=2, floatmode="fixed")}')
    print(f'    off-centre |K| (|dr|+|dc| >= 3): mean {np.abs(K[:, far]).mean():.3f}; '
          f'largest off-centre peak per head {np.array2string(K[:, far].max(1), precision=2, floatmode="fixed")}')
    return K


for run in sys.argv[1:]:
    th, init, mc, step = theta_of(run)
    one_d = bool(mc.get('rope_1d'))
    print(f'== {run} (step {step}, rope {"1D" if one_d else "2D"}, '
          f'{"fixed" if mc.get("rope_fixed", one_d) else "learned"})')
    if not np.allclose(th, init):
        d = th - init
        print(f'  moved from init: mean |delta| {np.abs(d).mean():.3f} rad, max {np.abs(d).max():.3f}; '
              f'|theta| mean {np.abs(th).mean():.3f} (init {np.abs(init).mean():.3f})')
        describe('initial draw', init, one_d)
    describe('trained' if not np.allclose(th, init) else 'fixed', th, one_d)
