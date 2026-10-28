"""Torus toy, explicit position-operator features (the regime where the old theory says l1 selection matters):
query exactness of lasso / min-norm least squares / the WTA delta write (sparsemax + leak), converged and 8 blocks.
Cell-space products only (psi [B, 48, 640]), float64, CPU.

Features of output cell o: psi(o) = concat over the 128 transforms t' of onehot5(input colour at SRC[t', o]) / sqrt(128),
so |psi| = 1 and the true rule is 5-sparse: Theta*[c(i), (t, i)] = 1.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/arc1-coupled')
from toy import torus_group as TG  # noqa: E402

torch.set_num_threads(32)
torch.set_default_dtype(torch.float64)
N_I = int(sys.argv[1]) if len(sys.argv) > 1 else 512


def sparsemax(z):
    zs, _ = z.sort(-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ks = (1 + k * zs > cs).sum(-1, keepdim=True)
    return (z - (cs.gather(-1, ks - 1) - 1) / ks).clamp_min(0)


def soft(x, t):
    return torch.sign(x) * (x.abs() - t).clamp_min(0)


z = np.load('/workspace/arc1-coupled/data/gtorus_instances.npz')
sel = np.random.default_rng(0).permutation(len(z['group']))[:N_I]
din, dout, tin, tout = (torch.as_tensor(z[k][sel].astype(np.int64)) for k in ('demo_in', 'demo_out', 'test_in', 'test_out'))
grp = z['group'][sel]
SRC = torch.as_tensor(TG.SRC)


def feats(grid):
    B, K = grid.shape[:2]
    g = grid[..., SRC.T]                                              # [B, K, 16(o), 128(t')]
    return torch.nn.functional.one_hot(g, 5).double().reshape(B, K * 16, 640) / np.sqrt(128)


P_d = feats(din.reshape(N_I, 3, 16))                                  # [B, 48, 640]
Y_d = torch.nn.functional.one_hot(dout.reshape(N_I, 48), 5).double()
P_q = feats(tin.reshape(N_I, 1, 16))
Y_q = tout.reshape(N_I, 16)
n = 48.0
K48 = P_d @ P_d.mT / n                                               # [B, 48, 48]
L = torch.linalg.eigvalsh(K48)[..., -1][:, None, None]               # lambda_max(G) = lambda_max(K48)
groups = {'all': np.ones(N_I, bool), 'train': grp == 0, 'heldout_combo': grp == 1, 'heldout_t': grp == 2}


def report(T, name):
    pred = (P_q @ T.mT).argmax(-1)
    e = (pred == Y_q).all(-1).double()
    zd = P_d @ T.mT
    zy = (zd * Y_d).sum(-1)
    zo = zd.masked_fill(Y_d > 0, -1e30).max(-1).values
    res = {k: round(float(e[torch.as_tensor(v)].mean()), 4) for k, v in groups.items()}
    res['demo_cells_won'] = round(float((zy > zo).double().mean()), 4)
    res['nnz_per_colour_row'] = round(float((T.abs() > 1e-6).double().sum(-1).mean()), 1)
    print(name, json.dumps(res), flush=True)
    return res


out = {'n_instances': N_I}
grad = lambda T: ((P_d @ T.mT - Y_d).mT @ P_d) / n                   # T G - S in cell space
out['ls_minnorm'] = report(((torch.linalg.pinv(K48) @ Y_d).mT @ P_d) / n, 'ls_minnorm')
th = torch.zeros(N_I, 5, 640)
for _ in range(8):
    th = soft(th - grad(th), 1e-3)
out['lasso_deployed_8_step1_lam1e-3'] = report(th, 'lasso_deployed_8')
for lam in (1e-2, 1e-3):
    th = torch.zeros(N_I, 5, 640)
    yk, tk = th.clone(), 1.0
    for _ in range(3000):                                             # FISTA, step 1 / lambda_max
        new = soft(yk - grad(yk) / L, lam / L)
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        yk = new + (tk - 1) / tk1 * (new - th)
        th, tk = new, tk1
    out[f'lasso_lam{lam}_fista3000'] = report(th, f'lasso_lam{lam}')
for eta, T in ((0.125, 8), (0.125, 300), (1e-2, 3000), (1e-3, 15000)):
    th = torch.zeros(N_I, 5, 640)
    for _ in range(T):
        th = (1 - eta) * th + ((Y_d - sparsemax(P_d @ th.mT)).mT @ P_d) / n
    out[f'wta_eta{eta}_T{T}'] = report(th, f'wta_eta{eta}_T{T}')
print(json.dumps(out, indent=1))
