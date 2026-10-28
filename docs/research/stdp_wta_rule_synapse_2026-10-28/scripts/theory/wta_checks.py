"""Numerical checks for the WTA delta rule synapse (sparsemax read + leaky delta write), float64, CPU.

A1 contraction at step 1 (|psi| <= 1): rate vs 1 - eta.
A2 eta -> 0: fixed point -> hard-margin multiclass SVM (min-norm, all demo margins >= 1); dual alpha_ij = p_ij / (eta n).
A3 separation vs interpolation (n >> p): demo-cell argmax fit of least squares (min-norm) vs the WTA write.
B  torus toy, explicit position-operator features: query exactness of lasso / least squares / WTA, converged and 8 blocks.
"""
import json
import sys

import numpy as np
import torch

torch.set_num_threads(16)
torch.set_default_dtype(torch.float64)
out = {}


def sparsemax(z):
    zs, _ = z.sort(-1, descending=True)
    K = z.shape[-1]
    k = torch.arange(1, K + 1, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ks = (1 + k * zs > cs).sum(-1, keepdim=True)
    tau = (cs.gather(-1, ks - 1) - 1) / ks
    return (z - tau).clamp_min(0)


def wta_step(th, psi, Y, w, eta):
    """th [B,C,p], psi [B,n,p], Y [B,n,C], w [B,n]: th <- (1 - eta) th + (1/n) sum w (y - sparsemax(th psi)) psi^T."""
    n = w.sum(-1).clamp_min(1)[:, None, None]
    P = sparsemax(psi @ th.mT)
    return (1 - eta) * th + ((w[..., None] * (Y - P)).mT @ psi) / n, P


def soft(x, t):
    return torch.sign(x) * (x.abs() - t).clamp_min(0)


def margins(th, psi, Y):
    z = psi @ th.mT
    zy = (z * Y).sum(-1)
    zo = z.masked_fill(Y > 0, -1e30).max(-1).values
    return zy - zo


# ------------------------------------------------------------------------------------------------- A: generic problems
def problem(B, n, p, C, roles, noise, seed):
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(B, C * roles, p, generator=g)
    lab = torch.arange(C).repeat_interleave(roles)
    idx = torch.randint(0, C * roles, (B, n), generator=g)
    x = proto.gather(1, idx[..., None].expand(B, n, p)) + noise * torch.randn(B, n, p, generator=g)
    psi = x / x.norm(dim=-1, keepdim=True)
    Y = torch.nn.functional.one_hot(lab[idx], C).double()
    return psi, Y, proto, lab


psi, Y, proto, lab = problem(B=64, n=120, p=64, C=5, roles=6, noise=0.05, seed=0)
w = torch.ones(psi.shape[:2])
# A1: contraction, fixed psi
for eta in (0.125, 0.05):
    th = torch.zeros(psi.shape[0], 5, 64)
    traj = []
    for t in range(3000):
        th, _ = wta_step(th, psi, Y, w, eta)
        traj.append(th.clone())
    star = traj[-1]
    d = torch.stack([(x - star).flatten(1).norm(dim=-1) for x in traj[:400]])     # [T, B]
    rate = ((d[200] / d[100]) ** (1 / 100)).median().item()
    out[f'A1_rate_eta{eta}'] = dict(measured=rate, one_minus_eta=1 - eta)

# A2: eta -> 0 limit; reference hard-margin SVM by a long, small-eta run and by its KKT form
res = {}
for eta in (1e-2, 1e-3, 1e-4):
    th = torch.zeros(psi.shape[0], 5, 64)
    for t in range(int(40 / eta)):
        th, P = wta_step(th, psi, Y, w, eta)
    m = margins(th, psi, Y)
    n = psi.shape[1]
    alpha = P * (1 - Y) / (eta * n)                                   # dual variables of the losers
    recon = torch.einsum('bnc,bnp->bcp', alpha.sum(-1, keepdim=True) * Y - alpha, psi)
    act = alpha > 1e-9
    zz = psi @ th.mT
    gap = ((zz * Y).sum(-1, keepdim=True) - zz)                       # z_y - z_j
    res[eta] = dict(min_margin_median=m.min(-1).values.median().item(),
                    frac_demo_cells_won=(m > 0).double().mean().item(),
                    norm_median=th.flatten(1).norm(dim=-1).median().item(),
                    dual_reconstruction_err=((recon - th).flatten(1).norm(dim=-1) / th.flatten(1).norm(dim=-1)).max().item(),
                    active_gap_max=gap[act].max().item() if act.any() else None)
    res[eta]['theta'] = th
ths = [res[e].pop('theta') for e in (1e-2, 1e-3, 1e-4)]
res['rel_change_1e-3_to_1e-4'] = ((ths[1] - ths[2]).flatten(1).norm(dim=-1) / ths[2].flatten(1).norm(dim=-1)).median().item()
out['A2_eta_to_0'] = {str(k): v for k, v in res.items()}

# A3: separation vs interpolation, n >> p, finite blocks and converged
res = {}
for roles in (2, 6, 20):
    psi, Y, proto, lab = problem(B=64, n=240, p=64, C=5, roles=roles, noise=0.05, seed=1)
    w = torch.ones(psi.shape[:2])
    G = psi.mT @ psi / psi.shape[1]
    S = Y.mT @ psi / psi.shape[1]
    ls = S @ torch.linalg.pinv(G)                                     # min-norm least squares (lambda = 0 limit)
    lw = torch.zeros_like(ls)
    for t in range(8):                                                # the deployed lambda = 0 head: 8 Landweber steps
        lw = lw - (lw @ G - S)
    th8 = torch.zeros_like(ls)
    for t in range(8):
        th8, _ = wta_step(th8, psi, Y, w, 0.125)
    th = torch.zeros_like(ls)
    for t in range(20000):
        th, _ = wta_step(th, psi, Y, w, 1e-3)
    # held-out cells from the same roles (query cells in the demos' role cones)
    g = torch.Generator().manual_seed(7)
    idx = torch.randint(0, 5 * roles, (64, 200), generator=g)
    xq = proto.gather(1, idx[..., None].expand(64, 200, 64)) + 0.05 * torch.randn(64, 200, 64, generator=g)
    psq = xq / xq.norm(dim=-1, keepdim=True)
    Yq = torch.nn.functional.one_hot(lab[idx], 5).double()
    acc = lambda T, P_, Y_: ((P_ @ T.mT).argmax(-1) == Y_.argmax(-1)).double().mean().item()
    res[f'roles_per_colour_{roles}'] = dict(
        distinct_roles=5 * roles,
        demo_fit=dict(ls_minnorm=acc(ls, psi, Y), landweber8=acc(lw, psi, Y), wta8=acc(th8, psi, Y), wta_conv=acc(th, psi, Y)),
        heldout_cells=dict(ls_minnorm=acc(ls, psq, Yq), landweber8=acc(lw, psq, Yq), wta8=acc(th8, psq, Yq), wta_conv=acc(th, psq, Yq)))
out['A3_separation_vs_interpolation'] = res
print(json.dumps(out, indent=1), flush=True)

# ------------------------------------------------------------------------------------------------- B: torus toy
sys.path.insert(0, '/workspace/arc1-coupled')
from toy import torus_group as TG                                     # noqa: E402

z = np.load('/workspace/arc1-coupled/data/gtorus_instances.npz')
N_I = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
din, dout, tin, tout = (torch.as_tensor(z[k][:N_I].astype(np.int64)) for k in ('demo_in', 'demo_out', 'test_in', 'test_out'))
grp = z['group'][:N_I]
SRC = torch.as_tensor(TG.SRC)                                         # [128, 16]: output cell o reads input cell SRC[t, o]


def feats(grid):                                                      # grid [B, K, 16] -> psi [B, K*16, 128*5] unit
    Bk = grid.shape[:-1]
    g = grid[..., SRC.T]                                              # [B, K, 16(o), 128(t)]
    f = torch.nn.functional.one_hot(g, 5).double().reshape(*Bk, 16, 128 * 5) / np.sqrt(128)
    return f.reshape(Bk[0], -1, 128 * 5)


psi_d = feats(din.reshape(N_I, 3, 16))                                # [B, 48, 640]
Y_d = torch.nn.functional.one_hot(dout.reshape(N_I, 48), 5).double()
psi_q = feats(tin.reshape(N_I, 1, 16))                                # [B, 16, 640]
Y_q = tout.reshape(N_I, 16)
w = torch.ones(N_I, 48)
nd = 48.0
G = psi_d.mT @ psi_d / nd                                            # [B, 640, 640]
S = Y_d.mT @ psi_d / nd
L = torch.linalg.eigvalsh(G)[..., -1][:, None, None]


def exact(T):
    pred = (psi_q @ T.mT).argmax(-1)
    return (pred == Y_q).all(-1).double()


def lasso(lam, iters, step):
    th = torch.zeros(N_I, 5, 640)
    yk, tk = th.clone(), 1.0
    for _ in range(iters):                                            # FISTA
        th_new = soft(yk - step * (yk @ G - S), step * lam)
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        yk = th_new + (tk - 1) / tk1 * (th_new - th)
        th, tk = th_new, tk1
    return th


resB = {}
groups = {'all': np.ones(N_I, bool), 'train': grp == 0, 'heldout_combo': grp == 1, 'heldout_t': grp == 2}
summ = lambda e: {k: round(float(e[torch.as_tensor(v)].mean()), 4) for k, v in groups.items()}
resB['lasso_1e-3_fista3000'] = summ(exact(lasso(1e-3, 3000, 1 / L)))
resB['lasso_1e-2_fista3000'] = summ(exact(lasso(1e-2, 3000, 1 / L)))
resB['ls_minnorm'] = summ(exact(S @ torch.linalg.pinv(G)))
th = torch.zeros(N_I, 5, 640)
for _ in range(8):
    th = soft(th - (th @ G - S), 1e-3)
resB['lasso_deployed_8steps_step1_lam1e-3'] = summ(exact(th))
for eta, T in ((0.125, 8), (0.125, 2000), (1e-2, 4000), (1e-3, 40000)):
    th = torch.zeros(N_I, 5, 640)
    for _ in range(T):
        th, _ = wta_step(th, psi_d, Y_d, w, eta)
    m = margins(th, psi_d, Y_d)
    resB[f'wta_eta{eta}_T{T}'] = dict(summ(exact(th)), demo_cells_won=round((m > 0).double().mean().item(), 4))
out['B_torus_position_operator'] = resB
print(json.dumps(out['B_torus_position_operator'], indent=1), flush=True)
