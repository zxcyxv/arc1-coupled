"""Existence check (no learning) on the 4x4 torus rule-synapse toy of the old repo (toy_g/ltd_rule.py), fixed Fourier
basis, CPU: the W step as FISTA-lasso (old), ISTA, the competitive circuit (E%-max surplus, no momentum) and hard
one-winner-per-row WTA (IHT k=1). The C step is the old one in every arm."""
import math, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from toy_g import ltd_rule as LR, data as D, group as G
torch.set_num_threads(16)

def infer(U, V, mask, mode, T, s_w=1.0, theta_w=1e-3, s_c=0.5, theta_c=0.01, omega=0.5, k=1):
    B = U.shape[0]; m = mask.to(U.real.dtype)[:, :, None, None]; U, V = U * m, V * m
    gram = torch.einsum('bkad,bkae->bde', U, U.conj()); lam = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-6)
    W = torch.zeros(B, 16, 16, dtype=U.dtype); Y, t = W, 1.0
    C = torch.full((B, 4, 4), 0.25)
    for it in range(T):
        base = Y if mode == 'fista' else W
        YU = U @ base.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), YU)) * m
        back = torch.einsum('bca,bkcd->bkad', C.to(U.dtype), E)
        cn = C.sum(2).amax(1).clamp(min=1e-3)
        eta = (s_w / (lam * cn))[:, None, None]
        u = base + eta * torch.einsum('bkad,bkae->bde', back, U.conj())
        if mode == 'fista':
            Wn = LR.soft(u, eta * theta_w); tn = (1 + math.sqrt(1 + 4 * t * t)) / 2
            Y, W, t = Wn + ((t - 1) / tn) * (Wn - W), Wn, tn
        elif mode == 'ista':
            W = LR.soft(u, eta * theta_w)
        elif mode == 'circuit':
            R = (LR.soft(u, eta * theta_w) - W).abs().amax(-1, keepdim=True)        # per output channel (row)
            W = LR.soft(u, eta * theta_w + omega * R)
        elif mode == 'kwta':
            mag = u.abs(); kth = mag.topk(k, -1).values[..., -1:]
            W = torch.where(mag >= kth, u, torch.zeros_like(u))
        WU = U @ W.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), WU)) * m
        eta_c = (s_c / (WU.abs() ** 2).sum((1, 2, 3)).clamp(min=1e-6))[:, None, None]
        C = LR.soft(torch.clamp(C + eta_c * torch.einsum('bkcd,bkad->bca', E, WU.conj()).real, min=0), eta_c * theta_c)
        C = C / C.sum(1, keepdim=True).clamp(min=1e-6).maximum(torch.ones_like(C[:, :1]))
    return W, C

inst = D.load_instances(); group = inst['group'][inst['task']]
din = torch.as_tensor(inst['demo_in'].reshape(-1, 3, 16).astype(np.int64)); dout = torch.as_tensor(inst['demo_out'].reshape(-1, 3, 16).astype(np.int64))
qin = torch.as_tensor(inst['test_in'].reshape(-1, 16).astype(np.int64)); target = torch.as_tensor(inst['test_out'].reshape(-1, 16).astype(np.int64))
mask = torch.ones(len(din), 3, dtype=torch.bool)
Q = LR.fourier('cpu'); U, V = LR.images(din, Q), LR.images(dout, Q); Uq = LR.images(qin, Q)
for mode, T, kw in (('fista', 150, {}), ('fista', 16, {}), ('ista', 16, {}), ('ista', 150, {}), ('circuit', 16, {}), ('circuit', 150, {}),
                    ('circuit', 16, dict(omega=0.3)), ('kwta', 16, {}), ('kwta', 150, {})):
    t0 = time.time()
    W, C = infer(U, V, mask, mode, T, **kw)
    s = LR.read(Uq, W, C, Q); pred = torch.cat([torch.full_like(s[:, :1], 0.5), s], 1).argmax(1)
    ok = (pred == target).all(1).numpy()
    print(f'{mode:8s} T={T:3d} {kw}: exact g0 {ok[group == 0].mean():.3f} g1 {ok[group == 1].mean():.3f} g2 {ok[group == 2].mean():.3f} | W entries>0.05 median {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f} | {time.time() - t0:.0f}s', flush=True)
