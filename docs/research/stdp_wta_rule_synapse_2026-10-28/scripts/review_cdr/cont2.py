"""Adversarial check: is the 16-step circuit advantage on the torus toy just continuation (a decreasing threshold)?
Same fixed Fourier basis, same C step as toy_kernel.py. Arms: FISTA lasso (best fixed threshold), circuit (window beta,
E%-max omega), FISTA/ISTA + geometric continuation thr_t = max(thr, lam0 * rho^t) with lam0 = per-row max |eta e_0|."""
import math, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from toy_g import ltd_rule as LR, data as D
torch.set_num_threads(32)
def infer(U, V, mask, T, mode, beta=0.0, omega=0.0, theta_w=1e-3, rho=None, frac=1.0, s_c=0.5, theta_c=0.01):
    B = U.shape[0]; m = mask.to(U.real.dtype)[:, :, None, None]; U, V = U * m, V * m
    gram = torch.einsum('bkad,bkae->bde', U, U.conj()); lam = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-6)
    W = torch.zeros(B, 16, 16, dtype=U.dtype); Y, t = W, 1.0; C = torch.full((B, 4, 4), 0.25); Xi = torch.zeros_like(W)
    lam0 = None
    for it in range(T):
        base = Y if mode in ('fista', 'fista_cont') else W
        YU = U @ base.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), YU)) * m
        back = torch.einsum('bca,bkcd->bkad', C.to(U.dtype), E)
        cn = C.sum(2).amax(1).clamp(min=1e-3); eta = (1.0 / (lam * cn))[:, None, None]
        e = torch.einsum('bkad,bkae->bde', back, U.conj())
        if mode in ('window', 'hb_cont'):
            Xi = beta * Xi + e; Kb = 1.0 / (1.0 - beta); u = W + eta * Xi
            R = (LR.soft(u, eta * Kb * theta_w) - W).abs().amax(-1, keepdim=True) / Kb
            thr = eta * Kb * theta_w + Kb * omega * R
            if mode == 'hb_cont':
                if lam0 is None: lam0 = (eta * e).abs().amax(-1, keepdim=True)
                thr = torch.maximum(thr, frac * lam0 * rho ** it)
            Wn = LR.soft(u, thr)
        else:
            u = base + eta * e
            thr = eta * theta_w
            if mode.endswith('_cont'):
                if lam0 is None: lam0 = (eta * e).abs().amax(-1, keepdim=True)   # per-row lambda_max at W = 0
                thr = torch.maximum(thr, frac * lam0 * rho ** it)
            Wn = LR.soft(u, thr)
        if mode in ('fista', 'fista_cont'):
            tn = (1 + math.sqrt(1 + 4 * t * t)) / 2; Y, W, t = Wn + ((t - 1) / tn) * (Wn - W), Wn, tn
        else:
            W = Wn
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
def score(W, C):
    s = LR.read(Uq, W, C, Q); pred = torch.cat([torch.full_like(s[:, :1], 0.5), s], 1).argmax(1)
    ok = (pred == target).all(1).numpy()
    return ok.mean(), ok[group == 0].mean(), ok[group == 1].mean(), ok[group == 2].mean()
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16
runs = []
for mode in ('ista_cont', 'fista_cont'):
    for frac in (0.5, 0.3):
        for rho in (0.8, 0.85, 0.9):
            for th in (1e-3, 0.05):
                runs.append((mode, dict(theta_w=th, rho=rho, frac=frac)))
for frac in (0.5, 0.3):
    for rho in (0.7, 0.8, 0.85):
        runs.append(('hb_cont', dict(beta=0.85, theta_w=1e-3, rho=rho, frac=frac)))
for mode, kw in runs:
    W, C = infer(U, V, mask, T, mode, **kw); a, g0, g1, g2 = score(W, C)
    print(f'T={T} {mode:10s} {kw}: all {a:.3f} | g0 {g0:.3f} g1 {g1:.3f} g2 {g2:.3f} | W>0.05 med {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f}', flush=True)
