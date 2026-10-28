"""Torus rule-synapse toy, fixed Fourier basis: the W step as the window form
   W+ = soft(W + eta (K * e)_t, eta Kbar (theta_w + omega R)),  K = delta_0 + post-leading lobe beta^tau (tau >= 1),
   Kbar = 1 / (1 - beta);  R = row max |soft(W + eta (K*e), eta Kbar theta_w) - W| / (eta Kbar)
against FISTA + E%-max and inertial (iPiano-type) + E%-max."""
import math, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from toy_g import ltd_rule as LR, data as D, group as G
torch.set_num_threads(16)
def infer(U, V, mask, T, mode, beta=0.0, omega=0.5, theta_w=1e-3, s_c=0.5, theta_c=0.01):
    B = U.shape[0]; m = mask.to(U.real.dtype)[:, :, None, None]; U, V = U * m, V * m
    gram = torch.einsum('bkad,bkae->bde', U, U.conj()); lam = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-6)
    W = torch.zeros(B, 16, 16, dtype=U.dtype); Wp = W; Y, t = W, 1.0; C = torch.full((B, 4, 4), 0.25); Xi = torch.zeros_like(W)
    for it in range(T):
        base = Y if mode == 'fista' else W
        YU = U @ base.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), YU)) * m
        back = torch.einsum('bca,bkcd->bkad', C.to(U.dtype), E)
        cn = C.sum(2).amax(1).clamp(min=1e-3); eta = (1.0 / (lam * cn))[:, None, None]
        e = torch.einsum('bkad,bkae->bde', back, U.conj())
        if mode == 'window':
            Xi = beta * Xi + e; Kb = 1.0 / (1.0 - beta)
            u = W + eta * Xi
            R = (LR.soft(u, eta * Kb * theta_w) - W).abs().amax(-1, keepdim=True) / Kb
            Wn = LR.soft(u, eta * Kb * theta_w + Kb * omega * R)
        elif mode == 'inertial':
            u = W + eta * e + beta * (W - Wp)
            R = (LR.soft(W + eta * e, eta * theta_w) - W).abs().amax(-1, keepdim=True)
            Wn = LR.soft(u, eta * theta_w + omega * R)
        else:
            u = base + eta * e
            R = (LR.soft(u, eta * theta_w) - base).abs().amax(-1, keepdim=True)
            Wn = LR.soft(u, eta * theta_w + omega * R)
        if mode == 'fista':
            tn = (1 + math.sqrt(1 + 4 * t * t)) / 2; Y, W, t = Wn + ((t - 1) / tn) * (Wn - W), Wn, tn
        else:
            Wp, W = W, Wn
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
    return (pred == target).all(1).numpy()
for T in (16, 32, 1000):
    for mode, kw in (('fista', dict()), ('window', dict(beta=0.5)), ('window', dict(beta=0.7)), ('window', dict(beta=0.85)),
                     ('inertial', dict(beta=0.45)), ('inertial', dict(beta=0.7)), ('window', dict(beta=0.7, omega=0.0))):
        if T == 1000 and mode == 'window' and kw.get('omega') == 0.0: continue
        W, C = infer(U, V, mask, T, mode, **kw); ok = score(W, C)
        print(f'T={T:4d} {mode:8s} {kw}: exact g0 {ok[group==0].mean():.3f} g1 {ok[group==1].mean():.3f} g2 {ok[group==2].mean():.3f} | W>0.05 median {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f}', flush=True)
