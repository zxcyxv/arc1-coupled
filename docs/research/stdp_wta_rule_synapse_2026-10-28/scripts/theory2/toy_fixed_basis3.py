import math, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from toy_g import ltd_rule as LR, data as D, group as G
torch.set_num_threads(16)
def infer(U, V, mask, T, theta_w=1e-3, s_c=0.5, theta_c=0.01, omega=0.5, mom=False, dterm=0.0):
    B = U.shape[0]; m = mask.to(U.real.dtype)[:, :, None, None]; U, V = U * m, V * m
    gram = torch.einsum('bkad,bkae->bde', U, U.conj()); lam = torch.linalg.eigvalsh(gram)[:, -1].clamp(min=1e-6)
    W = torch.zeros(B, 16, 16, dtype=U.dtype); Y, t = W, 1.0; C = torch.full((B, 4, 4), 0.25); e_prev = None
    for it in range(T):
        base = Y if mom else W
        YU = U @ base.transpose(-1, -2)[:, None]
        E = (V - torch.einsum('bca,bkad->bkcd', C.to(U.dtype), YU)) * m
        back = torch.einsum('bca,bkcd->bkad', C.to(U.dtype), E)
        cn = C.sum(2).amax(1).clamp(min=1e-3); eta = (1.0 / (lam * cn))[:, None, None]
        e = torch.einsum('bkad,bkae->bde', back, U.conj())
        d = eta * (e + (dterm * (e - e_prev) if (dterm and e_prev is not None) else 0)); e_prev = e
        u = base + d
        R = (LR.soft(base + eta * e, eta * theta_w) - base).abs().amax(-1, keepdim=True)
        Wn = LR.soft(u, eta * theta_w + omega * R)
        if mom:
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
for T in (16, 24, 32):
    for kw in (dict(), dict(omega=0.0, mom=True), dict(mom=True), dict(dterm=0.5), dict(dterm=-0.5), dict(dterm=1.0), dict(omega=0.4), dict(omega=0.6)):
        W, C = infer(U, V, mask, T, **kw)
        s = LR.read(Uq, W, C, Q); pred = torch.cat([torch.full_like(s[:, :1], 0.5), s], 1).argmax(1)
        ok = (pred == target).all(1).numpy()
        print(f'T={T:3d} {kw}: exact all {ok.mean():.3f} | W>0.05 median {(W.abs() > 0.05).sum((1, 2)).float().median().item():.0f}', flush=True)
