"""Torus position-operator check, second part: the WTA write (sparsemax residual) WITH the constant depression +
floor (soft threshold) kept, i.e. prox-gradient on sparsemax loss + lam |Theta|_1 (an l1-margin rule)."""
import json, sys
import torch
sys.argv = [sys.argv[0], '512']
exec(open('wta_torus.py').read().split("out = {'n_instances'")[0])
out = {}
gsm = lambda T: ((sparsemax(P_d @ T.mT) - Y_d).mT @ P_d) / n
for lam in (1e-3, 3e-4):
    th = torch.zeros(N_I, 5, 640); yk, tk = th.clone(), 1.0
    for _ in range(3000):
        new = soft(yk - gsm(yk) / L, lam / L)
        tk1 = (1 + (1 + 4 * tk * tk) ** 0.5) / 2
        yk = new + (tk - 1) / tk1 * (new - th); th, tk = new, tk1
    out[f'wta_l1_lam{lam}_fista3000'] = report(th, f'wta_l1_lam{lam}')
th = torch.zeros(N_I, 5, 640)
for _ in range(8):
    th = soft(th - gsm(th), 1e-3)
out['wta_l1_deployed_8_step1_lam1e-3'] = report(th, 'wta_l1_deployed_8')
