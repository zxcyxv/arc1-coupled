"""Finite-T certificate on a trained checkpoint: with Theta_0 = 0 and fixed G, Landweber gives Theta_T = Theta* (I - (I - eta G)^T)
when S = Theta* G, so the query error is -Theta* (I - eta G)^T psi*. Compare |(I - eta G)^T phi_q| for today's step (eta = 1, T = 8)
and the proposal's (eta = 0.9 / lmax, T = 32), plus the numerical rank of the demo codes."""
import json, sys, torch
sys.path.insert(0, '/workspace/arc1-coupled')
exec(open('gram_spectrum.py').read().split("q = b['query']")[0].replace("run = sys.argv[1]", "run = sys.argv[1]"))
q = b['query'].bool()
phi, m, demo, group, G = rec[-1]
eye = torch.eye(G.shape[-1], dtype=torch.float64)
out = {k: [] for k in ('rank6', 'rank_rel', 'today', 'prop')}
for l in torch.nonzero(q).flatten().tolist():
    Gl = G[l].double(); ev = torch.linalg.eigvalsh(Gl); lmax = ev[-1]
    out['rank6'].append(int((ev > 1e-6 * lmax).sum())); out['rank_rel'].append(int((ev > lmax / 32).sum()))
    Q = phi[l][m[l] > 0].double()
    A1 = torch.linalg.matrix_power(eye - Gl, 8); A2 = torch.linalg.matrix_power(eye - 0.9 / lmax * Gl, 32)
    out['today'].append((Q @ A1).norm(dim=-1).median().item()); out['prop'].append((Q @ A2).norm(dim=-1).median().item())
med = lambda v: torch.tensor(v, dtype=torch.float64).median().item()
print({k: round(med(v), 3) for k, v in out.items()}, 'lanes', len(out['today']))
