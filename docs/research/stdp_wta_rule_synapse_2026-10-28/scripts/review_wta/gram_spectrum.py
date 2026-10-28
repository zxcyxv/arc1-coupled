"""Spectrum of the rule head's leave-one-out Gram G on a trained <=10 checkpoint (CPU, FP32, eval batches):
lmax(G) vs tr G (the step-size claim), effective rank, and |(I - P_D) phi_query| (span certificate)."""
import json, sys, torch
sys.path.insert(0, '/workspace/arc1-coupled')
from arc.model import Coupled, ModelConfig
from arc.small import Episodes
run = sys.argv[1]
cfg = json.load(open(f'{run}/config.json'))
ck = torch.load(f'{run}/last.pt', map_location='cpu', weights_only=False)
data = Episodes(cfg['data'] if cfg['data'].startswith('/') else '/workspace/arc1-coupled/' + cfg['data'], seed=cfg['seed'], num_aug=cfg['num_aug'], D=cfg['D'])
mc = dict(dict(canvas=data.C), **cfg['model']); mc['fused_bands'] = False
torch.manual_seed(0)
model = Coupled(ModelConfig(**mc))
sd = ck.get('ema') or ck['model']
missing = model.load_state_dict(sd, strict=False); print('load', missing)
model.eval()
rec = []
orig = model.statistics
def stats(phi, y, m, demo, group, ng=None, kap=None):
    out = orig(phi, y, m, demo, group, ng, kap)
    rec.append((phi.detach().clone(), m.detach().clone(), demo.detach().clone(), group.detach().clone(), out[0].detach().clone()))
    return out
model.statistics = stats
torch.set_grad_enabled(False)
b = next(iter(data.eval_batches(16, 'test')))
model.segment(None, b)
q = b['query'].bool()
for i, (phi, m, demo, group, G) in enumerate(rec):
    ev = torch.linalg.eigvalsh(G[q].double())
    lmax, tr = ev[:, -1], ev.sum(-1)
    erank = tr / lmax
    # span certificate on query cells: projector onto span of the other demo lanes' labelled phis
    res = []
    for l in torch.nonzero(q).flatten().tolist():
        oth = (group == group[l]) & (demo > 0)
        oth[l] = False
        D = torch.cat([phi[j][m[j] > 0] for j in torch.nonzero(oth).flatten().tolist()]).double()
        Q = phi[l][m[l] > 0].double()
        U, s, Vh = torch.linalg.svd(D, full_matrices=False)
        r = int((s > 1e-6 * s[0]).sum()); V = Vh[:r]
        resid = (Q - (Q @ V.T) @ V).norm(dim=-1)
        res.append(resid.median().item())
    print(f'block {i}: lmax med {lmax.median():.4f} tr med {tr.median():.4f} 1/lmax / 1 = {(1/lmax).median():.1f}x  tr/lmax (eff. rank) med {erank.median():.1f}  query |(I-P_D)phi| med {torch.tensor(res).median():.3f}')
