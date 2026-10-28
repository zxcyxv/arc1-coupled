"""On a trained <=10 checkpoint (8/4 1D fixed RoPE, EMA 50k), per block: Gram spectrum of the leave-one-out G on query
lanes (n_l vs p, lmax, tr, smallest eigenvalue), phi drift between blocks, and the KKT residual of the carried theta
against the lasso on that block's statistics (is the head near stationarity after 8 one-step blocks?)."""
import json, sys, torch
sys.path.insert(0, '/workspace/arc1-coupled')
from arc.model import Coupled, ModelConfig, soft
from arc.small import Episodes
torch.set_num_threads(32)
cfg = json.load(open('/workspace/arc1-coupled/docs/evidence/small/b8g4_rope1d/config.json'))
ck = torch.load('/workspace/arc1-coupled/checkpoints/small/b8g4_rope1d_step50000_ema.pt', map_location='cpu', weights_only=False)
data = Episodes(cfg['data'], seed=cfg['seed'], num_aug=cfg['num_aug'], D=cfg['D'])
mc = dict(cfg['model']); mc['fused_bands'] = False; mc['canvas'] = data.C
model = Coupled(ModelConfig(**mc)); print(model.load_state_dict(ck['ema'], strict=False)); model.eval()
lam = (model.cfg.lam_floor + torch.nn.functional.softplus(model.lam_raw)).item(); print('lam', lam)
rec = []; orig_stats = model.statistics; orig_step = model.step
def stats(phi, y, m, demo, group, ng=None, kap=None):
    out = orig_stats(phi, y, m, demo, group, ng, kap); rec.append(dict(phi=phi.clone(), m=m.clone(), demo=demo.clone(), out=[o.clone() for o in out])); return out
def step(theta, G, S, gam=None, ext=None):
    res = orig_step(theta, G, S, gam, ext); rec[-1]['theta_in'] = theta.clone(); rec[-1]['res'] = [r.clone() for r in res]; return res
model.statistics, model.step = stats, step
torch.set_grad_enabled(False)
b = next(iter(data.eval_batches(16, 'test')))
model.segment(None, b)
q = b['query'].bool()
nl = None
prev = None
for i, r in enumerate(rec):
    G, S, C, H, s = r['out']; th, gam = r['res']
    w = r['m'] * r['demo'][:, None]
    grp = b['group'].long(); L = len(grp)
    oth = ((grp[:, None] == grp[None]) & ~torch.eye(L, dtype=torch.bool)).float(); n = oth @ w.sum(-1)
    ev = torch.linalg.eigvalsh(G[q].double())
    # KKT residual of the head's theta after this block's step for the lasso on this block's statistics (theta part, Gamma fixed)
    g = S - th @ G - (gam[..., None, None] * C).sum(1)
    on = th != 0
    kk_on = (g - lam * th.sign()).abs()[on].max().item() if on.any() else 0.0
    kk_off = (g.abs() - lam).clamp(min=0)[~on].max().item()
    drift = None if prev is None else ((r['phi'] - prev).norm(dim=-1)[r['m'] > 0]).median().item()
    prev = r['phi']
    print(f'block {i}: n_l med {n[q].median():.0f} (p={G.shape[-1]}) | lmax {ev[:, -1].median():.3f} tr {ev.sum(-1).median():.3f} '
          f'lmin {ev[:, 0].median():.2e} lmin/lmax {(ev[:, 0] / ev[:, -1]).median():.2e} | KKT theta on {kk_on:.3f} off {kk_off:.3f} '
          f'| |g| max {g.abs().max():.3f} | nnz/row {(th[q] != 0).sum(-1).float().mean():.1f} | phi drift {drift}', flush=True)
