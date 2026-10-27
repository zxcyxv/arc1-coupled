"""Sparsity of the lasso's Theta/Gamma on the 278 eval queries for a trained <=10 run (EMA weights, eager, one segment).
Per block: pre-threshold x = theta - grad; fraction of entries with |x| < t for t in (0.01, 0.03, 0.1), relative change
||soft(x, lam) - x|| / ||x|| at the run's lam, and the final Theta's exact-zero fraction. usage: theta_probe.py RUN [lam_eval]"""
import sys
from pathlib import Path, json
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import torch
from arc.small import Episodes, load_config
from arc.model import Coupled, ModelConfig
run = sys.argv[1]
ck = torch.load(run + '/last.pt', map_location='cuda', weights_only=False)
meta = ck['meta']
cfg = load_config(None, [])
model = Coupled(ModelConfig(**meta['model'])).cuda()
sd = dict(ck['model']); sd.update(ck['ema'])
model.load_state_dict(sd)
model.eval()
lam = model.cfg.lam_floor
data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
rec = []
orig = model.step
def step(theta, G, S, gam=None, ext=None):
    g = theta @ G - S
    if gam is not None:
        g = g + (gam[..., None, None] * ext[0]).sum(1)
    x = (theta - g)
    rec.append(('theta', x.detach()))
    out = orig(theta, G, S, gam, ext)
    if gam is not None:
        th, gm = out
        Ck, H, sv = ext
        gg = (th[:, None] * Ck).sum((-2, -1)) + (H * gam[:, None]).sum(-1) - sv
        rec.append(('gam', (gam - gg).detach()))
    return out
model.step = step
stats = {}
with torch.no_grad():
    for b in data.eval_batches(16):
        b = {k: v.cuda() for k, v in b.items()}
        rec.clear()
        model.segment(None, b)
        q = b['query'].bool()
        blk = 0
        for kind, x in rec:
            if kind == 'theta':
                blk += 1
            x = x[q].abs()
            if kind == 'theta':                       # only colour rows the task's demo outputs use
                live = x.amax(-1) > 1e-6
                xs = x[live]                          # [rows, 64]
                pr = (xs.pow(2).sum(-1) ** 2 / xs.pow(4).sum(-1).clamp_min(1e-30))
                e = stats.setdefault(('rows', blk), dict(pr=[], surv=[], nrow=[]))
                e['pr'].append(pr.cpu()); e['surv'].append((xs > max(lam, 0.01)).float().sum(-1).cpu()); e['nrow'].append(live.sum(-1).float().cpu())
            x = x.flatten(1)
            d = stats.setdefault((kind, blk), dict(n=0, lt01=0, lt03=0, lt1=0, num=0.0, den=0.0, med=[]))
            d['n'] += x.numel(); d['lt01'] += int((x < 0.01).sum()); d['lt03'] += int((x < 0.03).sum()); d['lt1'] += int((x < 0.1).sum())
            d['num'] += float((x.clamp(max=lam) ** 2).sum()) if lam > 0 else 0.0
            d['den'] += float((x ** 2).sum())
            d['med'].append(x.median(1).values.cpu())
print(run.split('/')[-1], 'lam', lam)
for (kind, blk), d in sorted(stats.items()):
    if kind == 'rows':
        if blk in (1, 4, 8):
            print(f"  rows  blk{blk}: live rows/lane {torch.cat(d['nrow']).mean():.1f}  participation(of 64) {torch.cat(d['pr']).median():.1f}"
                  f"  entries>0.01 per row {torch.cat(d['surv']).median():.0f}")
        continue
    if blk in (1, 4, 8) or kind == 'gam' and blk in (1, 8):
        print(f"  {kind:5s} blk{blk}: |x|<.01 {d['lt01']/d['n']:.2f}  <.03 {d['lt03']/d['n']:.2f}  <.1 {d['lt1']/d['n']:.2f}"
              f"  median|x| {torch.cat(d['med']).median():.4f}  shrink {((d['num']/d['den'])**.5 if lam else 0):.3f}")
