"""Does the input canvas carry the answer? Eval batches (every lane; perception never sees any output label), cells
(i, j) that are colour cells in both the input grid and the output grid at the same canvas position. Per block k (state
entering block k): h_in(i,j) and u_in = h_in + scale*emb_in on the input canvas, h_out(i,j) on the output canvas.
  nearest: argmax_c <u_in, E_c> equals y (the answer colour) / equals x (the input colour), on changed cells (y != x)
  ridge probe -> y, fit on even batches, tested on odd batches, accuracy on changed / unchanged cells, from
  [x one-hot only] (baseline), [h_in], [h_out].
usage: ans_trace.py RUN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import torch
import torch.nn.functional as F
from arc.small import Episodes
from arc.model import Coupled, ModelConfig
run = sys.argv[1]
ck = torch.load(run + '/last.pt', map_location='cuda', weights_only=False)
meta = ck['meta']
model = Coupled(ModelConfig(**meta['model'])).cuda()
sd = dict(ck['model']); sd.update(ck['ema']); model.load_state_dict(sd); model.eval()
data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
O, s = model.O, model.embed_scale
E = model.embed.weight.detach()[2:12].float()
H_in, H_out, U = {}, {}, {}
X, Y, B, Q = [], [], [], []
orig = model.block_step
blk = [0]
def bs(h, emb, fb, *a, **k):
    blk[0] += 1
    m = cur['m']
    H_in.setdefault(blk[0], []).append(h[:, :O].float()[m].cpu())
    H_out.setdefault(blk[0], []).append(h[:, O:].float()[m].cpu())
    U.setdefault(blk[0], []).append((h + s * emb)[:, :O].float()[m].cpu())
    return orig(h, emb, fb, *a, **k)
model.block_step = bs
with torch.no_grad():
    for bi, b in enumerate(data.eval_batches(16)):
        b = {k: v.cuda() for k, v in b.items()}
        x = b['inputs'][:, :O].long(); y = b['labels'][:, O:].long()
        m = (x >= 2) & (x <= 11) & (y >= 2) & (y <= 11)
        cur = dict(m=m); blk[0] = 0
        X.append((x[m] - 2).cpu()); Y.append((y[m] - 2).cpu()); B.append(torch.full((int(m.sum()),), bi))
        Q.append(b['query'].bool()[:, None].expand_as(m)[m].cpu())
        model.segment(None, b)
X, Y, B, Q = map(torch.cat, (X, Y, B, Q))
ch = Y != X
tr, te = B % 2 == 0, B % 2 == 1
print(run.split('/')[-1], f'cells {len(Y)}, changed {int(ch.sum())} ({float(ch.float().mean()):.2f}), query cells {int(Q.sum())}')

def ridge(F_, lam=1e-2):
    Xd = torch.cat([F_, torch.ones(len(F_), 1)], 1).double()
    oh = F.one_hot(Y, 10).double()
    A = Xd[tr].T @ Xd[tr] + lam * len(Xd[tr]) * torch.eye(Xd.shape[1], dtype=torch.double) / Xd.shape[1]
    W = torch.linalg.solve(A, Xd[tr].T @ oh[tr])
    ok = (Xd @ W).argmax(-1) == Y
    return float(ok[te & ch].float().mean()), float(ok[te & ~ch].float().mean())

xo = F.one_hot(X, 10).float()
print('  baseline probe from x only:  changed %.3f  unchanged %.3f' % ridge(xo))
print('  block | nearest-emb(u_in): =y  =x on changed | probe h_in: changed unchanged | probe h_out: changed unchanged')
for k in sorted(U):
    u = torch.cat(U[k]); hi = torch.cat(H_in[k]); ho = torch.cat(H_out[k])
    near = (u @ E.cpu().T).argmax(-1)
    a = ridge(hi); c = ridge(ho)
    print(f'  {k}: {float((near == Y)[ch].float().mean()):.3f} {float((near == X)[ch].float().mean()):.3f} | '
          f'{a[0]:.3f} {a[1]:.3f} | {c[0]:.3f} {c[1]:.3f}')
