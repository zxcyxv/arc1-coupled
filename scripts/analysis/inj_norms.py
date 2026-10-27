"""Per-token RMS on the output canvas of the eval queries, block by block: carried h, sqrt(d) emb (all 'unknown' there),
sqrt(d) W_fb r. usage: inj_norms.py RUN"""
import sys
from pathlib import Path, math
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import torch
from arc.small import Episodes
from arc.model import Coupled, ModelConfig
run = sys.argv[1]
ck = torch.load(run + '/last.pt', map_location='cuda', weights_only=False)
meta = ck['meta']
model = Coupled(ModelConfig(**meta['model'])).cuda()
sd = dict(ck['model']); sd.update(ck['ema']); model.load_state_dict(sd); model.eval()
data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
O, sd_ = model.O, math.sqrt(model.d)
acc = {}
orig = model.block_step
blk = [0]
def bs(h, emb, fb, *a, **k):
    blk[0] += 1
    q = cur['q']
    rms = lambda t: float(t[q][:, O:].float().pow(2).mean(-1).sqrt().mean())
    e = acc.setdefault(blk[0], [0.0, 0.0, 0.0, 0])
    e[0] += rms(h); e[1] += rms(sd_ * emb); e[2] += float((sd_ * fb)[q].float().pow(2).mean(-1).sqrt().mean()) if fb is not None else 0.0; e[3] += 1
    return orig(h, emb, fb, *a, **k)
model.block_step = bs
with torch.no_grad():
    for b in data.eval_batches(16):
        b = {k: v.cuda() for k, v in b.items()}
        cur = dict(q=b['query'].bool()); blk[0] = 0
        model.segment(None, b)
print(run.split('/')[-1], 'output canvas, per-token RMS   h | sqrt(d) x_emb | sqrt(d) W_fb r')
for i, (a, b_, c, n) in sorted(acc.items()):
    print(f'  block {i}: {a/n:.3f} | {b_/n:.3f} | {c/n:.3f}')
