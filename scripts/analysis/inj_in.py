"""Input canvas, eval batches (all lanes), per-token RMS block by block: carried h, embed_scale * emb, and the cosine
between h and the embedding (how much of the input the carried state still holds). Also the trained embedding rows'
RMS * embed_scale per token (1.0 at init). usage: inj_in.py RUN"""
import sys
from pathlib import Path
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
O, s = model.O, model.embed_scale
E = model.embed.weight * s
print(run.split('/')[-1], 'step', ck['step'], '| embed rows RMS x scale (pad, colours 0-9):',
      ' '.join(f'{x:.2f}' for x in E.pow(2).mean(-1).sqrt()[1:12].tolist()))
acc = {}
orig = model.block_step
blk = [0]
def bs(h, emb, fb, *a, **k):
    blk[0] += 1
    m = cur['m']
    hi, ei = h[:, :O].float()[m], (s * emb[:, :O]).float()[m]
    e = acc.setdefault(blk[0], [0.0, 0.0, 0.0, 0])
    e[0] += float(hi.pow(2).mean(-1).sqrt().mean()); e[1] += float(ei.pow(2).mean(-1).sqrt().mean())
    e[2] += float(torch.nn.functional.cosine_similarity(hi, ei, dim=-1).mean()); e[3] += 1
    return orig(h, emb, fb, *a, **k)
model.block_step = bs
with torch.no_grad():
    for b in data.eval_batches(16):
        b = {k: v.cuda() for k, v in b.items()}
        x = b['inputs'][:, :O].long()
        cur = dict(m=(x >= 2) & (x <= 11)); blk[0] = 0           # colour cells of the input canvas
        model.segment(None, b)
print('  input-canvas colour cells: block  h | scale*x_emb | cos(h, x_emb)')
for i, (a, b_, c, n) in sorted(acc.items()):
    print(f'    {i}: {a/n:.3f} | {b_/n:.3f} | {c/n:.3f}')
