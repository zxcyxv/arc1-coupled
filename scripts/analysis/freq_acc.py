"""Cell accuracy on the 278 eval queries bucketed by how often the true colour occurs in the group's demo outputs
(normal eval forward, EMA weights). usage: freq_acc.py RUN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import torch
from arc.small import Episodes
from arc.model import Coupled, ModelConfig, PAD, EOS
run = sys.argv[1]
ck = torch.load(run + '/last.pt', map_location='cuda', weights_only=False)
meta = ck['meta']
model = Coupled(ModelConfig(**meta['model'])).cuda()
sd = dict(ck['model']); sd.update(ck['ema']); model.load_state_dict(sd); model.eval()
data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
edges = [0, 0.05, 0.15, 0.4, 1.01]
ok = torch.zeros(len(edges) - 1); n = torch.zeros(len(edges) - 1); top_ok = torch.zeros(2); top_n = torch.zeros(2)
with torch.no_grad():
    for b in data.eval_batches(16):
        b = {k: v.cuda() for k, v in b.items()}
        logits, _ = model.segment(None, b)
        O = logits.shape[1]
        y = b['labels'][:, -O:].long()
        grid = (y != PAD) & (y != EOS)
        oh = torch.nn.functional.one_hot(y.clamp(0, 12), 13).float() * grid[..., None]
        for g in b['group'].unique():
            lanes = b['group'] == g
            dm = lanes & b['demo'].bool()
            q = lanes & b['query'].bool()
            if not q.any() or not dm.any():
                continue
            cnt = oh[dm].sum((0, 1)); freq = cnt / cnt.sum()
            for l in q.nonzero().flatten():
                c = y[l][grid[l]]; good = (logits[l].argmax(-1) == y[l])[grid[l]].float()
                f = freq[c].cpu(); good = good.cpu()
                for i in range(len(edges) - 1):
                    s = (f >= edges[i]) & (f < edges[i + 1]); ok[i] += good[s].sum(); n[i] += s.sum()
                top = c == freq.argmax()
                top_ok[0] += good[top.cpu()].sum(); top_n[0] += top.sum().cpu(); top_ok[1] += good[~top.cpu()].sum(); top_n[1] += (~top).sum().cpu()
print(run.split('/')[-1], ' '.join(f"[{edges[i]:.2f},{edges[i+1]:.2f}) {ok[i]/n[i]:.3f} (n {int(n[i])})" for i in range(len(edges) - 1)),
      f"| most frequent {top_ok[0]/top_n[0]:.3f} others {top_ok[1]/top_n[1]:.3f}")
