"""One training segment (forward + backward) per lane on one H100 at TRM's ARC depth: 30 layer passes, the last 10 with
gradients. (a) a TRM-like block (post-norm, SDPA flash attention BF16, SwiGLU, hidden 512, 8 heads, no RoPE), T = 916
(900 cells + 16 puzzle-embedding tokens); (b) our bands block (arc.model, fused kernels, compiled layer, no rule head)
at canvas 21 (T = 882, the closest to 916) and at canvas 30 (T = 1800, the current stacked input/output layout).
Prints ms per segment and ms per lane; x 192 lanes / GPU = s/step at TRM's global batch 768 on 4 GPUs.
usage: speed_trm.py WHICH LANES   (WHICH: trm | bands21 | bands30)"""
import sys
from pathlib import Path, time, math
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import torch
import torch.nn as nn
import torch.nn.functional as F

which, lanes = sys.argv[1], int(sys.argv[2])
dev = torch.device('cuda')
PASSES, GRAD, D, H = int(__import__('os').environ.get('PASSES', 30)), int(__import__('os').environ.get('GRAD', 10)), 512, 8
torch.manual_seed(0)


def rms(x, eps=1e-5):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


class TRMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        inter = int(4 * D * 2 / 3 + 255) // 256 * 256
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.gu = nn.Linear(D, 2 * inter, bias=False)
        self.down = nn.Linear(inter, D, bias=False)

    def forward(self, h):
        B, T, _ = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, H, D // H).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        h = rms(h + self.o(a.transpose(1, 2).reshape(B, T, D)))
        g, u = self.gu(h).chunk(2, -1)
        return rms(h + self.down(F.silu(g) * u))


if which == 'trm':
    T = 916
    blocks = nn.ModuleList([TRMBlock() for _ in range(2)]).to(dev)        # L_layers 2, shared over the cycles
    head = nn.Linear(D, 12).to(dev)
    step_fn = torch.compile(lambda h: blocks[1](blocks[0](h)), dynamic=False)
    x = torch.randn(lanes, T, D, device=dev)
    y = torch.randint(0, 12, (lanes, T), device=dev)

    def segment():
        h = x
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for _ in range((PASSES - GRAD) // 2):
                    h = step_fn(h + x)
            for _ in range(GRAD // 2):
                h = step_fn(h + x)
            loss = F.cross_entropy(head(h).float().flatten(0, 1), y.flatten())
        loss.backward()
    params = list(blocks.parameters())
else:
    from arc.model import Coupled, ModelConfig
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    C = 21 if which == 'bands21' else 30
    m = Coupled(ModelConfig(canvas=C, d=D, heads=H, blocks=PASSES, grad_blocks=GRAD, frames=0,
                            bands=int(__import__('os').environ.get('BANDS', 4)),
                            band_periods={1: (6.0,), 2: (2.0, 20.0), 4: (2.0, 6.0, 20.0, 64.0)}[int(__import__('os').environ.get('BANDS', 4))],
                            fused_bands=True, trace_dtype=__import__('os').environ.get('TRACE', 'float32'))).to(dev)
    m.layer_fn = torch.compile(m.layer, dynamic=False)
    T = m.T
    batch = dict(inputs=torch.randint(0, 13, (lanes, T), device=dev),
                 labels=torch.randint(2, 12, (lanes, T), device=dev),
                 group=torch.arange(lanes, device=dev), demo=torch.zeros(lanes, dtype=torch.bool, device=dev))

    def segment():
        logits, _ = m.segment(None, batch, head=False)
        F.cross_entropy(logits.float().flatten(0, 1), batch['labels'][:, m.O:].flatten()).backward()
    params = list(m.parameters())

for _ in range(3):
    segment()
torch.cuda.synchronize()
n = 10
t0 = time.time()
for _ in range(n):
    segment()
torch.cuda.synchronize()
ms = (time.time() - t0) / n * 1e3
print(f'{which}: T {T}, lanes {lanes}, {ms:.1f} ms/segment, {ms / lanes:.3f} ms/lane, '
      f'x192 lanes = {ms / lanes * 192 / 1e3:.3f} s/step (768 global on 4 GPUs), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')
