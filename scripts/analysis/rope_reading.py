"""What the bands read takes from position vs content, per test pair of the <= 10 harness, split by outcome.

In a block the read of token o is q~_o (sum_t v_t k~_t^T + W)^T: its instantaneous part weighs token t by
s(o, t) = q~_o . k~_t (unit q, k after RoPE). For every query lane (the 278 test pairs, version 0, EMA weights, one
segment as in the harness's evaluation) and every labelled output cell o, per head and block:
  self     s(o, own input cell) / mean_t |s(o, t)|      the cell of the input canvas at o's (row, col)
  colour   [mean s(o, t) over input cells of o's answer colour - mean over input cells of other colours] / mean |s|
  posfree  share of sum_t |s(o, t)| carried by the position-free RoPE pairs (phase across the lane < pi / 4), i.e. the
           part of the q.k match that is content alone
Lanes are grouped: exact, near (1-3 wrong labelled cells), far (> 3). Run on the non-fused path (same formulas).

    python scripts/analysis/rope_reading.py RUN_DIR [RUN_DIR ...]"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
from arc.model import EOS, PAD, Coupled, ModelConfig  # noqa: E402
from arc.small import Episodes  # noqa: E402


def run(path):
    ck = torch.load(Path(path) / 'last.pt', map_location='cuda', weights_only=False)
    meta = ck['meta']
    mc = dict(meta['model'], fused_bands=False)
    model = Coupled(ModelConfig(**mc)).cuda()
    sd = dict(ck['model']); sd.update(ck['ema']); model.load_state_dict(sd); model.eval()
    data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
    C, O, T, H = model.C, model.O, model.T, model.H
    th = model.block.rope_theta.detach().double().cpu().numpy()                    # [H, P, 2]
    span = (np.abs(th[..., 0]) * (T - 1)) if mc.get('rope_1d') else (np.abs(th[..., 0]) * (2 * C - 1) + np.abs(th[..., 1]) * (C - 1))
    free = torch.tensor(span < math.pi / 4, device='cuda')                        # [H, P] position-free pairs
    dims = free.repeat_interleave(2, dim=1).float()                                # [H, 2P]: our RoPE pairs (2p, 2p + 1)

    rec = []
    orig = Coupled.bands_step

    def wrapped(self, qkv, ek, ev, w, tables=None):
        cos, sin = tables
        unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + self.cfg.eps)
        q, k, _ = qkv.float().view(*qkv.shape[:2], 3, self.H, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
        rec.append((self.rope(unit(q), (cos.float(), sin.float())), self.rope(unit(k), (cos.float(), sin.float()))))
        return orig(self, qkv, ek, ev, w, tables)
    Coupled.bands_step = wrapped

    stats = {g: [] for g in ('exact', 'near', 'far')}
    with torch.no_grad():
        for b in data.eval_batches(16):
            b = {k: v.cuda() for k, v in b.items()}
            rec.clear()
            logits, _ = model.segment(None, b)
            y = b['labels'][:, O:].long()
            x_in = b['inputs'][:, :O].long()
            for l in b['query'].nonzero().flatten().tolist():
                lab = y[l] != PAD
                wrong = int(((logits[l].argmax(-1) != y[l]) & lab).sum())
                g = 'exact' if wrong == 0 else ('near' if wrong <= 3 else 'far')
                cells = ((y[l] != PAD) & (y[l] != EOS)).nonzero().flatten()          # output-canvas grid cells
                ans = y[l, cells]                                                     # model tokens 2..11
                per_block = []
                for qr, kr in rec:                                                    # every block
                    qo = qr[l][:, O + cells]                                          # [H, n, D]
                    s = torch.einsum('hnd,htd->hnt', qo, kr[l])                       # [H, n, T]
                    s_free = torch.einsum('hnd,htd->hnt', qo * dims[:, None], kr[l])
                    norm = s.abs().mean(-1).clamp_min(1e-9)                           # [H, n]
                    self_ = s[:, torch.arange(len(cells)), cells] / norm
                    same = (x_in[l][None, :] == ans[:, None]).float()                 # [n, O] input cells of o's answer colour
                    col = ((x_in[l] >= 2) & (x_in[l] <= 11)).float()[None]
                    other = col - same
                    si = s[:, :, :O]
                    m_same = (si * same).sum(-1) / same.sum(-1).clamp_min(1)
                    m_other = (si * other).sum(-1) / other.sum(-1).clamp_min(1)
                    colour = (m_same - m_other) / norm
                    posfree = s_free.abs().sum(-1) / s.abs().sum(-1).clamp_min(1e-9)
                    per_block.append(torch.stack([self_.mean(1), colour.mean(1), posfree.mean(1)]).cpu())  # [3, H]
                stats[g].append(torch.stack(per_block))                               # [blocks, 3, H]
    Coupled.bands_step = orig
    print(f'== {path}: rope {"1D" if mc.get("rope_1d") else "2D"} '
          f'{"fixed" if mc.get("rope_fixed", mc.get("rope_1d")) else "learned"}, position-free pairs per head '
          f'{free.sum(1).tolist()} of {free.shape[1]}')
    for g, v in stats.items():
        if not v:
            print(f'  {g:5s}: 0 lanes'); continue
        a = torch.stack(v)                                                            # [lanes, blocks, 3, H]
        last, allb = a[:, -1].mean(0), a.mean((0, 1))                                 # [3, H]
        print(f'  {g:5s} {len(v):3d} lanes | last block: self {last[0].mean():+.3f} colour {last[1].mean():+.3f} '
              f'posfree {last[2].mean():.3f} | all blocks: self {allb[0].mean():+.3f} colour {allb[1].mean():+.3f} '
              f'posfree {allb[2].mean():.3f}')
        print(f'        per head (last block)  self   {np.array2string(last[0].numpy(), precision=2, floatmode="fixed")}')
        print(f'                               colour {np.array2string(last[1].numpy(), precision=2, floatmode="fixed")}')
    return stats


for p in sys.argv[1:]:
    run(p)
