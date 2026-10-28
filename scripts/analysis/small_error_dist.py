"""How a <= 10 harness checkpoint's errors are spread over the test pairs (eager, no compile): wrong cells per pair
(labelled cells incl. EOS) binned, the dispersion var/mean, and on pairs whose output has the input's shape the accuracy
on the cells the rule changes and on those it keeps.

    python scripts/analysis/small_error_dist.py checkpoints/small/b8g4_rope1d_outffn_wdmat_step200000_ema.pt ...
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arc.model import Coupled, ModelConfig  # noqa: E402
from arc.small import Episodes  # noqa: E402

PAD = 1
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
for path in sys.argv[1:]:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    meta = ck['meta']
    model = Coupled(ModelConfig(**meta['model'])).cuda()
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'], strict=False)
    model.eval()
    data = Episodes(meta['data'], N=meta['model']['canvas'] - 1, D=meta['D'])
    W, N, CH, CHN, KP, KPN, GC, GN = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for b in data.eval_batches(meta['eval_groups']):
            b = {k: v.cuda() for k, v in b.items()}
            logits, _ = model.segment(None, b)
            q = b['query']
            O = logits.shape[1]
            y = b['labels'][q, -O:].long(); x = b['inputs'][q, :O].long()
            pred = logits[q].argmax(-1)
            lab = y != PAD
            ok = pred == y
            W += (lab & ~ok).sum(-1).tolist(); N += lab.sum(-1).tolist()
            for i in range(len(y)):
                li = lab[i]
                if not torch.equal(li, x[i] != PAD):          # different shape: skip changed/kept split
                    continue
                c = li & (y[i] != x[i]); k = li & (y[i] == x[i])
                CH.append(int((ok[i] & c).sum())); CHN.append(int(c.sum())); KP.append(int((ok[i] & k).sum())); KPN.append(int(k.sum()))
    W, N = np.array(W), np.array(N)
    ex = W == 0
    print(f"== {path.split('/')[-1]}: {len(W)} pairs, exact {ex.sum()}, cell acc (labelled) {1 - W.sum() / N.sum():.4f}")
    print(f"   wrong cells per pair: 0: {ex.sum()}, 1: {(W == 1).sum()}, 2-3: {((W >= 2) & (W <= 3)).sum()}, "
          f"4-10: {((W >= 4) & (W <= 10)).sum()}, 11-30: {((W >= 11) & (W <= 30)).sum()}, >30: {(W > 30).sum()}")
    nz = W[~ex]
    print(f"   unsolved pairs: mean wrong {nz.mean():.1f}, median {np.median(nz):.0f}, wrong fraction {(nz / N[~ex]).mean():.3f}; "
          f"dispersion var/mean over all pairs {W.var() / W.mean():.1f}")
    CH, CHN, KP, KPN = map(np.array, (CH, CHN, KP, KPN))
    print(f"   same-shape pairs {len(CH)}: changed-cell acc {CH.sum() / CHN.sum():.4f} ({CHN.sum()} cells), "
          f"kept-cell acc {KP.sum() / KPN.sum():.4f} ({KPN.sum()} cells)")
    del model; torch.cuda.empty_cache()
