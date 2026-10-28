"""Depth extrapolation of a <= 10 harness checkpoint (arc.small): the test pairs run for K segments in a row, the carry
(hidden state, bands traces, W, Theta, Gamma, read) passed from one segment to the next, so the recursion goes K times
the trained depth. Reports exact / cell / test CE per K and, against K = 1 (the trained depth), the pairs newly solved and
the pairs lost.

    python scripts/analysis/small_extrapolate.py checkpoints/small/b30g10_rope1d_outffn_wdmat_step200000_ema.pt 1 2 4 8 16 32
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arc.model import Coupled, ModelConfig  # noqa: E402
from arc.small import Episodes, query_loss, scores  # noqa: E402


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    ks = [int(x) for x in sys.argv[2:]] or [1, 2, 4, 8, 16]
    meta = ck['meta']
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False                     # as arc.small
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    model = Coupled(ModelConfig(**meta['model'])).to(device)
    missing = model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'], strict=False)
    assert not missing.unexpected_keys, missing
    model.eval()
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    model.layer_fn = torch.compile(model.layer, dynamic=False)
    data = Episodes(meta['data'], N=meta['model']['canvas'] - 1, D=meta['D'])
    batches = [{k: v.to(device) for k, v in b.items()} for b in data.eval_batches(meta['eval_groups'])]
    kmax, n = max(ks), sum(int(b['query'].sum()) for b in batches)
    print(f"{sys.argv[1]} (step {ck['step']}, {meta['model']['blocks']} blocks per segment), {n} test pairs", flush=True)
    carries, base = [None] * len(batches), None
    with torch.no_grad():
        for k in range(1, kmax + 1):                                  # every batch one segment further, results per K
            ok, cell, n_cell, ce = [], 0, 0, 0.0
            for i, b in enumerate(batches):
                logits, carries[i] = model.segment(carries[i], b)
                if k in ks:
                    y = b['labels'][b['query'], -logits.shape[1]:]
                    pred = logits[b['query']].argmax(-1)
                    lab = y != 1                                          # PAD = 1
                    ok += (((pred == y) | ~lab).all(-1)).tolist()
                    sc = scores(logits, b['labels'], b['query'])
                    cell, n_cell = cell + sc['cell'], n_cell + sc['n_cell']
                    ce += float(query_loss(logits, b['labels'], b['query'])) * int(b['query'].sum())
            if k in ks:
                ok = np.array(ok)
                base = ok if base is None else base
                print(f"K={k:>3} ({k * meta['model']['blocks']:>4} blocks): exact {ok.sum():>3}/{n}  cell {cell / n_cell:.4f}  "
                      f"test CE {ce / n:.4f}  | vs K={ks[0]}: +{int((ok & ~base).sum())} newly solved, "
                      f"-{int((~ok & base).sum())} lost", flush=True)

if __name__ == '__main__':
    main()
