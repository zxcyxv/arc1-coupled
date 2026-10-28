"""Readout at every depth of a <= 10 harness checkpoint (arc.small): one segment from a fresh carry with the number of
blocks set to B (the block weights are shared, so any B runs the same recursion and reads out after block B). Shows how
the answer forms before the trained depth and what happens right after it.

    python scripts/analysis/small_depth_profile.py checkpoints/small/b30g10_rope1d_outffn_wdmat_step200000_ema.pt 20 25 28 30 31 32 35 40 60
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arc.model import Coupled, ModelConfig  # noqa: E402
from arc.small import Episodes, query_loss, scores  # noqa: E402


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    depths = [int(x) for x in sys.argv[2:]]
    meta = ck['meta']
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False                     # as arc.small
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    cfg = ModelConfig(**meta['model'])
    model = Coupled(cfg).to(device)
    missing = model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'], strict=False)
    assert not missing.unexpected_keys, missing
    model.eval()
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    model.layer_fn = torch.compile(model.layer, dynamic=False)
    data = Episodes(meta['data'], N=meta['model']['canvas'] - 1, D=meta['D'])
    batches = [{k: v.to(device) for k, v in b.items()} for b in data.eval_batches(meta['eval_groups'])]
    n = sum(int(b['query'].sum()) for b in batches)
    print(f"{sys.argv[1]} (step {ck['step']}, trained depth {cfg.blocks} blocks), {n} test pairs", flush=True)
    base = None
    with torch.no_grad():
        for B in depths:
            model.cfg = dataclasses.replace(cfg, blocks=B, grad_blocks=min(cfg.grad_blocks, B))
            ok, cell, n_cell, ce = [], 0, 0, 0.0
            for b in batches:
                logits, _ = model.segment(None, b)
                y = b['labels'][b['query'], -logits.shape[1]:]
                pred = logits[b['query']].argmax(-1)
                lab = y != 1                                              # PAD = 1
                ok += (((pred == y) | ~lab).all(-1)).tolist()
                sc = scores(logits, b['labels'], b['query'])
                cell, n_cell = cell + sc['cell'], n_cell + sc['n_cell']
                ce += float(query_loss(logits, b['labels'], b['query'])) * int(b['query'].sum())
            ok = np.array(ok)
            if B == cfg.blocks:
                base = ok
            extra = f" | vs trained depth: +{int((ok & ~base).sum())} / -{int((~ok & base).sum())}" if base is not None else ''
            print(f"B={B:>3}: exact {ok.sum():>3}/{n}  cell {cell / n_cell:.4f}  test CE {ce / n:.4f}{extra}", flush=True)


if __name__ == '__main__':
    main()
