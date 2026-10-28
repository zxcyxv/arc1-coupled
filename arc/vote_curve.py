"""Vote evaluation in chunks of versions with the running scores after each chunk: versions 0..C-1 are evaluated, the
votes so far are scored and appended to <out>.jsonl, then C..2C-1, and so on to all versions. The last line equals
arc.evaluate --versions all (same records, same ranking).

    torchrun --standalone --nproc_per_node=4 -m arc.vote_curve --ckpt checkpoints/run_X/step_N.pt --chunk 256
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .data import Store
from .evaluate import load_weights, run, score_records, submission, version_set


def main():
    from .train import build_model, init_distributed, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--lanes', type=int, default=96)
    ap.add_argument('--segments', type=int, default=None, help='default: the run config loops (16)')
    ap.add_argument('--weights', default='ema', choices=('ema', 'raw'))
    ap.add_argument('--out', default=None, help='default: <ckpt dir>/vote_curve_<step>.jsonl')
    args = ap.parse_args()
    rank, ws, device = init_distributed()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(ck['config'])
    store = Store(cfg['data'])
    model = build_model(cfg, device)
    load_weights(model, ck, args.weights)
    if cfg['compile'] and device.type == 'cuda':
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        model.layer_fn = torch.compile(model.layer, dynamic=False)
    segments = args.segments or cfg['loops']
    out = Path(args.out or Path(args.ckpt).parent / f"vote_curve_{ck['step']}.jsonl")
    allv = version_set(store, 'all')
    recs, grids, t0 = [], {}, time.time()
    for a in range(0, len(allv), args.chunk):
        versions = allv[:a + args.chunk]
        res = run(model, store, allv[a:a + args.chunk], args.lanes, segments, rank, ws, device)
        if rank:
            continue
        recs += res[1]
        grids.update(res[2])
        m = score_records(store, recs, versions)
        m.update(step=ck['step'], weights=args.weights, segments=segments, seconds=time.time() - t0)
        with open(out, 'a') as f:
            f.write(json.dumps(m) + '\n')
        print(f"[VOTE] {len(versions)} versions: pass@1 {m['pass@1']:.4f} pass@2 {m['pass@2']:.4f} "
              f"pass@10 {m['pass@10']:.4f} view {m['view_exact']:.4f} ({m['seconds']:.0f}s)", flush=True)
    if rank == 0:
        out.with_suffix('.submission.json').write_text(json.dumps(submission(store, recs, grids)))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
