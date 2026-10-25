"""All-reduce a 16 MB tensor (the size of this model's gradients) across the GPUs and check the sum.

    torchrun --standalone --nproc_per_node=4 scripts/nccl_check.py      (exit 1 if wrong; a hang is caught by timeout)
"""
import os
import time

import torch
import torch.distributed as dist

torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
rank, ws = dist.get_rank(), dist.get_world_size()
x = torch.full((4_000_000,), float(rank + 1), device='cuda')
for _ in range(3):
    dist.all_reduce(x.clone())
torch.cuda.synchronize()
t0 = time.time()
y = x.clone()
dist.all_reduce(y)
torch.cuda.synchronize()
ok = bool((y == ws * (ws + 1) / 2).all())
if rank == 0:
    print(f'NCCL all_reduce of 16 MB over {ws} GPUs: {"ok" if ok else "WRONG"}, {(time.time() - t0) * 1e3:.1f} ms', flush=True)
dist.destroy_process_group()
raise SystemExit(0 if ok else 1)
