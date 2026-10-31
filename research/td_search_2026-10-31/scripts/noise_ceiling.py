"""Exploration ceiling of the v noise (frozen checkpoint, no training). For each noise size sigma (std sigma x rms(v), fresh
each block, as tf_noise), M independent runs of every test puzzle from a fresh state for 64 blocks. Reported: mean exact of
one run; any-run oracle (solved if any of the first 4 / all M runs is exact at block 64, labels pick: the ceiling a
reward-driven rule could reach); label-free pick of the lowest-entropy run. Reference: noise-free plain run, and the
branch search's label oracle 2013 at 64 blocks.
Usage: python noise_ceiling.py CKPT OUT_JSON [M=16]
"""
import json, sys, time
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK, OUT = sys.argv[1], sys.argv[2]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SIGMAS, BLOCKS, CH = (0.1, 0.3, 1.0, 3.0), 64, 16384
torch.manual_seed(0)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data(cfg['data_npz'], cfg['test_size'])
cfg.update(batch_size=CH, seq_len=36, vocab_size=8, amp=True, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'], strict=False)
m.eval()
m.config.loops = 10 ** 6
nb = m.config.blocks_per_seg
tok0 = torch.tensor(X.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)
N = len(tok0)


@torch.no_grad()
def run(tok):
    """Exact at block 64 and the blank-cell entropy, per lane."""
    ex, en = [], []
    for i in range(0, len(tok), CH):
        t = tok[i:i + CH]
        b = dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))
        c = m.initial_carry(b)
        for s in range(BLOCKS // nb):
            c, lg = m(c, b)
        x = lg.float()
        sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
        P = sm / sm.sum(-1, keepdim=True)
        pd = P[..., 2:8].clamp_min(1e-9)
        e = -(pd * pd.log()).sum(-1)
        blank = t == 1
        en.append((e * blank).sum(-1) / blank.sum(-1))
        ex.append((lg.argmax(-1) == lab.repeat(len(tok) // N, 1)[i:i + CH]).all(-1))
    return torch.cat(ex), torch.cat(en)


t0 = time.time()
res = dict(ckpt=CK, M=M, blocks=BLOCKS)
m.config.tf_noise_eval = False
ex, _ = run(tok0)
res['plain'] = int(ex.sum())
print(f'[{CK}] plain (noise off) {res["plain"]}/{N} at {BLOCKS} blocks; branch-search label oracle 2013', flush=True)
m.config.tf_noise_eval = True
for sig in SIGMAS:
    m.config.tf_noise = sig
    ex, en = run(tok0.repeat(M, 1))
    ex, en = ex.view(M, N), en.view(M, N)
    pick = en.argmin(0)
    r = dict(mean_one_run=float(ex.float().sum(1).mean()), oracle_4=int(ex[:4].any(0).sum()), oracle_M=int(ex.any(0).sum()),
             entropy_pick_4=int(ex[:4][en[:4].argmin(0), torch.arange(N)].sum()), entropy_pick_M=int(ex[pick, torch.arange(N)].sum()))
    res[f'sigma_{sig}'] = r
    print(f'   sigma {sig}: one run {r["mean_one_run"]:.0f} | any of 4 {r["oracle_4"]} | any of {M} {r["oracle_M"]} | '
          f'entropy pick of 4 {r["entropy_pick_4"]}, of {M} {r["entropy_pick_M"]} ({time.time() - t0:.0f}s)', flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
