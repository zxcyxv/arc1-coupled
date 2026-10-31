"""Decisive binary comparison: three-factor consolidation of rule-free branch search into a persistent fast memory, trained
end to end, vs the same trainer without it. 6x6-hard, the tiny bands model of configs/tf/sudoku6_hard_tiny.json
(puzzle embedding off), noise 0.1, from scratch, one seed.

Episode per training step (B puzzles in lockstep):
  segments 1-4 (blocks 1-16, loss on each segment) ->
  [arm 'consol' only, no grad] branch: the least confident blank cell x top-K digits as hypothetical input clues, each
      re-solved from a fresh state for R blocks; third factor delta_k = -(entropy_k - mean_k entropy) / max_k |.| of the
      settled boards (label-free); A = sum_k delta_k (W_k - mean_k W_k) ->
  segments 5-6 (blocks 17-24, loss on each), the root continuing with its own input; arm 'consol' reads W + g_h A at every
      block (g per head, learned from 0), arm 'plain' reads W.
Evaluation (EMA weights, 2048 test puzzles): plain recursion at 24 and at the matched 16 + K R + 8 blocks; consolidation
inference; explicit selection of the most confident candidate (for reference). Labels only for the loss and scoring.
Usage: python consol_train.py ARM OUT [steps=8000]
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data, SudokuTrainDataset
from lt.train import AdamATan2, EMA, log_stablemax

ARM, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
assert ARM in ('plain', 'consol')
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda'
B, K, R = 128, 3, 16
LR, WARM, WD = 2e-4, 500, 1.0
torch.manual_seed(0)
np.random.seed(0)
cfg = json.load(open('/workspace/arc1-coupled/configs/tf/sudoku6_hard_tiny.json'))
mc = dict(hidden_size=cfg['hidden_size'], num_heads=cfg['num_heads'], puzzle_emb_ndim=0, bands=cfg['bands'],
          band_periods=cfg['band_periods'], band_coef_std=cfg['band_coef_std'], tag_bound=cfg['tag_bound'],
          outer_rho=cfg['outer_rho'], outer_eta=cfg['outer_eta'], outer_eta_learn=cfg['outer_eta_learn'],
          outer_eta_cap=cfg['outer_eta_cap'], blocks_per_seg=4, loops=10 ** 6, act_ckpt=False, grid=6, seq_len=36,
          vocab_size=8, batch_size=B * K, tf_noise=0.1, tf_noise_eval=False, amp=True, kernel='torch')
m = LT(ModelConfig.from_dict(mc)).to(DEV)
H = m.H
m.tf_A_gain = nn.Parameter(torch.zeros(H, device=DEV))
GAIN_SCALE = 10.0      # gain = 10 x parameter: Adam moves a parameter about lr per step (the 10-31 scale lesson)
APERS = {'A': None}
orig_w = m.tf_write
def tf_write_A(w, s_, tf, dtype, e=None):
    w2, mm, rest = orig_w(w, s_, tf, dtype, e)
    if APERS['A'] is not None:
        mm = mm + (GAIN_SCALE * m.tf_A_gain.to(mm.dtype).view(1, -1, 1, 1) * APERS['A'].to(mm.dtype))
    return w2, mm, rest
m.tf_write = tf_write_A

named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
decay = lambda n, p: p.ndim >= 2 and 'rope_theta' not in n and 'band_' not in n
opt = AdamATan2([{'params': [p for n, p in named if not decay(n, p)], 'weight_decay': 0.0},
                 {'params': [p for n, p in named if decay(n, p)], 'weight_decay': WD}], lr=0, betas=(0.9, 0.95))
ema = EMA(m, 0.999)
tr_x, tr_y, te_x, te_y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
ds = SudokuTrainDataset(tr_x, tr_y, seed=0, num_aug=1000, global_batch_size=B, rank=0, world_size=1,
                        epochs_per_iter=10 ** 6, start_iter=0, total_iters=1)
it = iter(ds)


def stab_probs(logits):
    x = logits.float()
    sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return sm / sm.sum(-1, keepdim=True)


def ce(logits, labels):
    return -log_stablemax(logits.double(), -1).gather(-1, labels.long()[..., None]).squeeze(-1).mean()   # FP64 as lt.train


def entropy(P, tok):
    pd = P[..., 2:8].clamp_min(1e-9)
    e = -(pd * pd.log()).sum(-1)
    blank = tok == 1
    return (e * blank).sum(-1) / blank.sum(-1).clamp_min(1)


def batch_of(tok):
    return dict(inputs=tok, labels=torch.full_like(tok, 2), puzzle_identifiers=torch.zeros(len(tok), dtype=torch.int32, device=DEV))


def branches(tok, P, carry_root):
    """No grad: K hypothesis branches re-solved R blocks; returns A (third-factor-weighted memory deviation), the branch
    probabilities and entropies."""
    n = len(tok)
    pd = P[..., 2:8]
    conf = pd.max(-1).values.masked_fill(tok != 1, 2.0)
    cell = conf.argmin(-1)
    top = pd[torch.arange(n), cell].topk(K, -1).indices
    br = tok[:, None, :].repeat(1, K, 1).clone()
    br[torch.arange(n)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    bb = batch_of(br.view(-1, 36))
    c = m.initial_carry(bb)
    for s in range(R // 4):
        c, lg = m(c, bb)
    Pb = stab_probs(lg).view(n, K, 36, 8)
    ent = torch.stack([entropy(Pb[:, k], tok) for k in range(K)], 1)
    d = -(ent - ent.mean(1, keepdim=True))
    d = d / d.abs().amax(1, keepdim=True).clamp_min(1e-6)
    Wk = c.stdp_w.float().view(n, K, *carry_root.stdp_w.shape[1:])
    A = (d.view(n, K, 1, 1, 1) * (Wk - Wk.mean(1, keepdim=True))).sum(1)
    return A, Pb, ent


log = open(f'{OUT}/train.log', 'a')
t0 = time.time()
for step in range(1, STEPS + 1):
    lr = LR * min(1.0, step / WARM)
    for g in opt.param_groups:
        g['lr'] = lr
    _, bt = next(it)
    tok = bt['inputs'].to(DEV)
    lab = bt['labels'].to(DEV)
    b = dict(inputs=tok, labels=lab, puzzle_identifiers=torch.zeros(len(tok), dtype=torch.int32, device=DEV))
    m.train()
    APERS['A'] = None
    carry = m.initial_carry(b)
    losses = []
    for s in range(4):
        carry, logits = m(carry, b)
        losses.append(ce(logits, lab))
    if ARM == 'consol':
        with torch.no_grad():
            A, _, _ = branches(tok, stab_probs(logits.detach()), carry)
        APERS['A'] = A
    for s in range(2):
        carry, logits = m(carry, b)
        losses.append(ce(logits, lab))
    APERS['A'] = None
    loss = sum(losses) / len(losses)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    m.block.project_eta_(m.config) if hasattr(m.block, 'project_eta_') else None
    ema.update(m)
    if step % 250 == 0:
        ex = (logits.argmax(-1) == lab).all(-1).float().mean().item()
        msg = (f'step {step} loss {loss.item():.4f} seg1 {losses[0].item():.3f} seg4 {losses[3].item():.3f} seg6 {losses[5].item():.3f} '
               f'train exact@24 {ex:.3f} gain {(GAIN_SCALE * m.tf_A_gain.detach().cpu()).numpy().round(3).tolist()} lr {lr:.1e} {time.time() - t0:.0f}s')
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()


@torch.no_grad()
def evaluate(noise):
    m.eval()
    m.config.tf_noise_eval = noise
    tok = torch.tensor(te_x.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
    lab = torch.tensor(te_y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)
    N = len(tok)
    res = {}
    exact = lambda P: int((P.argmax(-1) == lab).all(-1).sum())
    def plain(blocks):
        out = []
        for i in range(0, N, B * K):
            bb = batch_of(tok[i:i + B * K]); c = m.initial_carry(bb)
            for s in range(blocks // 4):
                c, lg = m(c, bb)
            out.append(stab_probs(lg))
        return torch.cat(out)
    res['plain@24'] = exact(plain(24))
    res[f'plain@{16 + K * R + 8}'] = exact(plain(16 + K * R + 8))
    cons, sel, cons16 = [], [], []
    for i in range(0, N, B):
        t = tok[i:i + B]; bb = batch_of(t); c = m.initial_carry(bb)
        for s in range(4):
            c, lg = m(c, bb)
        P0 = stab_probs(lg)
        A, Pb, ent = branches(t, P0, c)
        e0 = entropy(P0, t)
        cand = torch.cat([P0[:, None], Pb], 1)
        ce_ = torch.cat([e0[:, None], ent], 1)
        sel.append(cand[torch.arange(len(t)), ce_.argmin(1)])
        APERS['A'] = A
        c2 = c
        for s in range(2):
            c2, lg2 = m(c2, bb)
        cons.append(stab_probs(lg2))
        for s in range(2):
            c2, lg3 = m(c2, bb)
        cons16.append(stab_probs(lg3))
        APERS['A'] = None
    res['consolidation@16+48+8'] = exact(torch.cat(cons))
    res['consolidation@16+48+16'] = exact(torch.cat(cons16))
    res['explicit selection@16+48'] = exact(torch.cat(sel))
    m.train()
    return res


with ema.swap(m):
    out = {f'noise_{n}': evaluate(n) for n in (False, True)}
out['gain'] = (GAIN_SCALE * m.tf_A_gain.detach().cpu()).tolist()
print(json.dumps(out), flush=True)
json.dump(out, open(f'{OUT}/eval.json', 'w'), indent=1)
with ema.swap(m):
    torch.save({'model': m.state_dict(), 'mc': mc, 'arm': ARM, 'steps': STEPS}, f'{OUT}/final_ema.pt')
