#!/usr/bin/env python
"""Diagnostic ceiling for the 'inter-cell consistency' premise (no part of any model; a rule check used only to
measure what a perfect consistency-learning grader could deliver), on the same frozen checkpoint, puzzles and probes as
scripts/tf_grader_gate.py.

1. Conflicts in the argmax boards of the puzzles unsolved at seg 4: boards with at least one conflict (two peers with
   the same argmax digit), conflicting pairs per board, the share of wrong and of right cells that sit in a conflict.
2. Soft conflict penalty pen_t = sum_{s in peers(t)} <p_t, p_s> (digit classes, stablemax p), board mean:
     corr : median within-puzzle corr(penalty_k, true CE_k) over K noise draws (puzzles open at the probe start)
     line : descend the penalty wrt W~ (leaf at the probe's block 1); true dCE / dexact / dacc vs step size,
            next to the oracle (descend the true CE).
Usage: python scripts/tf_grader_ceiling.py [--ckpt runs/tf6_hard/bands/step_30000.pt]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lt.data import load_data  # noqa: E402
from lt.model import LT, Carry, ModelConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='runs/tf6_hard/bands/step_30000.pt')
ap.add_argument('--data', default='data/sudoku6_hard_lt.npz')
ap.add_argument('--out', default='runs/bg6/ceiling_s30k.json')
ap.add_argument('--K', type=int, default=32)
ap.add_argument('--npuz', type=int, default=64)
args = ap.parse_args()
DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
ck = torch.load(os.path.join(REPO, args.ckpt), map_location='cpu', weights_only=False)
C = dict(ck['cfg'])
C.update(batch_size=4096, seq_len=C['grid'] ** 2 * C['num_grids'], vocab_size=C['grid'] + 2, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(C)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
for p_ in m.parameters():
    p_.requires_grad_(False)
cfg = m.config
LOOPS, NB = cfg.loops, cfg.blocks_per_seg
cfg.loops = 10 ** 6
SIG = cfg.tf_noise
H, D, T, V = m.H, m.dh, cfg.seq_len, cfg.vocab_size
N = cfg.grid
BR, BC = (2, 3) if N == 6 else (int(N ** 0.5),) * 2
peer = torch.zeros(T, T, dtype=torch.float64)
for t in range(T):
    for s in range(T):
        r1, c1, r2, c2 = t // N, t % N, s // N, s % N
        if s != t and (r1 == r2 or c1 == c2 or (r1 // BR == r2 // BR and c1 // BC == c2 // BC)):
            peer[t, s] = 1
peer = peer.to(DEV)


class Hooks:
    def __init__(self):
        self.orig_in, self.orig_w = m.tf_input, m.tf_write
        m.tf_input, m.tf_write = self.tf_input, self.tf_write
        m.classifier.register_forward_hook(lambda mod, inp, out: setattr(self, 'final', inp[0]))
        self.reset()

    def reset(self, noise=None, leaf=None, leaf_at=None):
        self.noise, self.leaf, self.leaf_at = noise, leaf, leaf_at
        self.hs, self.wcount, self.final = [], 0, None

    def tf_input(self, tf, h, i, store):
        elig, delta, n, w0, noise = self.orig_in(tf, h, i, store)
        self.hs.append(h)
        if self.noise is not None:
            noise = self.noise(i, (h.shape[0], H, h.shape[1], D))
        return (elig, delta, n, w0, noise)

    def tf_write(self, w, s, tf, dtype, e=None):
        idx = self.wcount
        self.wcount += 1
        if self.leaf_at == idx:
            w = w + self.leaf
        return self.orig_w(w, s, tf, dtype, e)


hk = Hooks()


def common_noise(seed, B, K=1):
    def f(i, shape):
        g = torch.Generator(device=DEV).manual_seed(seed * 1000 + i)
        return SIG * torch.randn((B,) + shape[1:], generator=g, device=DEV).repeat_interleave(K, 0)
    return f


def rep(x, K):
    if torch.is_tensor(x):
        return x.repeat_interleave(K, 0)
    if isinstance(x, dict):
        return {k: v.repeat_interleave(K, 0) for k, v in x.items()}
    return x


def mk(x, y):
    n = len(x)
    return dict(inputs=torch.tensor(x.reshape(n, -1).astype(np.int64) + 1, device=DEV),
                labels=torch.tensor(y.reshape(n, -1).astype(np.int64) + 1, device=DEV),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32, device=DEV))


def logp_of(h):
    x = m.classifier(h).double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(-1, keepdim=True))


def cell_ce(logp, labels):
    return -torch.gather(logp, -1, labels.expand(*logp.shape[:-1]).unsqueeze(-1)).squeeze(-1)


def penalty(logp):
    """Per-cell soft conflict pen_t = sum_s peer(t, s) <p_t, p_s>, p renormalised over the digit classes [..., T]
    (unnormalised, moving mass to the pad / blank classes would lower it)."""
    p = logp.exp()[..., 2:]
    p = p / p.sum(-1, keepdim=True)
    return (p * torch.einsum('ts,...sc->...tc', peer, p)).sum(-1)


@torch.no_grad()
def rollout(b, segs, noise_for_seg):
    carry = m.initial_carry(b)
    states = []
    for s in range(segs):
        hk.reset(noise=noise_for_seg(s))
        carry, _ = m(carry, b)
        states += hk.hs[1:] + [hk.final]
    return torch.stack(states), carry


_, _, te_x, te_y, _, _ = load_data(os.path.join(REPO, args.data))
allb = mk(te_x, te_y)
with torch.no_grad():
    carry = m.initial_carry(allb)
    for s in range(LOOPS):
        hk.reset(noise=common_noise(100 + s, len(te_x)))
        carry, logits = m(carry, allb)
    stuck = np.nonzero(~(logits.argmax(-1) == allb['labels']).all(-1).cpu().numpy())[0]
res = dict(ckpt=args.ckpt, unsolved_seg4=int(len(stuck)), conflicts={}, corr={}, line={})
print(f'[CEIL] {args.ckpt}: {len(stuck)} test puzzles unsolved at seg {LOOPS}', flush=True)

# 1. conflicts in the argmax boards
with torch.no_grad():
    b = mk(te_x[stuck], te_y[stuck])
    S, _ = rollout(b, 32, lambda s: common_noise(200 + s, len(stuck)))
    for n in (16, 32, 64, 128):
        a = logp_of(S[n - 1]).argmax(-1)                                     # [B, T]
        same = (a[:, :, None] == a[:, None, :]).double() * peer               # conflicting peer pairs (both orders)
        inconf = same.sum(-1) > 0
        wrong = a != b['labels']
        open_ = wrong.any(-1)
        r = dict(open_boards=int(open_.sum()),
                 open_with_conflict=float((inconf.any(-1) & open_).sum() / open_.sum().clamp_min(1)),
                 pairs_per_open_board=float((same.sum((1, 2)) / 2)[open_].mean()),
                 wrong_cells_in_conflict=float((inconf & wrong).sum() / wrong.sum().clamp_min(1)),
                 right_cells_in_conflict=float((inconf & ~wrong & open_[:, None]).sum() /
                                               (~wrong & open_[:, None]).sum().clamp_min(1)),
                 wrong_cells_per_open_board=float(wrong.sum(-1)[open_].double().mean()),
                 clue_mismatch=int(((b['inputs'] > 1) & (a != b['inputs'])).sum()))
        res['conflicts'][f'block{n}'] = r
        print(f'[CONFLICT] block {n}: {r["open_boards"]} unsolved boards, {100 * r["open_with_conflict"]:.1f}% with a '
              f'conflict, {r["pairs_per_open_board"]:.2f} conflicting pairs and {r["wrong_cells_per_open_board"]:.2f} '
              f'wrong cells per board; wrong cells in a conflict {100 * r["wrong_cells_in_conflict"]:.1f}%, right cells '
              f'{100 * r["right_cells_in_conflict"]:.1f}%; clue mismatches {r["clue_mismatch"]}', flush=True)
    del S

idx = stuck[:args.npuz]


def wcorr(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))


@torch.no_grad()
def corr_probe(s0, K, horizon):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    _, carry = rollout(b, s0, lambda s: common_noise(300 + s, B0))
    c, bb = Carry(**{f.name: rep(getattr(carry, f.name), K) for f in dataclasses.fields(carry)}), rep(b, K)
    base = common_noise(900 + s0, B0, K)
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(noise=lambda i, shape: SIG * torch.randn(shape, device=DEV) if i == 0 else base(i, shape))
        m(c, bb)
        states = hk.hs + [hk.final]
    finally:
        cfg.blocks_per_seg = nb0
    lab = bb['labels']
    open0 = ~(logp_of(states[0]).argmax(-1) == lab).all(-1).reshape(B0, K)[:, 0]
    out = dict(open_at_start=int(open0.sum()))
    R = lambda x: x.reshape(B0, K)
    for j in sorted({1, min(4, horizon + 1), horizon + 1}):
        lp = logp_of(states[j])
        cw = wcorr(R(penalty(lp).mean(-1)), R(cell_ce(lp, lab).mean(-1)))
        out[f'penalty@{j}'], out[f'penalty@{j}_open'] = cw.nanmedian().item(), cw[open0].nanmedian().item()
    return out


def line_probe(s0, horizon, fracs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    _, carry = rollout(b, s0, lambda s: common_noise(300 + s, B0))
    lab = b['labels']

    def fwd(z):
        nb0 = cfg.blocks_per_seg
        cfg.blocks_per_seg = horizon + 1
        try:
            hk.reset(noise=common_noise(777, B0), leaf=z, leaf_at=1)
            m(carry, b)
            return logp_of(hk.final), hk.hs[0]
        finally:
            cfg.blocks_per_seg = nb0
    with torch.no_grad():
        lp0, hstart = fwd(torch.zeros(B0, H, D, D, device=DEV))
        base_ce, base_ok = cell_ce(lp0, lab).mean(-1), lp0.argmax(-1) == lab
        open0 = ~(logp_of(hstart).argmax(-1) == lab).all(-1)
    objectives = {'ce (oracle)': lambda lp: cell_ce(lp, lab).mean(-1),
                  'conflict penalty (rule; ceiling)': lambda lp: penalty(lp).mean(-1)}
    wnorm = carry.stdp_w.double().reshape(B0, -1).norm(dim=-1)
    out = dict(base_ce=base_ce.mean().item(), base_exact=int(base_ok.all(-1).sum()), n=B0,
               open_at_start=int(open0.sum()), objectives={})
    grads = {}
    for name, f in objectives.items():
        z = torch.zeros(B0, H, D, D, device=DEV, requires_grad=True)
        with torch.enable_grad():
            lp, _ = fwd(z)
            grads[name], = torch.autograd.grad(f(lp).sum(), z)
        grads[name] = grads[name].double().reshape(B0, -1)
    for name, g in grads.items():
        cos = (g * grads['ce (oracle)']).sum(-1) / (g.norm(dim=-1) * grads['ce (oracle)'].norm(dim=-1)).clamp_min(1e-30)
        dirn = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        rows = []
        with torch.no_grad():
            for fr in fracs:
                lp, _ = fwd((fr * wnorm[:, None] * dirn).reshape(B0, H, D, D).float())
                ok = lp.argmax(-1) == lab
                dce = cell_ce(lp, lab).mean(-1) - base_ce
                rows.append(dict(frac=fr, dce=dce.mean().item(), dce_open=dce[open0].mean().item(),
                                 dexact=int(ok.all(-1).sum() - base_ok.all(-1).sum()),
                                 dacc=(ok.double().mean(-1) - base_ok.double().mean(-1)).mean().item()))
        out['objectives'][name] = dict(cos_median=cos.median().item(), cos_open_median=cos[open0].median().item(),
                                       line=rows)
    return out


for s0 in (4, 8, 16):
    key = f'block{s0 * NB + 1}'
    for horizon in (3, 11):
        o = corr_probe(s0, args.K, horizon)
        res['corr'][f'{key}_h{horizon}'] = o
        print(f'[CORR] {key}_h{horizon}: conflict penalty vs true CE, median within-puzzle corr (open at start '
              f'{o["open_at_start"]}): ' + '  '.join(f'@{k.split("@")[1]} {v:+.3f} [{o[k + "_open"]:+.3f}]'
                                                       for k, v in o.items() if k.startswith('penalty@')
                                                       and not k.endswith('_open')), flush=True)
    o = line_probe(s0, 11)
    res['line'][f'{key}_h11'] = o
    print(f'[LINE] {key}_h11: base CE {o["base_ce"]:.4f}, exact {o["base_exact"]}/{o["n"]} (open at start '
          f'{o["open_at_start"]}): true dCE (open puzzles) / dexact / dacc', flush=True)
    for name, v in o['objectives'].items():
        rows = '  '.join(f'{r["frac"]:g}: {r["dce"]:+.3f} ({r["dce_open"]:+.3f})/{r["dexact"]:+d}/{r["dacc"]:+.3f}'
                         for r in v['line'])
        print(f'   {name:34s} cos {v["cos_median"]:+.3f} (open {v["cos_open_median"]:+.3f}) | {rows}', flush=True)
json.dump(res, open(os.path.join(REPO, args.out), 'w'), indent=1)
print(f'[CEIL] done in {time.time() - t0:.0f}s', flush=True)
