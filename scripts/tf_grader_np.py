#!/usr/bin/env python
"""NP check for the board grader (after scripts/tf_grader_gate.py, which trains and saves the graders): does the
node-perturbation update built from the grader's per-cell judgement point along the true descent direction of W~?

At a probe block (puzzles unsolved at seg 4), K independent draws of that block's v-noise xi (common noise elsewhere).
Per draw, right after the perturbed block (one-block delay, as the rule would use it):
    delta_t = sum_c q_t(c) [log p_t(after)(c) - log p_t(before)(c)]      q = the board grader at the block's input
    U_cell  = sum_t delta_t (xi_t / sigma^2) k~_t^T                      per-cell credit
    U_board = (sum_t delta_t) sum_s (xi_s / sigma^2) k~_s^T              one board scalar
and the same with the true per-cell improvement CE_t(before) - CE_t(after) (oracle credit, labels). The true direction
is -dL/dW~ at the block after the perturbed one (a leaf there), L = true CE `horizon` blocks later. Per puzzle,
cos(E_k U, -E_k g) and the directional SNR per draw E<u, -g^> / std<u, -g^>; medians over puzzles.
Usage: python scripts/tf_grader_np.py [--ckpt ...] [--graders runs/bg6/gate_s30k_graders.pt]
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.argv, _argv = sys.argv[:1], sys.argv[1:]          # the gate module parses its own (default) arguments
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='runs/tf6_hard/bands/step_30000.pt')
ap.add_argument('--graders', default='runs/bg6/gate_s30k_graders.pt')
ap.add_argument('--out', default='runs/bg6/np_s30k.json')
ap.add_argument('--K', type=int, default=48)
ap.add_argument('--npuz', type=int, default=64)
args = ap.parse_args(_argv)

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
from lt.data import load_data  # noqa: E402
from lt.model import LT, Carry, ModelConfig  # noqa: E402

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

gs = torch.load(os.path.join(REPO, args.graders), map_location='cpu', weights_only=False)
ga = gs['args']
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('gate_defs', os.path.join(REPO, 'scripts', 'tf_grader_gate.py'))
src = open(spec.origin).read()
src = src[src.index('class Grader'):src.index('graders = nn.ModuleDict')]
ns = dict(nn=torch.nn, torch=torch, F=F, T=T, V=V, args=argparse.Namespace(**ga))
exec(src, ns)
board = ns['Grader'](True).to(DEV)
board.load_state_dict({k[len('board.'):]: v for k, v in gs['graders'].items() if k.startswith('board.')})
board.eval()
for p_ in board.parameters():
    p_.requires_grad_(False)


class Hooks:
    def __init__(self):
        self.orig_in, self.orig_w, self.orig_b = m.tf_input, m.tf_write, m.bands_step
        m.tf_input, m.tf_write, m.bands_step = self.tf_input, self.tf_write, self.bands
        m.classifier.register_forward_hook(lambda mod, inp, out: setattr(self, 'final', inp[0]))
        self.reset()

    def reset(self, noise=None, leaf=None, leaf_at=None):
        self.noise, self.leaf, self.leaf_at = noise, leaf, leaf_at
        self.hs, self.wcount, self.final, self.elig = [], 0, None, []

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

    def bands(self, q, k, v, state, tf=None):
        out = self.orig_b(q, k, v, state, tf)
        with torch.no_grad():                         # the block's xi and k~ (as in bands_step)
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            ang = m.rope_angles().float()
            ang = ang[None] if ang.dim() == 3 else ang
            kr = m.rope(unit(k.float()), (ang.cos(), ang.sin()))
            xi = tf[4].float() * v.float().pow(2).mean(-1, keepdim=True).sqrt()
            self.elig.append((xi, kr))
        return out


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


_, _, te_x, te_y, _, _ = load_data(os.path.join(REPO, ga['data']))
allb = mk(te_x, te_y)
with torch.no_grad():
    carry = m.initial_carry(allb)
    for s in range(LOOPS):
        hk.reset(noise=common_noise(100 + s, len(te_x)))
        carry, logits = m(carry, allb)
    stuck = np.nonzero(~(logits.argmax(-1) == allb['labels']).all(-1).cpu().numpy())[0]
idx = stuck[:args.npuz]
print(f'[NP] {args.ckpt}: {len(stuck)} test puzzles unsolved at seg {LOOPS}; probing {len(idx)} x {args.K} draws',
      flush=True)


def measure(s0, horizon, K):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    with torch.no_grad():
        carry = m.initial_carry(b)
        for s in range(s0):
            hk.reset(noise=common_noise(300 + s, B0))
            carry, _ = m(carry, b)
    c = Carry(**{f.name: rep(getattr(carry, f.name), K) for f in dataclasses.fields(carry)})
    bb = rep(b, K)
    base = common_noise(900 + s0, B0, K)
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(noise=lambda i, shape: SIG * torch.randn(shape, device=DEV) if i == 0 else base(i, shape),
                 leaf=torch.zeros(B0 * K, H, D, D, device=DEV, requires_grad=True), leaf_at=1)
        with torch.enable_grad():
            m(c, bb)
            hend, h0, h1 = hk.final, hk.hs[0], (hk.hs[1] if horizon > 0 else hk.final)
            L = cell_ce(logp_of(hend), bb['labels']).mean(-1)
            g, = torch.autograd.grad(L.sum(), hk.leaf)
    finally:
        cfg.blocks_per_seg = nb0
    with torch.no_grad():
        xi, kr = hk.elig[0]                                              # block 0 (perturbed): [B, H, T, D]
        lp0, lp1 = logp_of(h0.detach()), logp_of(h1.detach())
        q0 = torch.softmax(board(lp0, bb['inputs']).double(), -1)
        d_grader = (q0 * (lp1 - lp0)).sum(-1)                            # [B, T] per-cell improvement under q
        lab = bb['labels']
        d_true = cell_ce(lp0, lab) - cell_ce(lp1, lab)                   # true per-cell improvement (oracle)
        Lr = L.detach().double().reshape(B0, K)
        adv = -(Lr - Lr.mean(1, keepdim=True)).reshape(-1)               # oracle: long-term outcome, mean baseline
        e = lambda w: torch.einsum('bht,bhtd,bhte->bhde', w.float()[:, None, :].expand(-1, H, -1), xi, kr) / SIG ** 2
        full = torch.einsum('bhtd,bhte->bhde', xi, kr) / SIG ** 2
        U = {'grader per-cell': e(d_grader), 'grader board': d_grader.sum(-1).float()[:, None, None, None] * full,
             'true per-cell (oracle)': e(d_true), 'true board (oracle)': d_true.sum(-1).float()[:, None, None, None] * full,
             'long-term true CE (oracle)': adv.float()[:, None, None, None] * full}
        g = g.double().reshape(B0, K, -1)
        gbar = g.mean(1)
        ghat = gbar / gbar.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        wn = c.stdp_w.double().reshape(B0, K, -1)[:, 0].norm(dim=-1)
        out = dict(L_end=Lr.mean().item(), corr_board_delta_vs_true_board_delta=torch.nanmedian(torch.stack([
            torch.corrcoef(torch.stack((d_grader.sum(-1).reshape(B0, K)[i], d_true.sum(-1).reshape(B0, K)[i])))[0, 1]
            for i in range(B0)])).item())
        for name, u in U.items():
            u = u.double().reshape(B0, K, -1)
            mu = u.mean(1)
            cos = (mu * -gbar).sum(-1) / (mu.norm(dim=-1) * gbar.norm(dim=-1)).clamp_min(1e-30)
            proj = -(u * ghat[:, None]).sum(-1)
            out[name] = dict(cos=cos.median().item(), cos_pos=(cos > 0).double().mean().item(),
                             dir_snr=(proj.mean(1) / proj.std(1)).median().item(),
                             u_over_W=(u.norm(dim=-1).mean(1) / wn).median().item())
    return out


res = dict(ckpt=args.ckpt, graders=args.graders, K=args.K, n=len(idx), probes={})
for s0, horizon in ((4, 3), (4, 11), (8, 3), (8, 11), (16, 11)):
    key = f'block{s0 * NB + 1}_h{horizon}'
    o = measure(s0, horizon, args.K)
    res['probes'][key] = o
    print(f'[NP] {key}: L_end {o["L_end"]:.4f}; corr(board delta under q, true board delta) '
          f'{o["corr_board_delta_vs_true_board_delta"]:+.3f}', flush=True)
    for name in ('grader per-cell', 'grader board', 'true per-cell (oracle)', 'true board (oracle)',
                 'long-term true CE (oracle)'):
        v = o[name]
        print(f'   {name:28s} cos(E U, -grad) {v["cos"]:+.3f} (pos {v["cos_pos"]:.2f})  directional SNR/draw '
              f'{v["dir_snr"]:+.3f}  |U|/|W~| {v["u_over_W"]:.3f}', flush=True)
json.dump(res, open(os.path.join(REPO, args.out), 'w'), indent=1)
print(f'[NP] done in {time.time() - t0:.0f}s', flush=True)
