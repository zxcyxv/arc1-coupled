#!/usr/bin/env python
"""Gate for a board-level learned grader as the third factor (6x6 hard Sudoku, frozen bands checkpoint, no training of
the model itself).

Question: does a learned grader that sees the whole board of predictions carry outcome information at inference, on
puzzles still unsolved at the training depth, that the per-cell readout lacks?

1. Graders trained on the frozen model's own block outputs (augmented training puzzles, blocks 1..16 = the training
   depth, v-noise as in training); labels are the targets, as for C^ now:
     board : q_t = softmax(G(board)_t); G = pre-norm attention layers over the 36 cells; per-cell input
             [p_t, log p_t] (p = the classifier's stablemax distribution), the clue token and a learned position
     cell  : the same network without attention (each cell sees only itself)                      [control]
     chat  : Linear-SiLU-Linear on the cell's hidden h_t regressing its CE (the current C^ head)    [control]
   Judgement of a board, per cell: C^_t = -sum_c q_t(c) log p_t(c) (board / cell; '_fix' holds q at the probe's
   start state, i.e. the grader's view before the perturbation), chat's output, the entropy of p (self-confirming).
2. Test puzzles unsolved at seg 4 (block 16), probes starting at blocks 17 / 33 / 65:
     gap  : CE(q, y) against CE(p, y), cell and exact accuracy of argmax q against argmax p.
     corr : K independent draws of the probe block's v-noise, common noise elsewhere; median over puzzles of the
            within-puzzle correlation between the judgement and the true CE at the same state (1, 4, 12 blocks after
            the perturbation) and of the judgement right after the perturbed block with the true CE at the end.
     line : descend each judgement wrt W~ (a leaf added at the probe's block 1, as in the earlier diag3) and measure the
            true CE / exact / cell accuracy after the horizon against step size; the oracle descends the true CE.
Usage: python scripts/tf_grader_gate.py [--ckpt runs/tf6_hard/bands/step_30000.pt] [--out runs/bg6/gate.json]
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lt.data import _apply_aug, _draw_aug_params, load_data  # noqa: E402
from lt.model import LT, Carry, ModelConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='runs/tf6_hard/bands/step_30000.pt')
ap.add_argument('--data', default='data/sudoku6_hard_lt.npz')
ap.add_argument('--out', default='runs/bg6/gate.json')
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--width', type=int, default=64)
ap.add_argument('--layers', type=int, default=2)
ap.add_argument('--K', type=int, default=32)
ap.add_argument('--npuz', type=int, default=64)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(args.seed)
t0 = time.time()

ck = torch.load(os.path.join(REPO, args.ckpt), map_location='cpu', weights_only=False)
C = dict(ck['cfg'])
C.update(batch_size=4096, seq_len=C['grid'] ** 2 * C['num_grids'], vocab_size=C['grid'] + 2, amp=False,
         act_ckpt=False)
m = LT(ModelConfig.from_dict(C)).to(DEV)
m.load_state_dict(ck['model'])                       # EMA weights (lt.train saves them as 'model')
m.eval()
for p_ in m.parameters():
    p_.requires_grad_(False)
cfg = m.config
LOOPS, NB = cfg.loops, cfg.blocks_per_seg
cfg.loops = 10 ** 6                                  # no lane replacement
SIG = cfg.tf_noise
H, D, T, V = m.H, m.dh, cfg.seq_len, cfg.vocab_size
assert SIG > 0 and m.prefix_len == 0


class Hooks:
    """Per-block control of the frozen model: v-noise per block, a leaf added to W~ at one block, the block inputs."""
    def __init__(self):
        self.orig_in, self.orig_w = m.tf_input, m.tf_write
        m.tf_input, m.tf_write = self.tf_input, self.tf_write
        m.classifier.register_forward_hook(self.cap)
        self.reset()

    def reset(self, noise=None, leaf=None, leaf_at=None):
        self.noise, self.leaf, self.leaf_at = noise, leaf, leaf_at
        self.hs, self.wcount, self.final = [], 0, None

    def cap(self, mod, inp, out):
        self.final = inp[0]

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


def indep_noise(i, shape):
    return SIG * torch.randn(shape, device=DEV)


def common_noise(seed, B, K=1):
    """sigma * eps per (seed, block) for B base lanes, the same for their K replicas."""
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


def rep_carry(c, K):
    return Carry(**{f.name: rep(getattr(c, f.name), K) for f in dataclasses.fields(c)})


def mk(x, y):
    n = len(x)
    return dict(inputs=torch.tensor(x.reshape(n, -1).astype(np.int64) + 1, device=DEV),
                labels=torch.tensor(y.reshape(n, -1).astype(np.int64) + 1, device=DEV),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32, device=DEV))


tr_x, tr_y, te_x, te_y, _, _ = load_data(os.path.join(REPO, args.data))
rng = np.random.default_rng(args.seed)


def aug_batch(n):
    xs, ys = [], []
    for i in rng.integers(0, len(tr_x), n):
        prm = _draw_aug_params(rng, tr_x.shape[-1])
        xs.append(_apply_aug(tr_x[i], *prm))
        ys.append(_apply_aug(tr_y[i], *prm))
    return mk(np.stack(xs), np.stack(ys))


def logp_of(h):
    """log stablemax of the classifier on states h [..., T, d] (FP64)."""
    x = m.classifier(h).double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return torch.log(s / s.sum(-1, keepdim=True))


def cell_ce(logp, labels):
    return -torch.gather(logp, -1, labels.expand(*logp.shape[:-1]).unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def rollout(b, segs, noise_for_seg, carry=None):
    """Block outputs [segs * NB, B, T, d] (block 1 first) and the last carry."""
    carry = m.initial_carry(b) if carry is None else carry
    states = []
    for s in range(segs):
        hk.reset(noise=noise_for_seg(s))
        carry, _ = m(carry, b)
        states += hk.hs[1:] + [hk.final]
    return torch.stack(states), carry


# ---------------------------------------------------------------------------------------------------------- graders
class Grader(nn.Module):
    def __init__(self, attn, d=args.width, layers=args.layers, heads=4):
        super().__init__()
        self.attn = attn
        self.inp = nn.Linear(2 * V, d)
        self.clue = nn.Embedding(V, d)
        self.pos = nn.Parameter(0.1 * torch.randn(T, d))
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            n1=nn.LayerNorm(d), att=nn.MultiheadAttention(d, heads, batch_first=True), n2=nn.LayerNorm(d),
            mlp=nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)))) for _ in range(layers))
        self.no = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)

    def forward(self, logp, inputs):
        """q logits [..., T, V] from log p [..., T, V] and the clue tokens [..., T]."""
        shp = logp.shape
        lp = logp.reshape(-1, T, V).float()
        x = self.inp(torch.cat([lp.exp(), lp.clamp_min(-20)], -1)) + self.clue(inputs.reshape(-1, T)) + self.pos
        for b in self.blocks:
            if self.attn:
                y = b['n1'](x)
                x = x + b['att'](y, y, y, need_weights=False)[0]
            x = x + b['mlp'](b['n2'](x))
        return self.out(self.no(x)).reshape(shp)


graders = nn.ModuleDict(dict(board=Grader(True), cell=Grader(False))).to(DEV)
chat = nn.Sequential(nn.Linear(m.d, 32), nn.SiLU(), nn.Linear(32, 1)).to(DEV)
params = list(graders.parameters()) + list(chat.parameters())
print(f'[GATE] ckpt {args.ckpt} step {ck["step"]}; graders board {sum(p.numel() for p in graders["board"].parameters())} '
      f'cell {sum(p.numel() for p in graders["cell"].parameters())} chat {sum(p.numel() for p in chat.parameters())} '
      f'params; model {sum(p.numel() for p in m.parameters())}', flush=True)
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)


def judge(name, logp, inputs, h=None, qfix=None):
    """Per-cell judgement (predicted CE) [..., T]."""
    if name == 'chat':
        return F.softplus(chat(h.float()).squeeze(-1)).double()
    if name == 'ent':
        return -(logp.exp() * logp).sum(-1)
    g = graders[name.split('_')[0]]
    q = torch.softmax(g(logp, inputs).double(), -1) if qfix is None else qfix
    return -(q * logp).sum(-1)


SUB, PER = 1024, 4
log = []
for it in range(args.steps // PER):
    b = aug_batch(256)
    S, _ = rollout(b, LOOPS, lambda s: indep_noise)                     # [16, 256, T, d]
    with torch.no_grad():
        lp = logp_of(S)
        ce = cell_ce(lp, b['labels'])
    N = S.shape[0] * S.shape[1]
    Sf, lpf, cef = S.reshape(N, T, -1), lp.reshape(N, T, V), ce.reshape(N, T)
    inp = b['inputs'].repeat(S.shape[0], 1)
    lab = b['labels'].repeat(S.shape[0], 1)
    for _ in range(PER):
        j = torch.randint(0, N, (SUB,), device=DEV)
        lb = F.cross_entropy(graders['board'](lpf[j], inp[j]).reshape(-1, V), lab[j].reshape(-1))
        lc = F.cross_entropy(graders['cell'](lpf[j], inp[j]).reshape(-1, V), lab[j].reshape(-1))
        lh = (F.softplus(chat(Sf[j]).squeeze(-1)) - cef[j].float()).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        (lb + lc + lh).backward()
        opt.step()
        sched.step()
    if (it + 1) % (250 // PER) == 0:
        with torch.no_grad():
            pce = cef.mean().item()
        log.append(dict(step=(it + 1) * PER, board=lb.item(), cell=lc.item(), chat=lh.item(), model_ce=pce))
        print(f'[TRAIN] step {(it + 1) * PER} q-CE board {lb.item():.4f} cell {lc.item():.4f} (model p-CE {pce:.4f}) '
              f'chat MSE {lh.item():.4f}  {time.time() - t0:.0f}s', flush=True)
graders.eval()
chat.eval()
for p_ in params:
    p_.requires_grad_(False)
res = dict(ckpt=args.ckpt, step=ck['step'], args=vars(args), train_log=log)

# ---------------------------------------------------------------------------------------------- unsolved at seg 4
allb = mk(te_x, te_y)
with torch.no_grad():
    carry = m.initial_carry(allb)
    for s in range(LOOPS):
        hk.reset(noise=common_noise(100 + s, len(te_x)))
        carry, logits = m(carry, allb)
    solved4 = (logits.argmax(-1) == allb['labels']).all(-1).cpu().numpy()
stuck = np.nonzero(~solved4)[0]
res['test_n'], res['unsolved_seg4'] = int(len(te_x)), int(len(stuck))
print(f'[GATE] test puzzles unsolved at seg {LOOPS}: {len(stuck)}/{len(te_x)}', flush=True)


# ---------------------------------------------------------------------------------------------- gap: q against p
@torch.no_grad()
def gap(idx, segs=32, blocks=(16, 17, 20, 32, 64, 128)):
    b = mk(te_x[idx], te_y[idx])
    S, _ = rollout(b, segs, lambda s: common_noise(200 + s, len(idx)))
    out = {}
    for n in blocks:
        h = S[n - 1]
        lp = logp_of(h)
        ce = cell_ce(lp, b['labels'])
        row = dict(p_ce=ce.mean().item(), p_acc=(lp.argmax(-1) == b['labels']).double().mean().item(),
                   p_exact=int((lp.argmax(-1) == b['labels']).all(-1).sum()))
        for g in ('board', 'cell'):
            ql = graders[g](lp, b['inputs']).double()
            row[f'{g}_q_ce'] = F.cross_entropy(ql.reshape(-1, V), b['labels'].reshape(-1)).item()
            row[f'{g}_q_acc'] = (ql.argmax(-1) == b['labels']).double().mean().item()
            row[f'{g}_q_exact'] = int((ql.argmax(-1) == b['labels']).all(-1).sum())
            j = judge(g, lp, b['inputs']).mean(-1)
            row[f'{g}_corr_board_across_puzzles'] = torch.corrcoef(torch.stack((j, ce.mean(-1))))[0, 1].item()
        j = judge('chat', lp, b['inputs'], h).mean(-1)
        row['chat_corr_board_across_puzzles'] = torch.corrcoef(torch.stack((j, ce.mean(-1))))[0, 1].item()
        out[f'block{n}'] = row
    return out


res['gap_stuck'] = gap(stuck)
print('[GAP] puzzles unsolved at seg 4 (all of them): CE and accuracy of the model p vs the graders q', flush=True)
for k, r in res['gap_stuck'].items():
    print(f'   {k:9s} p: CE {r["p_ce"]:.4f} acc {r["p_acc"]:.4f} exact {r["p_exact"]:4d} | board q: CE {r["board_q_ce"]:.4f} '
          f'acc {r["board_q_acc"]:.4f} exact {r["board_q_exact"]:4d} | cell q: CE {r["cell_q_ce"]:.4f} acc '
          f'{r["cell_q_acc"]:.4f} exact {r["cell_q_exact"]:4d} | corr(judgement, CE) across puzzles board '
          f'{r["board_corr_board_across_puzzles"]:+.3f} cell {r["cell_corr_board_across_puzzles"]:+.3f} chat '
          f'{r["chat_corr_board_across_puzzles"]:+.3f}', flush=True)

SIGNALS = ('board', 'board_fix', 'cell', 'cell_fix', 'chat', 'ent')


def wcorr(a, b):
    """Within-puzzle correlation over the K draws [B0, K] -> [B0] (NaN when a side is constant)."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))


@torch.no_grad()
def corr_probe(idx, s0, K, horizon):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    _, carry = rollout(b, s0, lambda s: common_noise(300 + s, B0))
    c, bb = rep_carry(carry, K), rep(b, K)
    base = common_noise(900 + s0, B0, K)
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(noise=lambda i, shape: indep_noise(i, shape) if i == 0 else base(i, shape))
        c2, _ = m(c, bb)
        states = hk.hs + [hk.final]                    # probe input, then the outputs of blocks 0..horizon
    finally:
        cfg.blocks_per_seg = nb0
    lab, inp = bb['labels'], bb['inputs']
    lp0 = logp_of(states[0])
    open0 = ~(lp0.argmax(-1) == lab).all(-1).reshape(B0, K)[:, 0]      # still unsolved at the probe start
    qfix = {g: torch.softmax(graders[g](lp0, inp).double(), -1) for g in ('board', 'cell')}
    at = {}
    for j in sorted({1, min(4, horizon + 1), horizon + 1}):
        h = states[j]
        lp = logp_of(h)
        ce = cell_ce(lp, lab)
        sig = {}
        for name in SIGNALS:
            qf = qfix[name.split('_')[0]] if name.endswith('_fix') else None
            sig[name] = judge(name, lp, inp, h, qf)
        at[j] = (ce, sig, (lp.argmax(-1) == lab).all(-1))
    R = lambda x: x.reshape(B0, K)
    out = dict(blocks_after=sorted(at))
    L_end = R(at[horizon + 1][0].mean(-1))
    out['solved_frac_end'] = at[horizon + 1][2].double().mean().item()
    out['puzzles_with_mixed_outcome'] = int(((R(at[horizon + 1][2].double()).mean(1) > 0) &
                                              (R(at[horizon + 1][2].double()).mean(1) < 1)).sum())
    out['true_ce_std_within'] = R(at[horizon + 1][0].mean(-1)).std(1).median().item()
    out['open_at_start'] = int(open0.sum())
    for j, (ce, sig, _) in at.items():
        L = R(ce.mean(-1))
        for name, v in sig.items():
            c = wcorr(R(v.mean(-1)), L)
            out[f'{name}@{j}'] = c.nanmedian().item()
            out[f'{name}@{j}_open'] = c[open0].nanmedian().item()
            if j == 1 and horizon + 1 != 1:
                c = wcorr(R(v.mean(-1)), L_end)
                out[f'{name}@1->end'] = c.nanmedian().item()
                out[f'{name}@1->end_open'] = c[open0].nanmedian().item()
    # per-cell, within cell: correlation over the draws pooled over the cells of a puzzle (cell means removed)
    ce, sig, _ = at[horizon + 1]
    for name in ('board_fix', 'board', 'cell_fix', 'chat'):
        a = sig[name].reshape(B0, K, T)
        y = ce.reshape(B0, K, T)
        a, y = a - a.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
        out[f'percell_{name}@end'] = ((a * y).sum((1, 2)) / (a.norm(dim=(1, 2)) * y.norm(dim=(1, 2)))).nanmedian().item()
    return out


def forward_from(carry, b, horizon, z, seed=777):
    """Blocks 0..horizon from carry with z added to W~ at block 1 (common noise); log p, final h, probe-input h."""
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(noise=common_noise(seed, len(b['labels'])), leaf=z, leaf_at=1)
        m(carry, b)
        h, h0 = hk.final, hk.hs[0]
    finally:
        cfg.blocks_per_seg = nb0
    return logp_of(h), h, h0


def line_probe(idx, s0, horizon, fracs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    _, carry = rollout(b, s0, lambda s: common_noise(300 + s, B0))
    lab, inp = b['labels'], b['inputs']
    with torch.no_grad():
        lp0, h0, hstart = forward_from(carry, b, horizon, torch.zeros(B0, H, D, D, device=DEV))
        qstart = {g: torch.softmax(graders[g](logp_of(hstart), inp).double(), -1) for g in ('board', 'cell')}
        qend0 = torch.softmax(graders['board'](lp0, inp).double(), -1)
        base_ce = cell_ce(lp0, lab).mean(-1)
        base_ok = lp0.argmax(-1) == lab
    objectives = {
        'ce (oracle)': lambda lp, h: cell_ce(lp, lab).mean(-1),
        'board': lambda lp, h: judge('board', lp, inp).mean(-1),
        'board_fix (q at probe start)': lambda lp, h: judge('board', lp, inp, qfix=qstart['board']).mean(-1),
        'board_end0 (q at base end)': lambda lp, h: judge('board', lp, inp, qfix=qend0).mean(-1),
        'cell_fix': lambda lp, h: judge('cell', lp, inp, qfix=qstart['cell']).mean(-1),
        'chat': lambda lp, h: judge('chat', lp, inp, h).mean(-1),
        'ent': lambda lp, h: judge('ent', lp, inp).mean(-1),
    }
    wnorm = carry.stdp_w.double().reshape(B0, -1).norm(dim=-1)
    grads = {}
    for name, f in objectives.items():
        z = torch.zeros(B0, H, D, D, device=DEV, requires_grad=True)
        with torch.enable_grad():
            lp, h, _ = forward_from(carry, b, horizon, z)
            g, = torch.autograd.grad(f(lp, h).sum(), z)
        grads[name] = g.double().reshape(B0, -1)
    gce = grads['ce (oracle)']
    out = dict(base_ce=base_ce.mean().item(), base_exact=int(base_ok.all(-1).sum()), n=B0, objectives={})
    for name, g in grads.items():
        cos = (g * gce).sum(-1) / (g.norm(dim=-1) * gce.norm(dim=-1)).clamp_min(1e-30)
        rows = []
        dirn = -g / g.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        with torch.no_grad():
            for fr in fracs:
                lp, _, _ = forward_from(carry, b, horizon, (fr * wnorm[:, None] * dirn).reshape(B0, H, D, D).float())
                ok = lp.argmax(-1) == lab
                rows.append(dict(frac=fr, dce=(cell_ce(lp, lab).mean(-1) - base_ce).mean().item(),
                                 dexact=int(ok.all(-1).sum() - base_ok.all(-1).sum()),
                                 dacc=(ok.double().mean(-1) - base_ok.double().mean(-1)).mean().item()))
        out['objectives'][name] = dict(cos_median=cos.median().item(), cos_pos=(cos > 0).double().mean().item(),
                                       line=rows)
    return out


idx = stuck[:args.npuz]
res['corr'], res['line'] = {}, {}
for s0 in (4, 8, 16):
    for horizon in (3, 11):
        key = f'block{s0 * NB + 1}_h{horizon}'
        o = corr_probe(idx, s0, args.K, horizon)
        res['corr'][key] = o
        print(f'[CORR] {key} ({len(idx)} puzzles unsolved at seg {LOOPS}, {args.K} draws of block {s0 * NB + 1} noise; '
              f'solved at end {o["solved_frac_end"]:.2f}, mixed-outcome puzzles {o["puzzles_with_mixed_outcome"]}, '
              f'true-CE std within {o["true_ce_std_within"]:.4f}): median within-puzzle corr(judgement, true CE)', flush=True)
        print(f'   (in brackets: the {o["open_at_start"]} puzzles still unsolved at the probe start)', flush=True)
        for name in SIGNALS:
            cells = '  '.join(f'@{j} {o[f"{name}@{j}"]:+.3f} [{o[f"{name}@{j}_open"]:+.3f}]' for j in o['blocks_after'])
            extra = (f'  @1->end {o[f"{name}@1->end"]:+.3f} [{o[f"{name}@1->end_open"]:+.3f}]'
                     if f'{name}@1->end' in o else '')
            pc = f'  per-cell@end {o[f"percell_{name}@end"]:+.3f}' if f'percell_{name}@end' in o else ''
            print(f'   {name:10s} {cells}{extra}{pc}', flush=True)
    key = f'block{s0 * NB + 1}_h11'
    o = line_probe(idx, s0, 11)
    res['line'][key] = o
    print(f'[LINE] {key}: base CE {o["base_ce"]:.4f}, exact {o["base_exact"]}/{o["n"]}; descend each objective wrt W~ '
          f'(step = frac x |W~|): true dCE / dexact / dacc', flush=True)
    for name, v in o['objectives'].items():
        rows = '  '.join(f'{r["frac"]:g}: {r["dce"]:+.3f}/{r["dexact"]:+d}/{r["dacc"]:+.3f}' for r in v['line'])
        print(f'   {name:30s} cos(grad, grad CE) {v["cos_median"]:+.3f} (pos {v["cos_pos"]:.2f}) | {rows}', flush=True)

res['seconds'] = time.time() - t0
os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
json.dump(res, open(os.path.join(REPO, args.out), 'w'), indent=1)
torch.save(dict(graders=graders.state_dict(), chat=chat.state_dict(), args=vars(args)),
           os.path.join(REPO, args.out.replace('.json', '_graders.pt')))
print(f'[GATE] done in {res["seconds"]:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
      flush=True)
