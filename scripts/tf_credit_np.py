#!/usr/bin/env python
"""Credit resolution of the node-perturbation update (frozen checkpoint, no training): one board advantage against
per-cell advantages, with the true outcome first (oracle) and then the graders of scripts/tf_grader_gate.py.

Puzzles unsolved at seg 4 (same selection and noise protocol as the gate); probe at block j (17 or 33): K independent
draws of block j's v-noise xi, common noise elsewhere; horizon h blocks after block j. Eligibility of token t:
e_t = (xi_t / sigma^2) k~_t^T (block j, per head). Updates for W~ at block j + 1, per draw k:
    (a) board    u = A sum_t e_t,      A   = -(L_k - mean_k L)          L = mean-cell true CE at the horizon end
    (b) per-cell u = sum_t A_t e_t,    A_t = -(CE_t,k - mean_k CE_t)    cell t's true CE at the horizon end
    and both with a grader's judgement in place of the true CE (A_t from grader differences across draws).
Per puzzle, against g = dL/dW~ at block j + 1 (a leaf there): cos(E_k u, -E_k g) (median over puzzles, fraction > 0),
the directional SNR per draw E<u, -g^> / std<u, -g^>, the vector SNR |E u| / rms|u - E u|, and a line search along
E_k u (and along -E_k g, the ceiling): true dCE / dexact after the horizon, step = frac x |W~|. Graders: within-cell
corr(A_t grader, A_t true) across draws (median over cells whose true CE varies; and pooled per puzzle).
Usage: python scripts/tf_credit_np.py [--ckpt ...] [--graders runs/bg6/gate_s30k_graders.pt]
"""
import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lt.data import load_data  # noqa: E402
from lt.model import LT, Carry, ModelConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='runs/tf6_hard/bands/step_30000.pt')
ap.add_argument('--graders', default='runs/bg6/gate_s30k_graders.pt')
ap.add_argument('--out', default='runs/bg6/credit_s30k.json')
ap.add_argument('--K', type=int, default=48)
ap.add_argument('--npuz', type=int, default=64)
ap.add_argument('--chunk', type=int, default=16)
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

gs = torch.load(os.path.join(REPO, args.graders), map_location='cpu', weights_only=False)
spec = importlib.util.spec_from_file_location('gate_defs', os.path.join(REPO, 'scripts', 'tf_grader_gate.py'))
src = open(spec.origin).read()
src = src[src.index('class Grader'):src.index('graders = nn.ModuleDict')]
ns = dict(nn=torch.nn, torch=torch, F=F, T=T, V=V, args=argparse.Namespace(**gs['args']))
exec(src, ns)
graders = {}
for name, attn in (('board', True), ('cell', False)):
    g = ns['Grader'](attn).to(DEV)
    g.load_state_dict({k[len(name) + 1:]: v for k, v in gs['graders'].items() if k.startswith(name + '.')})
    g.eval()
    for p_ in g.parameters():
        p_.requires_grad_(False)
    graders[name] = g


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
        if not self.elig:                             # block 0 of the call: its xi and k~ (as in bands_step)
            with torch.no_grad():
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


def judge(name, logp, inputs, qfix=None):
    q = torch.softmax(graders[name](logp, inputs).double(), -1) if qfix is None else qfix
    return -(q * logp).sum(-1)


_, _, te_x, te_y, _, _ = load_data(os.path.join(REPO, gs['args']['data']))
allb = mk(te_x, te_y)
with torch.no_grad():
    carry = m.initial_carry(allb)
    for s in range(LOOPS):
        hk.reset(noise=common_noise(100 + s, len(te_x)))
        carry, logits = m(carry, allb)
    stuck = np.nonzero(~(logits.argmax(-1) == allb['labels']).all(-1).cpu().numpy())[0]
idx = stuck[:args.npuz]
print(f'[CREDIT] {args.ckpt}: {len(stuck)} test puzzles unsolved at seg {LOOPS}; {len(idx)} puzzles x {args.K} draws',
      flush=True)


def probe_chunk(carry, b, s0, horizon, K):
    """One chunk of puzzles: per draw the eligibility, the true per-cell CE, the graders' per-cell judgements at the
    horizon end and g = dL/dW~ at block 1."""
    B0 = len(b['labels'])
    c, bb = Carry(**{f.name: rep(getattr(carry, f.name), K) for f in dataclasses.fields(carry)}), rep(b, K)
    base = common_noise(900 + s0, len(idx), K)
    sel = None
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(noise=lambda i, shape: SIG * torch.randn(shape, device=DEV) if i == 0 else base(i, shape)[sel],
                 leaf=torch.zeros(B0 * K, H, D, D, device=DEV, requires_grad=True), leaf_at=1)
        sel = probe_chunk.sel
        with torch.enable_grad():
            m(c, bb)
            hend, h0 = hk.final, hk.hs[0]
            ce = cell_ce(logp_of(hend), bb['labels'])                       # [B0 K, T]
            g, = torch.autograd.grad(ce.mean(-1).sum(), hk.leaf)
    finally:
        cfg.blocks_per_seg = nb0
    with torch.no_grad():
        lpe, lp0 = logp_of(hend.detach()), logp_of(h0.detach())
        inp = bb['inputs']
        jd = {'board': judge('board', lpe, inp),
              'board_fix': judge('board', lpe, inp, torch.softmax(graders['board'](lp0, inp).double(), -1)),
              'cell': judge('cell', lpe, inp)}
        xi, kr = hk.elig[0]
        open0 = ~(lp0.argmax(-1) == bb['labels']).all(-1)
        wrong = (lpe.argmax(-1) != bb['labels']).double()                 # [B0 K, T] at the horizon end
    return dict(ce=ce.detach().double(), g=g.double(), xi=xi, kr=kr, jd=jd, open0=open0, wrong=wrong)


def measure(s0, horizon, K, fracs=(0.01, 0.03, 0.1, 0.3, 1.0)):
    B0 = len(idx)
    b = mk(te_x[idx], te_y[idx])
    with torch.no_grad():
        carry = m.initial_carry(b)
        for s in range(s0):
            hk.reset(noise=common_noise(300 + s, B0))
            carry, _ = m(carry, b)
    U = {}                                                                 # name -> [B0, K, H D D] (float32)
    G, CE, OPEN, CORR, CORRP = [], [], [], {k: [] for k in ('board', 'board_fix', 'cell')}, {k: [] for k in ('board', 'board_fix', 'cell')}
    WRONG = []
    for lo in range(0, B0, args.chunk):
        hi = min(B0, lo + args.chunk)
        part = Carry(**{f.name: (getattr(carry, f.name)[lo:hi] if torch.is_tensor(getattr(carry, f.name)) else
                                 ({k: v[lo:hi] for k, v in getattr(carry, f.name).items()}
                                  if isinstance(getattr(carry, f.name), dict) else getattr(carry, f.name)))
                        for f in dataclasses.fields(carry)})
        bp = {k: v[lo:hi] for k, v in b.items()}
        probe_chunk.sel = torch.arange(lo * K, hi * K, device=DEV)
        r = probe_chunk(part, bp, s0, horizon, K)
        n = hi - lo
        ce = r['ce'].reshape(n, K, T)
        adv_cell = -(ce - ce.mean(1, keepdim=True))                        # A_t
        adv_board = adv_cell.mean(-1)                                      # A = -(L - mean L)
        e = lambda w: (torch.einsum('bt,bhtd,bhte->bhde', w.reshape(n * K, T).float(), r['xi'], r['kr'])
                       / SIG ** 2).reshape(n, K, -1)
        full = (torch.einsum('bhtd,bhte->bhde', r['xi'], r['kr']) / SIG ** 2).reshape(n, K, -1)
        rows = {'(a) board, true CE (oracle)': adv_board.float()[..., None] * full,
                '(b) per-cell, true CE (oracle)': e(adv_cell)}
        for gname, jv in r['jd'].items():
            j = jv.reshape(n, K, T)
            a_cell = -(j - j.mean(1, keepdim=True))
            rows[f'(a) board, grader {gname}'] = a_cell.mean(-1).float()[..., None] * full
            rows[f'(b) per-cell, grader {gname}'] = e(a_cell)
            x, y = a_cell, adv_cell                                        # within-cell corr across draws
            num = (x * y).sum(1)
            den = x.norm(dim=1) * y.norm(dim=1)
            live = y.std(1) > 1e-4
            CORR[gname].append(torch.where(live, num / den.clamp_min(1e-30), torch.nan).reshape(-1))
            CORRP[gname].append(num.sum(-1) / (x.norm(dim=(1, 2)) * y.norm(dim=(1, 2))).clamp_min(1e-30))
        for name, u in rows.items():
            U.setdefault(name, []).append(u)
        WRONG.append((r['wrong'].reshape(n, K, T).mean(1) > 0.5).reshape(-1))      # wrong in most draws
        G.append(r['g'].reshape(n, K, -1))
        CE.append(ce)
        OPEN.append(r['open0'].reshape(n, K)[:, 0])
    U = {k: torch.cat(v).double() for k, v in U.items()}
    g, open0 = torch.cat(G), torch.cat(OPEN)
    gbar = g.mean(1)
    ghat = gbar / gbar.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    out = dict(n=B0, K=K, open_at_start=int(open0.sum()), rows={},
               L_end=torch.cat(CE).mean().item(),
               grader_corr={k: dict(within_cell_median=torch.cat(v).nanmedian().item(),
                                    within_cell_live_cells=int((~torch.cat(v).isnan()).sum()),
                                    right_cells_median=torch.cat(v)[~torch.cat(WRONG)].nanmedian().item(),
                                    wrong_cells_median=torch.cat(v)[torch.cat(WRONG)].nanmedian().item(),
                                    wrong_live_cells=int((~torch.cat(v)[torch.cat(WRONG)].isnan()).sum()),
                                    pooled_per_puzzle_median=torch.cat(CORRP[k]).nanmedian().item())
                            for k, v in CORR.items()})
    # line search along E u (and -E g): common noise, step at W~ of block 1
    wnorm = carry.stdp_w.double().reshape(B0, -1).norm(dim=-1)

    def fwd(z):
        nb0 = cfg.blocks_per_seg
        cfg.blocks_per_seg = horizon + 1
        try:
            hk.reset(noise=common_noise(777, B0), leaf=z, leaf_at=1)
            with torch.no_grad():
                m(carry, b)
            lp = logp_of(hk.final)
        finally:
            cfg.blocks_per_seg = nb0
        return cell_ce(lp, b['labels']).mean(-1), (lp.argmax(-1) == b['labels']).all(-1)
    base_ce, base_ok = fwd(torch.zeros(B0, H, D, D, device=DEV))

    def line(dirn):
        rows = []
        for fr in fracs:
            l, ok = fwd((fr * wnorm[:, None] * dirn).reshape(B0, H, D, D).float())
            rows.append(dict(frac=fr, dce=(l - base_ce).mean().item(), dce_open=(l - base_ce)[open0].mean().item(),
                             dexact=int(ok.sum() - base_ok.sum())))
        return rows
    out['ceiling_line_-E[g]'] = line(-ghat)
    for name, u in U.items():
        mu = u.mean(1)
        cos = (mu * -gbar).sum(-1) / (mu.norm(dim=-1) * gbar.norm(dim=-1)).clamp_min(1e-30)
        proj = -(u * ghat[:, None]).sum(-1)
        vec = mu.norm(dim=-1) / (u - mu[:, None]).pow(2).sum(-1).mean(1).sqrt().clamp_min(1e-30)
        out['rows'][name] = dict(cos_median=cos.median().item(), cos_pos=(cos > 0).double().mean().item(),
                                 cos_open_median=cos[open0].median().item(),
                                 dir_snr=(proj.mean(1) / proj.std(1)).median().item(), vec_snr=vec.median().item(),
                                 line=line(mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-30)))
    out['base_ce'], out['base_exact'] = base_ce.mean().item(), int(base_ok.sum())
    return out


res = dict(ckpt=args.ckpt, graders=args.graders, K=args.K, npuz=len(idx), probes={})
fmt = lambda rows: '  '.join(f'{r["frac"]:g}: {r["dce"]:+.3f} ({r["dce_open"]:+.3f})/{r["dexact"]:+d}' for r in rows)
for s0, horizon in ((4, 11), (8, 11), (4, 3), (8, 3)):
    key = f'block{s0 * NB + 1}_h{horizon}'
    o = measure(s0, horizon, args.K)
    res['probes'][key] = o
    print(f'[CREDIT] {key}: {o["n"]} puzzles ({o["open_at_start"]} open at the probe start) x {o["K"]} draws; '
          f'base CE {o["base_ce"]:.4f} exact {o["base_exact"]}', flush=True)
    print(f'   {"ceiling: step along -E[g]":38s} {"":58s} | {fmt(o["ceiling_line_-E[g]"])}', flush=True)
    for name, v in o['rows'].items():
        print(f'   {name:38s} cos {v["cos_median"]:+.3f} (pos {v["cos_pos"]:.2f}, open {v["cos_open_median"]:+.3f}) '
              f'dirSNR {v["dir_snr"]:+.3f} vecSNR {v["vec_snr"]:.3f} | {fmt(v["line"])}', flush=True)
    for gname, v in o['grader_corr'].items():
        print(f'   corr(A_t grader {gname}, A_t true) within cell across draws: median {v["within_cell_median"]:+.3f} '
              f'({v["within_cell_live_cells"]} cells with varying CE; right cells {v["right_cells_median"]:+.3f}, wrong cells '
              f'{v["wrong_cells_median"]:+.3f} over {v["wrong_live_cells"]}); pooled per puzzle {v["pooled_per_puzzle_median"]:+.3f}',
              flush=True)
json.dump(res, open(os.path.join(REPO, args.out), 'w'), indent=1)
print(f'[CREDIT] done in {time.time() - t0:.0f}s; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB',
      flush=True)
