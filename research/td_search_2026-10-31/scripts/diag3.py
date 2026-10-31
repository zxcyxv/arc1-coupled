"""Diagnosis of the three-factor (node-perturbation) term on a trained checkpoint (6x6, tiny model).

A. share: how large the accumulated third-factor part of W is (side accumulator A_n = (1-eta) A_{n-1} + eta beta
   delta C_{n-1}) against W and against the write S, over a 32-segment run.
B. per-block NP update u = delta_{j+1} * e_j for the perturbation xi_j of one block: K independent draws of xi_j at
   the same state (common noise elsewhere). Vector SNR ||E u|| / rms||u - E u||; delta split into its state part
   (mean over draws) and its noise part (std over draws).
C. alignment with the true descent direction: g_k = d L_future / d W~ at the block where the update is added
   (leaf added to W~ there), L_future = stablemax CE of the output `horizon` blocks later. cos(E u, -E g),
   directional SNR E<-g^, u> / std<-g^, u>, and the same for an oracle that replaces delta by the true outcome
   advantage -(L_k - mean_k L_k) (perfect baseline, true labels).
Usage: python diag.py RUN_DIR [out.json]
"""
import dataclasses, glob, json, math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig, Carry
from lt.data import load_data

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
RUN = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None
MAXB = 4096


def load(run):
    paths = sorted(glob.glob(f'{run}/step_*.pt'), key=lambda p: int(p.split('_')[-1][:-3]))
    ck = torch.load(paths[-1], map_location='cpu', weights_only=False)
    cfg = dict(ck['cfg'])
    cfg.update(batch_size=MAXB, seq_len=cfg['grid'] ** 2 * cfg['num_grids'], vocab_size=cfg['grid'] + 2, amp=False,
               act_ckpt=False)
    m = LT(ModelConfig.from_dict(cfg)).to(DEV)
    m.load_state_dict(ck['model'])
    m.eval()
    loops = m.config.loops
    m.config.loops = 10 ** 6                     # no lane replacement
    return m, ck['step'], loops


m, STEP, LOOPS = load(RUN)
cfg = m.config
SIG = cfg.tf_noise
H, D = m.H, m.dh
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_lt.npz')


def batch(idx):
    n = len(idx)
    return dict(inputs=torch.tensor(X[idx].reshape(n, -1) + 1, dtype=torch.int32, device=DEV),
                labels=torch.tensor(Y[idx].reshape(n, -1) + 1, dtype=torch.int32, device=DEV),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32, device=DEV))


def rep(x, K):
    if torch.is_tensor(x):
        return x.repeat_interleave(K, 0)
    if isinstance(x, dict):
        return {k: v.repeat_interleave(K, 0) for k, v in x.items()}
    return x


def rep_carry(c, K):
    return Carry(**{f.name: rep(getattr(c, f.name), K) for f in dataclasses.fields(c)})


def cell_ce(logits, labels):
    x = logits.double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    lp = torch.log(s / s.sum(-1, keepdim=True))
    return -torch.gather(lp, -1, labels.long().unsqueeze(-1)).squeeze(-1).mean(-1)      # [B]


class Hooks:
    def __init__(self, model):
        self.m = model
        self.orig_in, self.orig_w = model.tf_input, model.tf_write
        model.tf_input, model.tf_write = self.tf_input, self.tf_write
        self.noise = None
        self.reset()

    def reset(self):
        self.deltas, self.writes, self.wcount = [], [], 0
        self.leaf, self.leaf_at, self.keep = None, None, ()
        self.A = getattr(self, 'A', None)

    def tf_input(self, tf, h, i, store):
        elig, delta, n, w0, noise = self.orig_in(tf, h, i, store)
        if self.noise is not None:
            noise = self.noise(i, (h.shape[0], H, h.shape[1], D))
        self.deltas.append(delta)
        return (elig, delta, n, w0, noise)

    def tf_write(self, w, s, tf, dtype, e=None):
        idx = self.wcount
        self.wcount += 1
        if self.leaf_at == idx:
            w = w + self.leaf
        out = self.orig_w(w, s, tf, dtype, e)
        rec = {}
        if 'e' in self.keep:
            rec['e'] = e.detach()
        if 'share' in self.keep:
            L = self.m.block
            eta = L.eta(cfg).to(dtype).view(1, -1, 1, 1)
            beta = (cfg.tf_beta_scale * L.tf_beta).to(dtype).view(1, -1, 1, 1)
            inc = (eta * beta * tf[1].to(dtype).view(-1, 1, 1, 1) * tf[0].to(dtype)).detach()
            self.A = inc if self.A is None else (1 - eta) * self.A + inc
            fn = lambda x: x.detach().flatten(2).norm(dim=-1)           # [B, H]
            rec.update(a=fn(self.A), w=fn(out[0]), s=fn(s), inc=fn(inc), es=fn(eta * s))
        self.writes.append(rec)
        return out


hk = Hooks(m)


def common_noise(seed, B, K=1):
    """sigma * eps drawn for B base lanes, the same for their K replicas, per (seed, block)."""
    def f(i, shape):
        g = torch.Generator(device=DEV).manual_seed(seed * 1000 + i)
        z = torch.randn((B,) + shape[1:], generator=g, device=DEV)
        return SIG * z.repeat_interleave(K, 0)
    return f


def run(idx, segs, seed=0):
    """Fresh lanes on puzzles idx, `segs` segments with common noise; returns the carry and per-segment exact."""
    b = batch(idx)
    carry = m.initial_carry(b)
    ex = []
    with torch.no_grad():
        for s in range(segs):
            hk.reset()
            hk.noise = common_noise(seed * 100 + s, len(idx))
            carry, logits = m(carry, b)
            ex.append((logits.argmax(-1) == b['labels']).all(-1))
    return carry, b, ex



import torch.nn.functional as F
t0 = time.time()
allidx = np.arange(len(X))
_, _, ex = run(allidx, LOOPS, seed=1)
unsolved = allidx[~ex[-1].cpu().numpy()]
print(f'step {STEP}; unsolved at seg{LOOPS}: {len(unsolved)}', flush=True)


CAP = {}
m.classifier.register_forward_hook(lambda mod, inp, out: CAP.__setitem__('h', inp[0]))


def heads(hidden):
    o = m.tf_heads(hidden.float())                                  # classifier input (prefix dropped), not detached
    return F.softplus(o[..., 0]).mean(-1), o[..., 1].mean(-1)


def forward_from(carry, b, horizon, z):
    """Blocks 0..horizon from carry with z added to W~ at block 1; returns per-lane objectives."""
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    try:
        hk.reset(); hk.noise = common_noise(777, len(b['labels'])); hk.keep = ()
        hk.leaf, hk.leaf_at = z, 1
        c2, logits = m(carry, b)
    finally:
        cfg.blocks_per_seg = nb0
    p = F.softmax(logits.double(), -1)
    hc, hv = heads(CAP['h'])
    return dict(ce=cell_ce(logits, b['labels']), chat=hc, negv=-hv,
                ent=-(p * p.clamp_min(1e-30).log()).sum(-1).mean(-1),
                exact=(logits.argmax(-1) == b['labels']).all(-1).double(),
                acc=(logits.argmax(-1) == b['labels']).double().mean(-1))


def study(idx, s0, horizon):
    carry, b, _ = run(idx, s0, seed=2)
    B0 = len(idx)
    z = torch.zeros(B0, H, D, D, device=DEV, requires_grad=True)
    with torch.enable_grad():
        o = forward_from(carry, b, horizon, z)
        grads = {k: torch.autograd.grad(o[k].sum(), z, retain_graph=True)[0].double().reshape(B0, -1)
                 for k in ('ce', 'chat', 'negv', 'ent')}
    gce = grads['ce']
    out = {'cos': {}}
    for k in ('chat', 'negv', 'ent'):
        cos = (grads[k] * gce).sum(-1) / (grads[k].norm(dim=-1) * gce.norm(dim=-1)).clamp_min(1e-30)
        out['cos'][k] = (cos.median().item(), (cos > 0).double().mean().item())
    # W~ scale for the step size: |W~| at block 1 is not exposed; use |grad| normalisation and a grid of step norms
    with torch.no_grad():
        base = forward_from(carry, b, horizon, torch.zeros(B0, H, D, D, device=DEV))
        wnorm = carry.stdp_w.double().reshape(B0, -1).norm(dim=-1)            # |W~| entering this segment
        out['line'] = {}
        for k in ('ce', 'chat', 'ent'):
            dirn = -grads[k] / grads[k].norm(dim=-1, keepdim=True).clamp_min(1e-30)
            rows = []
            for frac in (0.01, 0.03, 0.1, 0.3, 1.0):
                step = (frac * wnorm[:, None] * dirn).reshape(B0, H, D, D).float()
                o2 = forward_from(carry, b, horizon, step)
                rows.append((frac, (o2['ce'] - base['ce']).mean().item(), (o2['exact'] - base['exact']).sum().item(),
                             (o2['acc'] - base['acc']).mean().item()))
            out['line'][k] = rows
        out['base'] = dict(ce=base['ce'].mean().item(), exact=base['exact'].sum().item(), n=B0)
    return out


res = {}
for s0, horizon in ((1, 3), (4, 11), (10, 11)):
    key = f'block{s0 * cfg.blocks_per_seg + 1}_h{horizon}'
    o = study(unsolved[:64], s0, horizon)
    res[key] = o
    print(f'== {key} (64 puzzles unsolved at seg{LOOPS}; base CE {o["base"]["ce"]:.3f}, exact {o["base"]["exact"]:.0f}/64)', flush=True)
    print('   cos(grad of label-free objective, grad of true CE) wrt W~, median (frac>0): ' +
          ', '.join(f'{k} {v[0]:+.3f} ({v[1]:.2f})' for k, v in o['cos'].items()), flush=True)
    for k, rows in o['line'].items():
        print(f'   descend {k:4s}: ' + '  '.join(f'|dW|={fr:g}|W|: dCE {dce:+.3f} dexact {dex:+.0f} dacc {dacc:+.3f}'
                                             for fr, dce, dex, dacc in rows), flush=True)
if OUT:
    json.dump(res, open(OUT, 'w'), indent=1)
print(f'done in {time.time() - t0:.0f}s', flush=True)
