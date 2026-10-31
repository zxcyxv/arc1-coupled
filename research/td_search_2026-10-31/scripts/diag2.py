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


def candidates(idx, s0, K, horizon):
    """K = 2J draws of block-0 noise in antithetic pairs (xi, -xi); every candidate signal r_k measured at the end
    of the horizon (and delta_1, C^ one block later); gradient of the true loss wrt W~ at block 1 (leaf)."""
    B0 = len(idx)
    carry, b, _ = run(idx, s0, seed=2)
    c = rep_carry(carry, K)
    bb = {k: rep(v, K) for k, v in b.items()}
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1
    base = common_noise(900 + s0, B0, K)
    def noise(i, shape):
        if i == 0:
            z = torch.randn((shape[0] // 2,) + shape[1:], device=DEV)
            return SIG * torch.stack((z, -z), 1).reshape(shape)          # replica 2j: +z, 2j+1: -z
        return base(i, shape)
    try:
        hk.reset(); hk.noise = noise; hk.keep = ('e',)
        hk.leaf = torch.zeros(B0 * K, H, D, D, device=DEV, requires_grad=True); hk.leaf_at = 1
        with torch.enable_grad():
            carry2, logits = m(c, bb)
            L = cell_ce(logits, bb['labels'])
            g, = torch.autograd.grad(L.sum(), hk.leaf)
    finally:
        cfg.blocks_per_seg = nb0
    with torch.no_grad():
        p = F.softmax(logits.double(), -1)
        ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean(-1)
        conf = p.max(-1).values.mean(-1)
        chat_end, v_end = (x.mean(-1) for x in m.tf_head(carry2.hidden))
    e0 = hk.writes[0]['e'].double().reshape(B0, K, -1)
    g = g.double().reshape(B0, K, -1)
    R = lambda x: x.detach().double().reshape(B0, K)
    sigs = {'oracle: -true CE (labels)': -R(L), 'delta_1 (current rule)': R(hk.deltas[1]),
            'C^ one block later (-)': -R(hk.deltas[1] * 0 + 0) if False else None,
            '-C^ at horizon end': -R(chat_end), 'V at horizon end': R(v_end),
            '-entropy at horizon end': -R(ent), 'confidence at horizon end': R(conf)}
    sigs = {k: v for k, v in sigs.items() if v is not None}
    adv_true = -R(L)
    gbar = g.mean(1); ghat = gbar / gbar.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    out = {}
    for name, r in sigs.items():
        row = {}
        for bl, a in (('mean', r - r.mean(1, keepdim=True)),
                      ('pair', (r.reshape(B0, K // 2, 2) @ torch.tensor([0.5, -0.5], dtype=torch.float64, device=DEV)
                                ).repeat_interleave(2, 1) * torch.tensor([1.0, -1.0], device=DEV, dtype=torch.float64).repeat(K // 2))):
            u = a[..., None] * e0
            mu = u.mean(1)
            cos = (mu * -gbar).sum(-1) / (mu.norm(dim=-1) * gbar.norm(dim=-1)).clamp_min(1e-30)
            proj = -(u * ghat[:, None]).sum(-1)
            row[bl] = (cos.median().item(), (cos > 0).double().mean().item(), (proj.mean(1) / proj.std(1)).median().item())
        corr = torch.stack([torch.corrcoef(torch.stack((r[i], adv_true[i])))[0, 1] for i in range(B0)]).nanmedian().item()
        out[name] = dict(row, corr_true=corr)
    return out


res = {}
for s0, horizon in ((1, 3), (4, 11), (10, 11)):
    key = f'block{s0 * cfg.blocks_per_seg + 1}_h{horizon}'
    o = candidates(unsolved[:48], s0, 48, horizon)
    res[key] = o
    print(f'== {key}: cos(E u, -grad) median (frac>0) | directional SNR/draw | corr with true advantage', flush=True)
    for name, v in o.items():
        print(f'   {name:28s} mean-baseline cos {v["mean"][0]:+.3f} ({v["mean"][1]:.2f}) snr {v["mean"][2]:+.3f} | '
              f'antithetic-pair cos {v["pair"][0]:+.3f} ({v["pair"][1]:.2f}) snr {v["pair"][2]:+.3f} | corr {v["corr_true"]:+.3f}',
              flush=True)
if OUT:
    json.dump(res, open(OUT, 'w'), indent=1)
print(f'done in {time.time() - t0:.0f}s', flush=True)
