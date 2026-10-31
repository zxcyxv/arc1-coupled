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


t0 = time.time()
res = dict(run=RUN, step=STEP, sigma=SIG, beta=(cfg.tf_beta_scale * m.block.tf_beta).tolist(),
           eta=m.block.eta(cfg).tolist(), lam=m.block.tf_lam().tolist(), dims=H * D * D)

# ---------------------------------------------------------------- unsolved puzzles at the end of training depth
allidx = np.arange(len(X))
carry, b, ex = run(allidx, LOOPS, seed=1)
unsolved = allidx[~ex[-1].cpu().numpy()]
res['unsolved_at_train_depth'] = int(len(unsolved))
print(f'step {STEP} beta {[round(x, 3) for x in res["beta"]]} eta {res["eta"][0]:.3f}; unsolved at seg{LOOPS}: '
      f'{len(unsolved)}/{len(X)}', flush=True)

# ---------------------------------------------------------------- A. share of the third-factor part of W
rng = np.random.default_rng(0)
idxA = np.concatenate([unsolved[:128], rng.choice(np.setdiff1d(allidx, unsolved), 128, replace=False)])
b = batch(idxA)
carry = m.initial_carry(b)
hk.A = None
share = []
with torch.no_grad():
    for s in range(32):
        hk.reset()
        hk.keep = ('share',)
        hk.noise = common_noise(500 + s, len(idxA))
        carry, logits = m(carry, b)
        for r in hk.writes:
            share.append({k: v.mean().item() for k, v in r.items()})
blocks = [0, 3, 7, 15, 31, 63, 127]
res['share'] = {f'block{n + 1}': dict(A_over_W=share[n]['a'] / share[n]['w'], A_over_S=share[n]['a'] / share[n]['s'],
                                      inc_over_etaS=share[n]['inc'] / share[n]['es']) for n in blocks}
print('A. third-factor part of W (beta as trained):', flush=True)
for k, v in res['share'].items():
    print(f'   {k:9s} |A|/|W| {v["A_over_W"]:.4f}  |A|/|S| {v["A_over_S"]:.4f}  per-block |inc|/|eta S| {v["inc_over_etaS"]:.4f}')


# ---------------------------------------------------------------- B + C at several depths
def measure(idx, s0, K, horizon):
    B0 = len(idx)
    carry, b, _ = run(idx, s0, seed=2)
    c = rep_carry(carry, K)
    bb = {k: rep(v, K) for k, v in b.items()}
    nb0 = cfg.blocks_per_seg
    cfg.blocks_per_seg = horizon + 1                # block 0 perturbed, blocks 1..horizon after it
    base = common_noise(900 + s0, B0, K)

    def noise(i, shape):
        if i == 0:                                   # independent draws per replica
            return SIG * torch.randn(shape, device=DEV)
        return base(i, shape)
    try:
        hk.reset()
        hk.noise = noise
        hk.keep = ('e',)
        hk.leaf = torch.zeros(B0 * K, H, D, D, device=DEV, requires_grad=True)
        hk.leaf_at = 1                               # W~ of block 1: where delta_1 * C_0 is added
        with torch.enable_grad():
            _, logits = m(c, bb)
            L = cell_ce(logits, bb['labels'])         # [B0*K]
            g, = torch.autograd.grad(L.sum(), hk.leaf)
    finally:
        cfg.blocks_per_seg = nb0
    e0 = hk.writes[0]['e'].double().reshape(B0, K, -1)                # (xi / sigma^2)^T k~ of block 0
    d1 = hk.deltas[1].double().reshape(B0, K)                         # delta crediting block 0
    g = g.double().reshape(B0, K, -1)
    L = L.detach().double().reshape(B0, K)
    out = {}
    adv = -(L - L.mean(1, keepdim=True))                           # oracle: true outcome, perfect baseline
    for name, sig in (('td', d1), ('oracle', adv)):
        u = sig[..., None] * e0                                    # [B0, K, N]
        mu = u.mean(1)
        resid = (u - mu[:, None]).pow(2).sum(-1).mean(1).sqrt()
        gbar = g.mean(1)
        ghat = gbar / gbar.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        p = -(u * ghat[:, None]).sum(-1)                           # projection on the descent direction
        cos = (mu * -gbar).sum(-1) / (mu.norm(dim=-1) * gbar.norm(dim=-1)).clamp_min(1e-30)
        out[name] = dict(vec_snr=(mu.norm(dim=-1) / resid).median().item(),
                         cos_mean_update_vs_descent=cos.median().item(),
                         cos_mean_update_vs_descent_mean=cos.mean().item(),
                         frac_lanes_cos_pos=(cos > 0).double().mean().item(),
                         dir_snr_per_draw=(p.mean(1) / p.std(1)).median().item())
    out['delta_state_part'] = d1.mean(1).abs().median().item()
    out['delta_noise_part'] = d1.std(1).median().item()
    out['delta_abs'] = d1.abs().mean().item()
    out['adv_std'] = adv.std(1).median().item()
    out['corr_delta_vs_true_adv'] = torch.stack([torch.corrcoef(torch.stack((d1[i], adv[i])))[0, 1]
                                                 for i in range(B0)]).nanmedian().item()
    beta = (cfg.tf_beta_scale * m.block.tf_beta).double().view(1, H, 1).expand(1, H, D * D).reshape(1, 1, -1)
    eta = m.block.eta(cfg).double().view(1, H, 1).expand(1, H, D * D).reshape(1, 1, -1)
    lam = m.block.tf_lam().double().view(1, H, 1).expand(1, H, D * D).reshape(1, 1, -1)
    inc = eta * beta * (1 - lam) * d1[..., None] * e0                 # this block's contribution to W~ at block 1
    dL = (g * inc).sum(-1)                                          # first-order change of L_future
    out['first_order_dL_mean'] = dL.mean().item()
    out['first_order_dL_rms'] = dL.pow(2).mean().sqrt().item()
    out['L_future_mean'] = L.mean().item()
    out['grad_norm_median'] = g.norm(dim=-1).median().item()
    return out


res['depths'] = {}
idxB = unsolved[:48]
for s0, horizon in ((1, 3), (1, 11), (4, 11), (10, 11)):
    o = measure(idxB, s0, K=48, horizon=horizon)
    key = f'block{s0 * cfg.blocks_per_seg + 1}_h{horizon}'
    res['depths'][key] = o
    print(f'B/C at {key} (48 puzzles unsolved at seg{LOOPS}, 48 draws each):', flush=True)
    for name in ('td', 'oracle'):
        v = o[name]
        print(f'   {name:6s} vector SNR {v["vec_snr"]:.3f}  cos(E u, -grad) median {v["cos_mean_update_vs_descent"]:+.3f} '
              f'mean {v["cos_mean_update_vs_descent_mean"]:+.3f} (pos {v["frac_lanes_cos_pos"]:.2f})  '
              f'directional SNR/draw {v["dir_snr_per_draw"]:+.3f}')
    print(f'   delta: state part {o["delta_state_part"]:.2e}  noise part {o["delta_noise_part"]:.2e}  '
          f'corr(delta, true advantage) {o["corr_delta_vs_true_adv"]:+.3f}', flush=True)
    print(f'   first-order dL from this update (beta as trained): mean {o["first_order_dL_mean"]:+.2e} rms '
          f'{o["first_order_dL_rms"]:.2e}; L_future {o["L_future_mean"]:.3f}', flush=True)
res['seconds'] = time.time() - t0
if OUT:
    json.dump(res, open(OUT, 'w'), indent=1)
print(f'done in {res["seconds"]:.0f}s')
