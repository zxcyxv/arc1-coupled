"""Gate for the ARC three-factor plan (a): does changing the DEMONSTRATION lanes' bands W in the direction that lowers
their leave-one-out loss (computable at inference: every demo lane predicts its own output from the other demos'
Theta) help the QUERY? Frozen <= 10 harness checkpoint (arc.small), exact gradients (the ceiling of any three-factor
estimator of the same signal).

Per eval batch (groups = D demos + 1 query): a leaf Z added to every lane's W at block `zb`; L_demo = sum of the demo
lanes' per-lane mean stablemax CE (labels known at inference), L_q = the query lanes' (labels used only to score).
  cos      : cos(dL_demo/dZ, dL_q/dZ) over each group's demo lanes (does lowering the LOO loss move demo W the way the
             query needs?)
  descend  : Z = -frac |W_zb| g^ on the demo lanes only, g = dL_demo/dZ (legit) or dL_q/dZ (oracle for demo lanes),
             and on the query lane only from dL_q/dZ (oracle reference; no reward reaches the query lane in (a)).
Reports all 278 test pairs and the 88 ARC-1 evaluation-task pairs.
Usage: python arc_gate.py CKPT [zb]
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/arc1-coupled')
from arc.model import Coupled, ModelConfig, PAD, VOCAB
from arc.small import Episodes
from arc.evaluate import log_stablemax
import arc.model as AM
AM.sums_fwd = AM.sums_bwd = None              # H100-tuned Triton kernels exceed the 5090's shared memory: PyTorch path

DEV = 'cuda'
CK = sys.argv[1]
ZB = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NG = 8                                                     # groups per batch
FRACS = (0.01, 0.03, 0.1, 0.3, 1.0)

ck = torch.load(CK, map_location='cpu', weights_only=False)
meta = ck['meta']
torch.manual_seed(meta['seed'])                            # buffers (init_hidden, fixed RoPE) as in training
model = Coupled(ModelConfig(**dict(meta['model'], fused_bands=False))).to(DEV)
with torch.no_grad():
    for n, p in model.named_parameters():
        p.copy_(ck['ema'][n].to(p.dtype))
model.eval()
c = model.cfg
data = Episodes(meta['data'], D=meta['D'], holdout=True)
z = np.load(meta['data'], allow_pickle=False)
task_split = np.array([1 if str(nm).startswith('arc1/evaluation/') else 0 for nm in z['task_names']])
items = [(t, p) for t in sorted(data.held) for p in data.test_pairs[t]]   # tasks with a held-out (never trained) train pair
print(f'ckpt step {ck["step"]}, blocks {c.blocks}, leaf at block {ZB}; test pairs {len(items)}, eval-task pairs '
      f'{int(sum(task_split[t] == 1 for t, _ in items))}', flush=True)


def run(batch, Z=None, grad=False):
    """model.segment from fresh lanes with Z added to W before block ZB; gradients from block ZB on if grad."""
    x, labels, group = batch['inputs'].long(), batch['labels'], batch['group'].long()
    L = x.shape[0]
    dt = model.forward_dtype
    demo = batch['demo'].to(dt)
    yl = labels[:, model.O:].long()
    m = (yl != PAD).to(dt)
    y = F.one_hot(yl, VOCAB).to(dt)
    x_in = x[:, :model.O]
    carry = model.initial_carry(L, DEV)
    h, ek, ev, w = carry['h'], carry['ek'], carry['ev'], carry['w']
    theta, gam, r = carry['theta'], carry['gam'], carry['r']
    amp = model.amp(torch.device(DEV))
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
        emb = model.embed(x)
    tables = model.rope_tables() + (model.frame_tables() if c.frames else ())
    wnorm = None
    for i in range(c.blocks):
        on = grad and i >= ZB
        with torch.set_grad_enabled(on):
            if i == ZB:
                theta, r = theta.detach(), r.detach()
                gam = None if gam is None else gam.detach()
                wnorm = w.detach().flatten(1).norm(dim=-1)
                if Z is not None:
                    w = w + Z
            h, ek, ev, w, theta, gam, r = model.layer(h, emb, ek, ev, w, tables, theta, gam, r, x_in, y, m, demo,
                                                      group, amp, True)
    with torch.set_grad_enabled(grad):
        # same post-head path as Coupled.segment (out_ffn)
        from arc.model import rms_norm
        if c.out_ffn:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                base = h[:, model.O:] if c.direct_head else model.embed_scale * emb[:, model.O:].to(h.dtype)
                v = rms_norm(base + model.embed_scale * model.w_fb(r).to(h.dtype))
                g_, a_ = model.out_gate_up(v).chunk(2, dim=-1)
                h_out = rms_norm(v + model.out_down(F.silu(g_) * a_))
        with torch.autocast('cuda', enabled=False):
            if c.out_ffn:
                logits = model.classifier(h_out.to(dt))
            elif c.direct_head:
                logits = model.classifier(h[:, model.O:].to(dt)) + model.tau * r
            else:
                logits = model.classifier.bias.expand(h.shape[0], model.O, VOCAB) + model.tau * r
    return logits, wnorm


def lane_ce(logits, labels):
    yl = labels[:, model.O:].long()
    lab = (yl != PAD).double()
    cell = -torch.gather(log_stablemax(logits.double()), -1, yl[..., None]).squeeze(-1)
    return (cell * lab).sum(-1) / lab.sum(-1).clamp_min(1)                       # [L]


def lane_exact(logits, labels):
    yl = labels[:, model.O:]
    lab = yl != PAD
    return ((logits.argmax(-1) == yl) | ~lab).all(-1)



import types
from arc.model import soft
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LAMS = {'scale': None, 'mode': 'soft', 'theta': None}
MODES = ('soft', 'soft_norm', 'hard_norm')


def _shape(base, x, mode, lam, s):
    """mode soft: soft(x, s lam); soft_norm: that, rescaled per lane to |soft(x, lam)|; hard_norm: soft(x, lam) with the
    entries below (s - 1) lam (extra threshold, s > 1) zeroed, survivors unshrunk, rescaled to |soft(x, lam)|."""
    shp = (-1,) + (1,) * (x.dim() - 1)
    ref = soft(x, lam.view(shp))
    if mode == 'soft':
        return soft(x, (lam * s).view(shp))
    flat = lambda t: t.flatten(1).norm(dim=-1).view(shp)
    if mode == 'soft_norm':
        y = soft(x, (lam * s).view(shp))
    else:
        y = torch.where(ref.abs() > (lam * (s - 1).clamp_min(0)).view(shp), ref, torch.zeros_like(ref))
    return torch.where(flat(y) > 0, y * flat(ref) / flat(y).clamp_min(1e-30), y)


def step_scaled(self, theta, G, S_, gam=None, ext=None):
    """Coupled.step with the lasso threshold scaled per lane by LAMS['scale'] [L] in mode LAMS['mode']."""
    lam1 = (self.cfg.lam_floor + F.softplus(self.lam_raw))
    s, mode = LAMS['scale'], LAMS['mode']
    lam = lam1 * torch.ones_like(s)
    g = theta @ G - S_
    if gam is not None:
        g = g + (gam[..., None, None] * ext[0]).sum(1)
    theta = _shape(None, theta - g, mode, lam, s)
    LAMS['theta'] = theta.detach()
    if gam is None:
        return theta
    Ck, H_, sv = ext
    gg = (theta[:, None] * Ck).sum((-2, -1)) + (H_ * gam[:, None]).sum(-1) - sv
    return theta, _shape(None, gam - gg, mode, lam, s)


def held_batches():
    for i in range(0, len(items), NG):
        groups, held_idx, off = [], [], 0
        for t, p in items[i:i + NG]:
            demos = list(data.train_pairs[t][:data.D - 1]) + [data.held[t]]
            groups.append(data.group(t, demos + [p], 0, None, [True] * len(demos) + [False], [False] * len(demos) + [True]))
            held_idx.append(off + len(demos) - 1)
            off += len(demos) + 1
        b = data.collate(groups, pad_to=NG * (data.D + 1))
        hm = torch.zeros(len(b['demo']), dtype=torch.bool)
        hm[held_idx] = True
        b['held'] = hm
        yield b



model.step = types.MethodType(step_scaled, model)
lam0 = float(model.cfg.lam_floor + F.softplus(model.lam_raw))
print(f'learned lambda {lam0:.2e}; scales {SCALES}', flush=True)

t0 = time.time()
qi, rows = 0, []
for b in held_batches():
    b = {k: v.to(DEV) for k, v in b.items()}
    demo, query, group, held = b['demo'], b['query'], b['group'], b['held']
    L = demo.shape[0]
    per = []
    with torch.no_grad():
        for mode in MODES:
            for s in SCALES:
                LAMS['scale'], LAMS['mode'] = torch.full((L,), s, device=DEV), mode
                lg, _ = run(b, None, grad=False)
                th = LAMS['theta']
                per.append((lane_ce(lg, b['labels']), lane_exact(lg, b['labels']),
                            (th != 0).flatten(1).float().mean(-1), th.flatten(1).norm(dim=-1)))
    gids = group[query].tolist()
    for gid in gids:
        t, p = items[qi]; qi += 1
        ql = ((group == gid) & query).nonzero()[0, 0]
        hl = ((group == gid) & held).nonzero()[0, 0]
        dl = (group == gid) & demo & ~held
        rows.append(dict(task=int(t), split=int(task_split[t]),
                         held_ce=[float(ce[hl]) for ce, ex, nz, nm in per], held_ex=[bool(ex[hl]) for ce, ex, nz, nm in per],
                         seen_ce=[float(ce[dl].mean()) for ce, ex, nz, nm in per],
                         q_ce=[float(ce[ql]) for ce, ex, nz, nm in per], q_ex=[bool(ex[ql]) for ce, ex, nz, nm in per],
                         q_nnz=[float(nz[ql]) for ce, ex, nz, nm in per], q_norm=[float(nm[ql]) for ce, ex, nz, nm in per]))
print(f'{len(rows)} queries in {time.time() - t0:.0f}s', flush=True)
NS = len(SCALES)
base = SCALES.index(1.0)


def report(sel, tag):
    R = [r for r in rows if sel(r)]
    print(f'== {tag}: {len(R)} pairs')
    for mi, mode in enumerate(MODES):
        o = mi * NS
        n0 = np.mean([r['q_norm'][o + base] for r in R])
        print(f'   [{mode}]')
        for k, sc in enumerate(SCALES):
            j = o + k
            print(f'     x{sc:<5g} query exact {sum(r["q_ex"][j] for r in R):3d} CE {np.mean([r["q_ce"][j] for r in R]):.3f} | '
                  f'held demo CE {np.mean([r["held_ce"][j] for r in R]):.3f} | trained demos CE {np.mean([r["seen_ce"][j] for r in R]):.3f} | '
                  f'Theta nonzero {np.mean([r["q_nnz"][j] for r in R]):.3f}, |Theta| {np.mean([r["q_norm"][j] for r in R]) / n0:.2f}x')
        pick = lambda r: o + int(np.argmin(r['held_ce'][o:o + NS]))
        def occ(r):
            ok = [k for k in range(NS) if r['held_ex'][o + k]]
            return o + (max(ok) if ok else base)
        print(f'     per task: default {sum(r["q_ex"][o + base] for r in R)} | by held demo CE {sum(r["q_ex"][pick(r)] for r in R)} | '
              f'sparsest keeping held demo exact {sum(r["q_ex"][occ(r)] for r in R)} | oracle {sum(max(r["q_ex"][o:o + NS]) for r in R)}', flush=True)


report(lambda r: True, 'all held-out tasks')
report(lambda r: r['split'] == 1, 'ARC-1 evaluation tasks')
json.dump(rows, open('/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/arc_lam2.json', 'w'))
