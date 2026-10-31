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
LAMS = {'scale': None}


def step_scaled(self, theta, G, S_, gam=None, ext=None):
    """Coupled.step with the lasso threshold multiplied per lane by LAMS['scale'] [L]."""
    lam = (self.cfg.lam_floor + F.softplus(self.lam_raw)) * LAMS['scale']
    g = theta @ G - S_
    if gam is not None:
        g = g + (gam[..., None, None] * ext[0]).sum(1)
    theta = soft(theta - g, lam.view(-1, 1, 1))
    if gam is None:
        return theta
    Ck, H_, sv = ext
    gg = (theta[:, None] * Ck).sum((-2, -1)) + (H_ * gam[:, None]).sum(-1) - sv
    return theta, soft(gam - gg, lam.view(-1, 1))


def held_batches(v=0):
    for i in range(0, len(items), NG):
        groups, held_idx, off = [], [], 0
        for t, p in items[i:i + NG]:
            demos = list(data.train_pairs[t][:data.D - 1]) + [data.held[t]]
            groups.append(data.group(t, demos + [p], v, None, [True] * len(demos) + [False], [False] * len(demos) + [True]))
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

V = 8
t0 = time.time()
held = {}            # (version, query index) -> list over scales of (held CE, held exact, query CE, query exact)
for v in range(V):
    qi = 0
    for b in held_batches(v):
        b = {k: x.to(DEV) for k, x in b.items()}
        demo, query, group, hmask = b['demo'], b['query'], b['group'], b['held']
        L = demo.shape[0]
        per = []
        with torch.no_grad():
            for sc in SCALES:
                LAMS['scale'] = torch.full((L,), sc, device=DEV)
                lg, _ = run(b, None, grad=False)
                per.append((lane_ce(lg, b['labels']), lane_exact(lg, b['labels'])))
        for gid in group[query].tolist():
            ql = ((group == gid) & query).nonzero()[0, 0]
            hl = ((group == gid) & hmask).nonzero()[0, 0]
            held[(v, qi)] = [(float(ce[hl]), bool(ex[hl]), float(ce[ql]), bool(ex[ql])) for ce, ex in per]
            qi += 1
    print(f'version {v} done {time.time() - t0:.0f}s', flush=True)
NQ = len(items)
base = SCALES.index(1.0)
rows = []
for q in range(NQ):
    t, p = items[q]
    r = dict(split=int(task_split[t]))
    r['q_ex0'] = [held[(0, q)][k][3] for k in range(len(SCALES))]           # query in version 0 (no augmentation)
    r['q_ce0'] = [held[(0, q)][k][2] for k in range(len(SCALES))]
    r['held_ce0'] = [held[(0, q)][k][0] for k in range(len(SCALES))]
    r['held_ce_avg'] = [float(np.mean([held[(v, q)][k][0] for v in range(V)])) for k in range(len(SCALES))]
    r['held_ex_avg'] = [float(np.mean([held[(v, q)][k][1] for v in range(V)])) for k in range(len(SCALES))]
    r['q_ex_avg'] = [float(np.mean([held[(v, q)][k][3] for v in range(V)])) for k in range(len(SCALES))]
    rows.append(r)


def rep(sel, tag):
    R = [r for r in rows if sel(r)]
    pick = lambda r, key: int(np.argmin(r[key]))
    print(f'== {tag}: {len(R)} queries (version 0 scored)')
    print(f'   default x1 {sum(r["q_ex0"][base] for r in R)} | by held demo CE, version 0 only {sum(r["q_ex0"][pick(r, "held_ce0")] for r in R)}'
          f' | by held demo CE averaged over {V} augmented versions {sum(r["q_ex0"][pick(r, "held_ce_avg")] for r in R)}'
          f' | by held demo exact rate over versions (then CE) {sum(r["q_ex0"][max(range(len(SCALES)), key=lambda k: (r["held_ex_avg"][k], -r["held_ce_avg"][k]))] for r in R)}'
          f' | oracle {sum(max(r["q_ex0"]) for r in R)}')
    hc = np.array([r['held_ex_avg'][base] for r in R]); qx = np.array([r['q_ex_avg'][base] for r in R])
    print(f'   held-demo exact rate over versions vs query exact rate over versions (learned lambda): corr {np.corrcoef(hc, qx)[0, 1]:+.3f}; '
          f'query exact rate where held rate >= 0.5: {qx[hc >= 0.5].mean():.2f} (n {int((hc >= 0.5).sum())}), < 0.5: {qx[hc < 0.5].mean():.2f}', flush=True)


rep(lambda r: True, 'all held-out tasks')
rep(lambda r: r['split'] == 1, 'ARC-1 evaluation tasks')
json.dump(rows, open('/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/arc_lam_aug.json', 'w'))
