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


t0 = time.time()
qi = 0
rows = []                                                    # per query: split, base exact/ce, and per variant/frac
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


for b in held_batches():
    b = {k: v.to(DEV) for k, v in b.items()}
    demo, query, group = b['demo'], b['query'], b['group']
    L = demo.shape[0]
    Zleaf = torch.zeros(L, c.heads, c.d // c.heads, c.d // c.heads, device=DEV, requires_grad=True)
    logits, wnorm = run(b, Zleaf, grad=True)
    ce = lane_ce(logits, b['labels'])
    held = b['held']
    L_demo, L_q = (ce * held).sum(), (ce * query).sum()          # reward: the never-trained demo's LOO loss
    g_demo, = torch.autograd.grad(L_demo, Zleaf, retain_graph=True)
    g_q, = torch.autograd.grad(L_q, Zleaf)
    with torch.no_grad():
        base_ex = lane_exact(logits, b['labels'])
        base_ce = ce.detach()
        demo_ce0 = base_ce
        # per group: cos over its demo lanes; normalised directions per group (demo lanes) and per query lane
        gids = group[query].tolist()
        dirs = {}
        for name, g, mask in (('demo_legit', g_demo, demo), ('demo_oracle', g_q, demo), ('query_oracle', g_q, query)):
            d = torch.zeros_like(g)
            for gid in gids:
                sel = (group == gid) & mask
                gg = g[sel]
                nrm = gg.flatten().norm().clamp_min(1e-30)
                # step size: frac x the RMS |W| of the selected lanes, spread over them in the gradient's shape
                scale = wnorm[sel].pow(2).mean().sqrt() * (sel.sum().float()).sqrt()
                d[sel] = -gg / nrm * scale
            dirs[name] = d
        cosv = {}
        for gid in gids:
            sel = (group == gid) & demo
            a, q_ = g_demo[sel].flatten().double(), g_q[sel].flatten().double()
            cosv[gid] = float((a @ q_) / (a.norm() * q_.norm()).clamp_min(1e-30))
        gq_on_query = {gid: float(g_demo[(group == gid) & query].abs().max()) for gid in gids}
        res = {name: [] for name in dirs}
        for name, d in dirs.items():
            for frac in FRACS:
                lg, _ = run(b, frac * d, grad=False)
                ce2 = lane_ce(lg, b['labels'])
                res[name].append((lane_exact(lg, b['labels']), ce2))
    for gid in gids:
        t, p = items[qi]
        qi += 1
        ql = ((group == gid) & query).nonzero()[0, 0]
        dl = ((group == gid) & demo)
        hl = ((group == gid) & held)
        row = dict(task=int(t), split=int(task_split[t]), base_ex=bool(base_ex[ql]), base_ce=float(base_ce[ql]),
                   demo_ce=float(demo_ce0[hl].mean()), seen_demo_ce=float(demo_ce0[dl & ~held].mean()), cos=cosv[gid], gdemo_on_query=gq_on_query[gid])
        for name in dirs:
            row[name] = [(bool(ex[ql]), float(ce2[ql]), float(ce2[hl].mean())) for ex, ce2 in res[name]]
        rows.append(row)
print(f'{len(rows)} queries in {time.time() - t0:.0f}s', flush=True)


def summary(sel, tag):
    R = [r for r in rows if sel(r)]
    n = len(R)
    print(f'== {tag}: {n} pairs, base exact {sum(r["base_ex"] for r in R)}, query CE {np.mean([r["base_ce"] for r in R]):.3f}, '
          f'held-out demo LOO CE {np.mean([r["demo_ce"] for r in R]):.3f} (trained demos {np.mean([r["seen_demo_ce"] for r in R]):.3f})')
    cs = np.array([r['cos'] for r in R])
    print(f'   cos(grad demo-LOO loss, grad query loss) over demo lanes: median {np.median(cs):+.3f}, mean {cs.mean():+.3f}, '
          f'frac>0 {np.mean(cs > 0):.2f}; max |dL_demo/dZ| on the query lane {max(r["gdemo_on_query"] for r in R):.1e}')
    for name in ('demo_legit', 'demo_oracle', 'query_oracle'):
        parts = []
        for k, frac in enumerate(FRACS):
            ex = sum(r[name][k][0] for r in R) - sum(r['base_ex'] for r in R)
            dce = np.mean([r[name][k][1] - r['base_ce'] for r in R])
            dd = np.mean([r[name][k][2] - r['demo_ce'] for r in R])
            parts.append(f'{frac:g}: dex {ex:+d} dCEq {dce:+.3f} dCEdemo {dd:+.3f}')
        print(f'   {name:12s} ' + ' | '.join(parts), flush=True)


summary(lambda r: True, 'all held-out tasks')
summary(lambda r: r['split'] == 1, 'ARC-1 evaluation tasks')
summary(lambda r: r['split'] == 0, 'training tasks')
json.dump(rows, open(f'/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/arc_gate2_zb{ZB}.json', 'w'))
