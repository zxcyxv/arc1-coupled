"""Reviewer checks: (1) colour-uniform row drift in the sparsemax write (invisible to argmax, visible to W_fb);
(2) sort-free sparsemax equals sort sparsemax and compiles fullgraph with static shapes (CPU inductor);
(3) full-support fraction along the deployed 8/30-step trajectory."""
import torch, json
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def sparsemax(z):
    zs, _ = z.sort(-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    cs = zs.cumsum(-1)
    ks = (1 + k * zs > cs).sum(-1, keepdim=True)
    return (z - (cs.gather(-1, ks - 1) - 1) / ks).clamp_min(0)

def sparsemax_nosort(z):
    # c is in the support iff 1 + k_c z_c > sum_{j: z_j >= z_c} z_j, with k_c = #{j: z_j >= z_c}; tau from the support
    ge = (z[..., None, :] >= z[..., :, None]).to(z.dtype)          # [.., c, j]: z_j >= z_c
    kc = ge.sum(-1); sc = (ge * z[..., None, :]).sum(-1)
    sup = (1 + kc * z > sc).to(z.dtype)
    tau = ((sup * z).sum(-1, keepdim=True) - 1) / sup.sum(-1, keepdim=True)
    return (z - tau).clamp_min(0)

out = {}
z = torch.randn(2000, 13) * 3
out['nosort_vs_sort_maxdiff'] = (sparsemax(z) - sparsemax_nosort(z)).abs().max().item()
f = torch.compile(sparsemax_nosort, fullgraph=True, dynamic=False)
g = torch.compile(sparsemax, fullgraph=True, dynamic=False)
z32 = z.float()
out['compiled_nosort_maxdiff'] = (f(z32).double() - sparsemax(z)).abs().max().item()
out['compiled_sort_maxdiff'] = (g(z32).double() - sparsemax(z)).abs().max().item()

# synthetic demo set: 13 colours (2 never targeted, like UNKNOWN/PAD), p=64, n=240 cells, unit codes clustered by colour role
B, n, p, C = 64, 240, 64, 13
proto = torch.randn(B, 11 * 6, p)
lab = torch.arange(2, 13).repeat_interleave(6)
idx = torch.randint(0, 66, (B, n))
x = proto.gather(1, idx[..., None].expand(B, n, p)) + 0.05 * torch.randn(B, n, p)
phi = x / x.norm(dim=-1, keepdim=True)
Y = torch.nn.functional.one_hot(lab[idx], C).double()
G = phi.mT @ phi / n; S = Y.mT @ phi / n
th_lin = torch.zeros(B, C, p); th_sm = torch.zeros(B, C, p)
res = {}
for t in range(1, 31):
    th_lin = th_lin - (th_lin @ G - S)
    zsm = phi @ th_sm.mT
    P = sparsemax(zsm)
    full = (P > 0).all(-1).double().mean().item()
    th_sm = th_sm + ((Y - P).mT @ phi) / n
    if t in (1, 4, 8, 16, 30):
        D = th_sm - th_lin                                       # should be a colour-uniform row while full support holds
        u = D.mean(1, keepdim=True)
        nonuni = (D - u).flatten(1).norm(dim=-1) / th_lin.flatten(1).norm(dim=-1)
        r_lin = phi @ th_lin.mT; r_sm = phi @ th_sm.mT
        shift = (r_sm - r_lin).mean(-1)                          # per-cell colour-uniform offset of r
        res[t] = dict(full_support_frac_before_step=round(full, 4),
                      nonuniform_part_rel=float(nonuni.median()),
                      uniform_offset_of_r_rms=float(shift.pow(2).mean().sqrt()),
                      r_lin_rms=float(r_lin.pow(2).mean().sqrt()),
                      argmax_agree=float(((r_sm.argmax(-1) == r_lin.argmax(-1)).double().mean())))
out['uniform_row_drift'] = res
print(json.dumps(out, indent=1))
