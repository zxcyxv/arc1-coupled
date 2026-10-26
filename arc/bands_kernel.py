"""The bands trace step of arc.model.Coupled.bands_step as one Triton kernel each way (CUDA, FP32 arithmetic).

Per head h, band r, token t and channel d, with z_r = zr + i zi, gamma_r, a_r, b_r from Block.bands():
    E_K,r <- z_r E_K,r + gamma_r k~;   D_V,r = z_r E_V,r;   E_V,r <- D_V,r + gamma_r v
    k_mix = sum_r Re(a_r E_K,r) (new E_K),   v_mix = sum_r Re(b_r D_V,r)
The same formulas as the PyTorch path; only the memory traffic changes. The traces E_K, E_V [B,H,T,2RD] ((Re, Im) x R
x D on the last axis) are read once and written once per block, k_mix and v_mix come out of the same pass, and the
backward pass stores nothing of its own: the update is linear, so its gradients are written from the old traces and
the inputs, which the chain keeps anyway. The parameter gradients [7,H,R] are summed per tile into a buffer and reduced
afterwards (deterministic). The new traces are stored in `store` (FP32, BF16 or FP16); every update and gradient is
FP32 arithmetic.
"""
import torch
import triton
import triton.language as tl

BT = 32                                                   # tokens per program


@triton.jit
def _bands_fwd(KR, V, EK, EV, P, EKO, EVO, KMIX, VMIX, T, H, R: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    pid_t, pid_bh = tl.program_id(0), tl.program_id(1)
    h = pid_bh % H
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    mask = (t < T)[:, None]
    row = pid_bh * T + t
    xo = row[:, None] * D + d[None, :]
    kr = tl.load(KR + xo, mask=mask, other=0.0)
    v = tl.load(V + xo, mask=mask, other=0.0)
    kmix = tl.zeros([BT, D], dtype=tl.float32)
    vmix = tl.zeros([BT, D], dtype=tl.float32)
    so = row[:, None] * (2 * R * D) + d[None, :]
    for r in tl.static_range(R):
        pr = h * R + r
        zr = tl.load(P + pr)
        zi = tl.load(P + H * R + pr)
        g = tl.load(P + 2 * H * R + pr)
        ar = tl.load(P + 3 * H * R + pr)
        ai = tl.load(P + 4 * H * R + pr)
        br = tl.load(P + 5 * H * R + pr)
        bi = tl.load(P + 6 * H * R + pr)
        o_re = so + r * D
        o_im = o_re + R * D
        ekr = tl.load(EK + o_re, mask=mask, other=0.0).to(tl.float32)
        eki = tl.load(EK + o_im, mask=mask, other=0.0).to(tl.float32)
        evr = tl.load(EV + o_re, mask=mask, other=0.0).to(tl.float32)
        evi = tl.load(EV + o_im, mask=mask, other=0.0).to(tl.float32)
        nkr = zr * ekr - zi * eki + g * kr
        nki = zr * eki + zi * ekr
        dvr = zr * evr - zi * evi
        dvi = zr * evi + zi * evr
        kmix += ar * nkr - ai * nki
        vmix += br * dvr - bi * dvi
        tl.store(EKO + o_re, nkr.to(EKO.dtype.element_ty), mask=mask)
        tl.store(EKO + o_im, nki.to(EKO.dtype.element_ty), mask=mask)
        tl.store(EVO + o_re, (dvr + g * v).to(EVO.dtype.element_ty), mask=mask)
        tl.store(EVO + o_im, dvi.to(EVO.dtype.element_ty), mask=mask)
    tl.store(KMIX + xo, kmix, mask=mask)
    tl.store(VMIX + xo, vmix, mask=mask)


@triton.jit
def _bands_bwd(KR, V, EK, EV, P, GEKO, GEVO, GKMIX, GVMIX, GKR0, GV0, GEK, GEV, GKR, GV, GP, T, H,
         R: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, HAS_GEK: tl.constexpr, HAS_GEV: tl.constexpr,
         HAS_G0: tl.constexpr):
    pid_t, pid_bh = tl.program_id(0), tl.program_id(1)
    h = pid_bh % H
    t = pid_t * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    mask = (t < T)[:, None]
    row = pid_bh * T + t
    xo = row[:, None] * D + d[None, :]
    kr = tl.load(KR + xo, mask=mask, other=0.0)
    v = tl.load(V + xo, mask=mask, other=0.0)
    gkm = tl.load(GKMIX + xo, mask=mask, other=0.0)
    gvm = tl.load(GVMIX + xo, mask=mask, other=0.0)
    if HAS_G0:                                            # other gradients of k~ and v to add (arc/sums_kernel.py)
        gkr = tl.load(GKR0 + xo, mask=mask, other=0.0)
        gv = tl.load(GV0 + xo, mask=mask, other=0.0)
    else:
        gkr = tl.zeros([BT, D], dtype=tl.float32)
        gv = tl.zeros([BT, D], dtype=tl.float32)
    so = row[:, None] * (2 * R * D) + d[None, :]
    gp = GP + (pid_t * tl.num_programs(1) + pid_bh) // H * (7 * H * R)     # this tile's [7,H,R] partial sums
    for r in tl.static_range(R):
        pr = h * R + r
        zr = tl.load(P + pr)
        zi = tl.load(P + H * R + pr)
        g = tl.load(P + 2 * H * R + pr)
        ar = tl.load(P + 3 * H * R + pr)
        ai = tl.load(P + 4 * H * R + pr)
        br = tl.load(P + 5 * H * R + pr)
        bi = tl.load(P + 6 * H * R + pr)
        o_re = so + r * D
        o_im = o_re + R * D
        ekr = tl.load(EK + o_re, mask=mask, other=0.0).to(tl.float32)
        eki = tl.load(EK + o_im, mask=mask, other=0.0).to(tl.float32)
        evr = tl.load(EV + o_re, mask=mask, other=0.0).to(tl.float32)
        evi = tl.load(EV + o_im, mask=mask, other=0.0).to(tl.float32)
        Gr = ar * gkm                                     # d/d new E_K (Re, Im): the carried gradient + k_mix's
        Gi = -ai * gkm
        if HAS_GEK:
            Gr += tl.load(GEKO + o_re, mask=mask, other=0.0).to(tl.float32)
            Gi += tl.load(GEKO + o_im, mask=mask, other=0.0).to(tl.float32)
        ger = tl.zeros([BT, D], dtype=tl.float32)         # d/d new E_V
        gei = tl.zeros([BT, D], dtype=tl.float32)
        if HAS_GEV:
            ger = tl.load(GEVO + o_re, mask=mask, other=0.0).to(tl.float32)
            gei = tl.load(GEVO + o_im, mask=mask, other=0.0).to(tl.float32)
        Gdr = ger + br * gvm                              # d/d D_V
        Gdi = gei - bi * gvm
        tl.store(GEK + o_re, (zr * Gr + zi * Gi).to(GEK.dtype.element_ty), mask=mask)
        tl.store(GEK + o_im, (zr * Gi - zi * Gr).to(GEK.dtype.element_ty), mask=mask)
        tl.store(GEV + o_re, (zr * Gdr + zi * Gdi).to(GEV.dtype.element_ty), mask=mask)
        tl.store(GEV + o_im, (zr * Gdi - zi * Gdr).to(GEV.dtype.element_ty), mask=mask)
        gkr += g * Gr
        gv += g * ger
        nkr = zr * ekr - zi * eki + g * kr
        nki = zr * eki + zi * ekr
        dvr = zr * evr - zi * evi
        dvi = zr * evi + zi * evr
        tl.store(gp + pr, tl.sum(tl.sum(Gr * ekr + Gi * eki + Gdr * evr + Gdi * evi, 1), 0))
        tl.store(gp + H * R + pr, tl.sum(tl.sum(Gi * ekr - Gr * eki + Gdi * evr - Gdr * evi, 1), 0))
        tl.store(gp + 2 * H * R + pr, tl.sum(tl.sum(Gr * kr + ger * v, 1), 0))
        tl.store(gp + 3 * H * R + pr, tl.sum(tl.sum(gkm * nkr, 1), 0))
        tl.store(gp + 4 * H * R + pr, tl.sum(tl.sum(-gkm * nki, 1), 0))
        tl.store(gp + 5 * H * R + pr, tl.sum(tl.sum(gvm * dvr, 1), 0))
        tl.store(gp + 6 * H * R + pr, tl.sum(tl.sum(-gvm * dvi, 1), 0))
    tl.store(GKR + xo, gkr, mask=mask)
    tl.store(GV + xo, gv, mask=mask)


def _new(kr, R, store):
    B, H, T, D = kr.shape
    return kr.new_empty(B, H, T, 2 * R * D, dtype=store)


@torch.library.custom_op('arc::bands_trace', mutates_args=(), device_types='cuda')
def bands_trace(kr: torch.Tensor, v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor, P: torch.Tensor,
                store: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """kr, v [B,H,T,D] FP32 (k~ and v), ek, ev [B,H,T,2RD] (FP32, BF16 or FP16), P [7,H,R] FP32 (Block.bands()
    stacked) -> new ek, ev in `store`, k_mix, v_mix [B,H,T,D] FP32."""
    B, H, T, D = kr.shape
    R = P.shape[-1]
    kr, v, ek, ev, P = (x.contiguous() for x in (kr, v, ek, ev, P))
    eko, evo = _new(kr, R, store), _new(kr, R, store)
    kmix, vmix = kr.new_empty(B, H, T, D), kr.new_empty(B, H, T, D)
    _bands_fwd[(triton.cdiv(T, BT), B * H)](kr, v, ek, ev, P, eko, evo, kmix, vmix, T, H, R=R, D=D, BT=BT)
    return eko, evo, kmix, vmix


@bands_trace.register_fake
def _(kr, v, ek, ev, P, store):
    B, H, T, D = kr.shape
    R = P.shape[-1]
    return _new(kr, R, store), _new(kr, R, store), kr.new_empty(B, H, T, D), kr.new_empty(B, H, T, D)


def trace_bwd(kr, v, ek, ev, P, geko, gevo, gkmix, gvmix, gkr0=None, gv0=None, inplace=False):
    """The trace gradient kernel. gkr0, gv0 (optional): other gradients of k~ and v that the kernel adds to its own;
    inplace (with gkr0, gv0 contiguous FP32 and not needed afterwards): the sums are written over them (no new
    buffers). Returns g_kr, g_v, g_ek, g_ev, g_P."""
    B, H, T, D = kr.shape
    R = P.shape[-1]
    kr, v, ek, ev, P = (x.contiguous() for x in (kr, v, ek, ev, P))
    gkmix = torch.zeros_like(kr) if gkmix is None else gkmix.contiguous()
    gvmix = torch.zeros_like(kr) if gvmix is None else gvmix.contiguous()
    geko = None if geko is None else geko.contiguous()
    gevo = None if gevo is None else gevo.contiguous()
    gkr0 = None if gkr0 is None else gkr0.contiguous()
    gv0 = None if gv0 is None else gv0.contiguous()
    gek, gev = torch.empty_like(ek), torch.empty_like(ev)
    if inplace and gkr0 is not None:
        gkr, gv = gkr0, gv0                                # each program reads its tile before writing it
    else:
        gkr, gv = torch.empty_like(kr), torch.empty_like(kr)
    nt = triton.cdiv(T, BT)
    part = P.new_empty(nt * B, 7, H, R)
    _bands_bwd[(nt, B * H)](kr, v, ek, ev, P, ek if geko is None else geko, ev if gevo is None else gevo, gkmix, gvmix,
                            kr if gkr0 is None else gkr0, kr if gv0 is None else gv0, gek, gev, gkr, gv, part, T, H,
                            R=R, D=D, BT=BT, HAS_GEK=geko is not None, HAS_GEV=gevo is not None,
                            HAS_G0=gkr0 is not None, num_warps=8)
    return gkr, gv, gek, gev, part.sum(0)


@torch.library.custom_op('arc::bands_trace_bwd', mutates_args=(), device_types='cuda')
def bands_trace_bwd(kr: torch.Tensor, v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor, P: torch.Tensor,
                    geko: torch.Tensor | None, gevo: torch.Tensor | None, gkmix: torch.Tensor | None,
                    gvmix: torch.Tensor | None, gkr0: torch.Tensor | None = None,
                    gv0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                              torch.Tensor]:
    return trace_bwd(kr, v, ek, ev, P, geko, gevo, gkmix, gvmix, gkr0, gv0)


@bands_trace_bwd.register_fake
def _(kr, v, ek, ev, P, geko, gevo, gkmix, gvmix, gkr0=None, gv0=None):
    B, H, T, D = kr.shape
    return (kr.new_empty(B, H, T, D), kr.new_empty(B, H, T, D), ek.new_empty(ek.shape), ev.new_empty(ev.shape),
            P.new_empty(P.shape))


def _setup(ctx, inputs, output):
    kr, v, ek, ev, P, store = inputs
    ctx.save_for_backward(kr, v, ek, ev, P)


def _backward(ctx, geko, gevo, gkmix, gvmix):
    gkr, gv, gek, gev, gP = bands_trace_bwd(*ctx.saved_tensors, geko, gevo, gkmix, gvmix)
    return gkr, gv, gek, gev, gP, None


bands_trace.register_autograd(_backward, setup_context=_setup)
