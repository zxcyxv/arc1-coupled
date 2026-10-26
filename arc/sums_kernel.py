"""The token sums of arc.model.Coupled.bands_step (S, W, the read) on tensor cores in 3xTF32 (CUDA, FP32 in and out).

Per lane b and head h, over the T tokens (k~ = kr, q~ = qr, all [T,D]):
    S = v^T k_mix + v_mix^T k~;   W' = (1 - eta) W + eta S;   m = v^T k~ + W';   read = q~ m^T
Every product is tl.dot(..., input_precision='tf32x3'): each FP32 operand is split into a TF32 high part and a TF32
remainder and three tensor-core products are summed (hi hi + hi lo + lo hi) with FP32 accumulation, which is FP32-level
accuracy (the dropped lo lo term is below FP32 rounding); single-pass TF32 and BF16 are never used. The token sums
(K = T, outputs D x D per lane and head) are split over NS token ranges per (b, h) whose partial sums are added in a
fixed order afterwards (deterministic; no atomics).

With the trace (TRACE, BandsFused) the forward kernel is arc/bands_kernel.py's trace update with the sums added on the
same token tiles (k_mix and v_mix are written only for the backward pass); without it (BandsRead) the kernel reads
k_mix and v_mix. The read is a second pass (it needs the whole sums). The backward pass is three kernels: g_m and g_q~
(one pass over G and q~), the small D x D gradients, and the token-tile products g_k_mix = v g_S, g_v_mix = k~ g_S^T,
v g_m + v_mix g_S, k~ g_m^T + k_mix g_S^T, which arc/bands_kernel.py's trace gradient kernel adds to its own in place.
(Fusing those products into the trace gradient kernel was 2-3x slower: the MMA layout of the products spread to the
band loop, and the 64 x 64 operands exhausted the registers.) Fixed launch configurations, no autotuning, no host sync.

Backward, with G the gradient of the read and g_W' that of W' (if any):
    g_m = G^T q~;  g_q~ = G m;  g_W'tot = g_m + g_W';  g_S = eta g_W'tot;  g_W = (1 - eta) g_W'tot;
    g_eta = sum (S - W) . g_W'tot
The read is returned as a [B,H,T,D] view of [B,T,H,D] memory (the layout the block's output projection reads), and its
gradient may come in any layout with unit channel stride.
"""
import torch
import triton
import triton.language as tl

from .bands_kernel import trace_bwd

PREC = 'tf32x3'
# per kernel: tokens per tile, token ranges per (b, h), warps, pipeline stages
BT_FWD, NS_FWD, WARPS_FWD, STAGES_FWD = 32, 4, 4, 2      # forward trace + sums
BT_READ, WARPS_READ = 64, 4                               # the read
BT_GM, NS_GM, WARPS_GM, STAGES_GM = 64, 4, 4, 2           # backward g_m sums and g_q~
BT_BWD, NS_BWD, WARPS_BWD, STAGES_BWD = 64, 2, 4, 2       # backward sums gradients (two programs per range)


def _bd(D):
    return max(16, triton.next_power_of_2(D))


PROGRAMS = 1536                                           # at least this many programs (fewer lanes: more ranges)


def _tps(T, ns, bt, nbh, per_range=1):
    """Tokens per range (a multiple of the tile): ns ranges per (b, h), more if nbh * ns * per_range < PROGRAMS."""
    ns = max(ns, triton.cdiv(PROGRAMS, nbh * per_range))
    return triton.cdiv(triton.cdiv(T, ns), bt) * bt


@triton.jit
def _fwd(KR, V, EK, EV, P, EKO, EVO, KMIX, VMIX, PART, T, H, TPS, NBH,
         R: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, TRACE: tl.constexpr,
         STORE_MIX: tl.constexpr, PREC: tl.constexpr):
    pid_s, bh = tl.program_id(0), tl.program_id(1)
    h = bh % H
    d = tl.arange(0, BD)
    dm = d < D
    acc_s = tl.zeros([BD, BD], dtype=tl.float32)
    acc_p = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, TPS, BT):
        t = pid_s * TPS + i + tl.arange(0, BT)
        mask = (t < T)[:, None] & dm[None, :]
        row = bh * T + t
        xo = row[:, None] * D + d[None, :]
        kr = tl.load(KR + xo, mask=mask, other=0.0)
        v = tl.load(V + xo, mask=mask, other=0.0)
        if TRACE:
            kmix = tl.zeros([BT, BD], dtype=tl.float32)
            vmix = tl.zeros([BT, BD], dtype=tl.float32)
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
            if STORE_MIX:                                         # kept for the backward pass
                tl.store(KMIX + xo, kmix, mask=mask)
                tl.store(VMIX + xo, vmix, mask=mask)
        else:
            kmix = tl.load(KMIX + xo, mask=mask, other=0.0)
            vmix = tl.load(VMIX + xo, mask=mask, other=0.0)
        vt = tl.trans(v)
        acc_s = tl.dot(vt, kmix, acc_s, input_precision=PREC)
        acc_s = tl.dot(tl.trans(vmix), kr, acc_s, input_precision=PREC)
        acc_p = tl.dot(vt, kr, acc_p, input_precision=PREC)
    mm = dm[:, None] & dm[None, :]
    po = ((pid_s * NBH + bh) * 2 * D + d[:, None]) * D + d[None, :]          # PART [NS, BH, 2, D, D]
    tl.store(PART + po, acc_s, mask=mm)
    tl.store(PART + po + D * D, acc_p, mask=mm)


@triton.jit
def _fwd_finish(PART, W, ETA, S, WO, M, H, NS, NBH, D: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    d = tl.arange(0, BD)
    mm = (d < D)[:, None] & (d < D)[None, :]
    oo = d[:, None] * D + d[None, :]
    s = tl.zeros([BD, BD], dtype=tl.float32)
    p = tl.zeros([BD, BD], dtype=tl.float32)
    for j in range(NS):
        po = (j * NBH + bh) * 2 * D * D + oo
        s += tl.load(PART + po, mask=mm, other=0.0)
        p += tl.load(PART + po + D * D, mask=mm, other=0.0)
    eta = tl.load(ETA + bh % H)
    w = tl.load(W + bh * D * D + oo, mask=mm, other=0.0)
    wn = (1 - eta) * w + eta * s
    tl.store(S + bh * D * D + oo, s, mask=mm)
    tl.store(WO + bh * D * D + oo, wn, mask=mm)
    tl.store(M + bh * D * D + oo, p + wn, mask=mm)


@triton.jit
def _read(QR, M, OUT, T, H, sob, soh, sot, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, PREC: tl.constexpr):
    pid_t, bh = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    dm = d < D
    t = pid_t * BT + tl.arange(0, BT)
    mask = (t < T)[:, None] & dm[None, :]
    m = tl.load(M + bh * D * D + d[:, None] * D + d[None, :], mask=dm[:, None] & dm[None, :], other=0.0)
    q = tl.load(QR + (bh * T + t)[:, None] * D + d[None, :], mask=mask, other=0.0)
    out = tl.dot(q, tl.trans(m), input_precision=PREC)                 # read = q~ m^T
    oo = (bh // H) * sob + (bh % H) * soh + t[:, None] * sot + d[None, :]
    tl.store(OUT + oo, out, mask=mask)


@triton.jit
def _bwd_gm(G, QR, M, GQR, PART, T, H, TPS, NBH, sgb, sgh, sgt,
            D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, PREC: tl.constexpr):
    pid_s, bh = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    dm = d < D
    mm = dm[:, None] & dm[None, :]
    m = tl.load(M + bh * D * D + d[:, None] * D + d[None, :], mask=mm, other=0.0)
    acc = tl.zeros([BD, BD], dtype=tl.float32)
    go_bh = (bh // H) * sgb + (bh % H) * sgh
    for i in range(0, TPS, BT):
        t = pid_s * TPS + i + tl.arange(0, BT)
        mask = (t < T)[:, None] & dm[None, :]
        g = tl.load(G + go_bh + t[:, None] * sgt + d[None, :], mask=mask, other=0.0)
        xo = (bh * T + t)[:, None] * D + d[None, :]
        q = tl.load(QR + xo, mask=mask, other=0.0)
        acc = tl.dot(tl.trans(g), q, acc, input_precision=PREC)       # g_m = G^T q~
        tl.store(GQR + xo, tl.dot(g, m, input_precision=PREC), mask=mask)     # g_q~ = G m
    po = (pid_s * NBH + bh) * D * D + d[:, None] * D + d[None, :]
    tl.store(PART + po, acc, mask=mm)


@triton.jit
def _bwd_finish(PART, GWO, S, W, ETA, GM, GS, GW, GETA, H, NS, NBH, HAS_GW: tl.constexpr,
                D: tl.constexpr, BD: tl.constexpr):
    bh = tl.program_id(0)
    d = tl.arange(0, BD)
    mm = (d < D)[:, None] & (d < D)[None, :]
    oo = bh * D * D + d[:, None] * D + d[None, :]
    gm = tl.zeros([BD, BD], dtype=tl.float32)
    for j in range(NS):
        gm += tl.load(PART + j * NBH * D * D + oo, mask=mm, other=0.0)
    gw = gm
    if HAS_GW:
        gw = gm + tl.load(GWO + oo, mask=mm, other=0.0)
    eta = tl.load(ETA + bh % H)
    s = tl.load(S + oo, mask=mm, other=0.0)
    w = tl.load(W + oo, mask=mm, other=0.0)
    tl.store(GM + oo, gm, mask=mm)
    tl.store(GS + oo, eta * gw, mask=mm)
    tl.store(GW + oo, (1 - eta) * gw, mask=mm)
    tl.store(GETA + bh, tl.sum(tl.sum((s - w) * gw, 1), 0))


@triton.jit
def _bwd_dots(KR, V, KMIX, VMIX, GM, GS, GKMIX, GVMIX, GKR, GV, T, TPS, NS,
              D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, PREC: tl.constexpr):
    """Programs 0..NS-1 (token ranges): g_k_mix = v g_S, g_k~ = v g_m + v_mix g_S; programs NS..2NS-1: g_v_mix = k~ g_S^T,
    g_v = k~ g_m^T + k_mix g_S^T (two matrices per program keep the operands in registers)."""
    pid, bh = tl.program_id(0), tl.program_id(1)
    second = pid >= NS
    pid_s = pid - NS * second.to(tl.int32)
    d = tl.arange(0, BD)
    dm = d < D
    mm = dm[:, None] & dm[None, :]
    oo = bh * D * D + d[:, None] * D + d[None, :]
    ot = bh * D * D + d[None, :] * D + d[:, None]                         # transposed
    if second:
        gm = tl.load(GM + ot, mask=mm, other=0.0)
        gs = tl.load(GS + ot, mask=mm, other=0.0)
        X, XMIX, GXMIX, GX = KR, KMIX, GVMIX, GV
    else:
        gm = tl.load(GM + oo, mask=mm, other=0.0)
        gs = tl.load(GS + oo, mask=mm, other=0.0)
        X, XMIX, GXMIX, GX = V, VMIX, GKMIX, GKR
    for i in range(0, TPS, BT):
        t = pid_s * TPS + i + tl.arange(0, BT)
        mask = (t < T)[:, None] & dm[None, :]
        xo = (bh * T + t)[:, None] * D + d[None, :]
        x = tl.load(X + xo, mask=mask, other=0.0)
        tl.store(GXMIX + xo, tl.dot(x, gs, input_precision=PREC), mask=mask)
        y = tl.load(XMIX + xo, mask=mask, other=0.0)                       # v_mix (first) or k_mix (second)
        tl.store(GX + xo, tl.dot(y, gs, tl.dot(x, gm, input_precision=PREC), input_precision=PREC), mask=mask)


# ---------------------------------------------------------------------------------------------------- launchers
def _read_out(qr):
    B, H, T, D = qr.shape
    return qr.new_empty(B, T, H, D).transpose(1, 2)                   # [B,H,T,D] view of [B,T,H,D]


def _forward(qr, kr, v, ek, ev, P, kmix, vmix, w, eta, store, save_mix=False):
    B, H, T, D = kr.shape
    BD, NBH = _bd(D), B * H
    trace = ek is not None
    R = P.shape[-1] if trace else 1
    qr, kr, v, w = (x.contiguous() for x in (qr, kr, v, w))
    eta = eta.reshape(H).contiguous()
    if trace:
        ek, ev, P = ek.contiguous(), ev.contiguous(), P.contiguous()
        eko, evo = (kr.new_empty(B, H, T, 2 * R * D, dtype=store) for _ in range(2))
        kmix, vmix = (torch.empty_like(kr) if save_mix else kr.new_empty(0) for _ in range(2))
    else:
        kmix, vmix = kmix.contiguous(), vmix.contiguous()
        eko = evo = None
    tps = _tps(T, NS_FWD, BT_FWD, NBH)
    ns = triton.cdiv(T, tps)
    part = kr.new_empty(ns, NBH, 2, D, D)
    _fwd[(ns, NBH)](kr, v, ek, ev, P, eko, evo, kmix, vmix, part, T, H, tps, NBH, R=R, D=D, BD=BD, BT=BT_FWD,
                    TRACE=trace, STORE_MIX=save_mix, PREC=PREC, num_warps=WARPS_FWD, num_stages=STAGES_FWD)
    s, wn, m = (torch.empty_like(w) for _ in range(3))
    _fwd_finish[(NBH,)](part, w, eta, s, wn, m, H, ns, NBH, D=D, BD=BD)
    read = _read_out(qr)
    _read[(triton.cdiv(T, BT_READ), NBH)](qr, m, read, T, H, *read.stride()[:3], D=D, BD=BD, BT=BT_READ, PREC=PREC,
                                     num_warps=WARPS_READ)
    return read, eko, evo, wn, s, m, kmix, vmix


def _backward(g_read, g_w, g_ek, g_ev, qr, kr, v, ek, ev, P, kmix, vmix, w, eta, s, m):
    B, H, T, D = kr.shape
    BD, NBH = _bd(D), B * H
    trace = ek is not None
    if g_read is None:
        g_read = torch.zeros_like(qr)
    elif g_read.stride(-1) != 1:
        g_read = g_read.contiguous()
    eta = eta.reshape(H).contiguous()
    qr, kr, v, kmix, vmix, w, s, m = (x.contiguous() for x in (qr, kr, v, kmix, vmix, w, s, m))
    # g_m = G^T q~ (partial sums over token ranges) and g_q~ = G m
    gqr = torch.empty_like(qr)
    tps = _tps(T, NS_GM, BT_GM, NBH)
    ns = triton.cdiv(T, tps)
    part = kr.new_empty(ns, NBH, D, D)
    _bwd_gm[(ns, NBH)](g_read, qr, m, gqr, part, T, H, tps, NBH, *g_read.stride()[:3], D=D, BD=BD, BT=BT_GM, PREC=PREC,
                       num_warps=WARPS_GM, num_stages=STAGES_GM)
    gm, gs, gw = (torch.empty_like(w) for _ in range(3))
    geta = kr.new_empty(NBH)
    _bwd_finish[(NBH,)](part, w if g_w is None else g_w.contiguous(), s, w, eta, gm, gs, gw, geta, H, ns, NBH,
                        HAS_GW=g_w is not None, D=D, BD=BD)
    # the token sums' gradients of k~, v, k_mix and v_mix
    gkm, gvm, gkr, gv = (torch.empty_like(kr) for _ in range(4))
    tps = _tps(T, NS_BWD, BT_BWD, NBH, 2)
    ns = triton.cdiv(T, tps)
    _bwd_dots[(2 * ns, NBH)](kr, v, kmix, vmix, gm, gs, gkm, gvm, gkr, gv, T, tps, ns, D=D, BD=BD, BT=BT_BWD,
                             PREC=PREC, num_warps=WARPS_BWD, num_stages=STAGES_BWD)
    geta = geta.view(B, H).sum(0).view(1, H, 1, 1)
    if trace:                                             # + the trace's (its kernel adds gkr, gv to its own)
        gkr, gv, gek, gev, gP = trace_bwd(kr, v, ek, ev, P, g_ek, g_ev, gkm, gvm, gkr, gv, inplace=True)
        return gqr, gkr, gv, gek, gev, gP, gw, geta
    return gqr, gkr, gv, gkm, gvm, gw, geta


# ---------------------------------------------------------------------------------------------------- custom ops
@torch.library.custom_op('arc::bands_sums_fwd', mutates_args=(), device_types='cuda')
def bands_sums_fwd(qr: torch.Tensor, kr: torch.Tensor, v: torch.Tensor, ek: torch.Tensor, ev: torch.Tensor,
                   P: torch.Tensor, w: torch.Tensor, eta: torch.Tensor, store: torch.dtype, save_mix: bool
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor]:
    """The trace update and the token sums: qr, kr, v [B,H,T,D] FP32, ek, ev [B,H,T,2RD], P [7,H,R], w [B,H,D,D],
    eta [1,H,1,1] -> read [B,H,T,D] (a view of [B,T,H,D]), new ek, ev (in `store`), W', S, m [B,H,D,D] and, with
    save_mix (for the backward pass), k_mix, v_mix [B,H,T,D] (else empty)."""
    return _forward(qr, kr, v, ek, ev, P, None, None, w, eta, store, save_mix)


@bands_sums_fwd.register_fake
def _(qr, kr, v, ek, ev, P, w, eta, store, save_mix):
    B, H, T, D = kr.shape
    R = P.shape[-1]
    new = lambda: kr.new_empty(B, H, T, 2 * R * D, dtype=store)
    mix = lambda: torch.empty_like(kr) if save_mix else kr.new_empty(0)
    e = torch.empty_like
    return _read_out(qr), new(), new(), e(w), e(w), e(w), mix(), mix()


@torch.library.custom_op('arc::bands_sums_bwd', mutates_args=(), device_types='cuda')
def bands_sums_bwd(g_read: torch.Tensor | None, g_w: torch.Tensor | None, g_ek: torch.Tensor | None,
                   g_ev: torch.Tensor | None, qr: torch.Tensor, kr: torch.Tensor, v: torch.Tensor, ek: torch.Tensor,
                   ev: torch.Tensor, P: torch.Tensor, kmix: torch.Tensor, vmix: torch.Tensor, w: torch.Tensor,
                   eta: torch.Tensor, s: torch.Tensor, m: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor]:
    """Gradients of q~, k~, v, ek, ev, P, W and eta."""
    return _backward(g_read, g_w, g_ek, g_ev, qr, kr, v, ek, ev, P, kmix, vmix, w, eta, s, m)


@bands_sums_bwd.register_fake
def _(g_read, g_w, g_ek, g_ev, qr, kr, v, ek, ev, P, kmix, vmix, w, eta, s, m):
    e = torch.empty_like
    return e(qr), e(kr), e(v), e(ek), e(ev), e(P), e(w), e(eta)


@torch.library.custom_op('arc::sums_fwd', mutates_args=(), device_types='cuda')
def sums_fwd(qr: torch.Tensor, kr: torch.Tensor, v: torch.Tensor, kmix: torch.Tensor, vmix: torch.Tensor,
             w: torch.Tensor, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The token sums alone (k_mix, v_mix given): read (a view of [B,T,H,D]), W', S, m."""
    read, _, _, wn, s, m, _, _ = _forward(qr, kr, v, None, None, None, kmix, vmix, w, eta, None)
    return read, wn, s, m


@sums_fwd.register_fake
def _(qr, kr, v, kmix, vmix, w, eta):
    return _read_out(qr), torch.empty_like(w), torch.empty_like(w), torch.empty_like(w)


@torch.library.custom_op('arc::sums_bwd', mutates_args=(), device_types='cuda')
def sums_bwd(g_read: torch.Tensor | None, g_w: torch.Tensor | None, qr: torch.Tensor, kr: torch.Tensor,
             v: torch.Tensor, kmix: torch.Tensor, vmix: torch.Tensor, w: torch.Tensor, eta: torch.Tensor,
             s: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                        torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of q~, k~, v, k_mix, v_mix, W and eta."""
    return _backward(g_read, g_w, None, None, qr, kr, v, None, None, None, kmix, vmix, w, eta, s, m)


@sums_bwd.register_fake
def _(g_read, g_w, qr, kr, v, kmix, vmix, w, eta, s, m):
    e = torch.empty_like
    return e(qr), e(kr), e(v), e(kmix), e(vmix), e(w), e(eta)
