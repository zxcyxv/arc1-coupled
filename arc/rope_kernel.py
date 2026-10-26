"""The unit q/k and their RoPE of arc.model.Coupled.bands_step as one Triton kernel each way (CUDA, FP32 arithmetic).

From the projections qkv [B,T,3,H,D] (q, k, v; BF16 under autocast, or FP32), per head h, token t and channel pair p:
    x~ = x / (|x| + eps);   y_2p = x~_2p cos - x~_2p+1 sin,   y_2p+1 = x~_2p sin + x~_2p+1 cos
for x = q and k, with cos, sin [H,T,P] = Coupled.rope_tables() (computed once per segment); v is only cast. The outputs
qr, kr, v [B,H,T,D] are FP32 and contiguous (the layout the bands trace and the token sums read), so the head split, the
casts and the RoPE are one pass over qkv. The same formulas as the PyTorch path; only the memory traffic changes.

Backward: the rotation is orthogonal, so x~ = R^T y and g_x~ . x~ = g_y . y; with n = |x|,
    g_x = (R^T g_y - (g_y . y) (n + eps) / n R^T y) / (n + eps)       (the second term 0 where n = 0, as autograd)
    g_cos = sum_b sum_{q,k} (g_y,2p x~_2p + g_y,2p+1 x~_2p+1),   g_sin = sum_b sum_{q,k} (g_y,2p+1 x~_2p - g_y,2p x~_2p+1)
from the saved outputs and the norms n (no input is kept). g_qkv comes out in the layout and dtype of qkv; the table
gradients are summed over NB lanes inside each program, then over the lane chunks (deterministic).
"""
import torch
import triton
import triton.language as tl

BT, WARPS = 32, 4                                         # forward: tokens per program, warps
BT_BWD, NB, WARPS_BWD = 32, 12, 8                         # backward: tokens and lanes per program


@triton.jit
def _rope_unit(X, C, S, eps, D: tl.constexpr, BT: tl.constexpr):
    x0, x1 = tl.split(tl.reshape(X, (BT, D // 2, 2)))
    n = tl.sqrt_rn(tl.sum(x0 * x0 + x1 * x1, 1))      # IEEE sqrt and division, as PyTorch
    den = (n + eps)[:, None]
    x0 = tl.div_rn(x0, den)
    x1 = tl.div_rn(x1, den)
    y = tl.join(x0 * C - x1 * S, x0 * S + x1 * C)
    return tl.reshape(y, (BT, D)), n


@triton.jit
def _qk_rope_fwd(QKV, COS, SIN, QR, KR, VO, NQ, NK, T, H, eps, D: tl.constexpr, BT: tl.constexpr):
    pid_t, pid_bh = tl.program_id(0), tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    t = pid_t * BT + tl.arange(0, BT)
    tm = t < T
    d = tl.arange(0, D)
    p = tl.arange(0, D // 2)
    mask = tm[:, None]
    xo = ((b * T + t)[:, None] * 3 * H + h) * D + d[None, :]          # qkv[b, t, j, h, :] at xo + j H D
    oo = (pid_bh * T + t)[:, None] * D + d[None, :]
    to = (h * T + t)[:, None] * (D // 2) + p[None, :]
    c = tl.load(COS + to, mask=mask, other=0.0)
    s = tl.load(SIN + to, mask=mask, other=0.0)
    q = tl.load(QKV + xo, mask=mask, other=0.0).to(tl.float32)
    qr, nq = _rope_unit(q, c, s, eps, D, BT)
    tl.store(QR + oo, qr, mask=mask)
    tl.store(NQ + pid_bh * T + t, nq, mask=tm)
    k = tl.load(QKV + xo + H * D, mask=mask, other=0.0).to(tl.float32)
    kr, nk = _rope_unit(k, c, s, eps, D, BT)
    tl.store(KR + oo, kr, mask=mask)
    tl.store(NK + pid_bh * T + t, nk, mask=tm)
    v = tl.load(QKV + xo + 2 * H * D, mask=mask, other=0.0).to(tl.float32)
    tl.store(VO + oo, v, mask=mask)


@triton.jit
def _rope_unit_bwd(G, Y, n, C, S, eps, D: tl.constexpr, BT: tl.constexpr):
    g0, g1 = tl.split(tl.reshape(G, (BT, D // 2, 2)))
    y0, y1 = tl.split(tl.reshape(Y, (BT, D // 2, 2)))
    dot = tl.sum(g0 * y0 + g1 * y1, 1)
    xh0 = C * y0 + S * y1                                  # x~ = R^T y
    xh1 = C * y1 - S * y0
    gc = g0 * xh0 + g1 * xh1
    gs = g1 * xh0 - g0 * xh1
    den = n + eps
    coef = tl.where(n > 0, tl.div_rn(dot * den, n), 0.0)[:, None]
    den = den[:, None]
    gx0 = tl.div_rn(C * g0 + S * g1 - coef * xh0, den)
    gx1 = tl.div_rn(C * g1 - S * g0 - coef * xh1, den)
    return tl.reshape(tl.join(gx0, gx1), (BT, D)), gc, gs


@triton.jit
def _qk_rope_bwd(GQR, GKR, GV, QR, KR, NQ, NK, COS, SIN, GQKV, GCOS, GSIN, B, T, H, eps, NB,
                 D: tl.constexpr, BT: tl.constexpr):
    pid_t, h, pid_c = tl.program_id(0), tl.program_id(1), tl.program_id(2)     # lanes pid_c NB .. + NB
    t = pid_t * BT + tl.arange(0, BT)
    tm = t < T
    d = tl.arange(0, D)
    p = tl.arange(0, D // 2)
    mask = tm[:, None]
    to = (h * T + t)[:, None] * (D // 2) + p[None, :]
    c = tl.load(COS + to, mask=mask, other=0.0)
    s = tl.load(SIN + to, mask=mask, other=0.0)
    gc = tl.zeros([BT, D // 2], dtype=tl.float32)
    gs = tl.zeros([BT, D // 2], dtype=tl.float32)
    for b in range(pid_c * NB, tl.minimum(pid_c * NB + NB, B)):
        bh = b * H + h
        oo = (bh * T + t)[:, None] * D + d[None, :]
        xo = ((b * T + t)[:, None] * 3 * H + h) * D + d[None, :]
        nq = tl.load(NQ + bh * T + t, mask=tm, other=1.0)
        gx, gc1, gs1 = _rope_unit_bwd(tl.load(GQR + oo, mask=mask, other=0.0), tl.load(QR + oo, mask=mask, other=0.0),
                                      nq, c, s, eps, D, BT)
        tl.store(GQKV + xo, gx.to(GQKV.dtype.element_ty), mask=mask)
        nk = tl.load(NK + bh * T + t, mask=tm, other=1.0)
        gx, gc2, gs2 = _rope_unit_bwd(tl.load(GKR + oo, mask=mask, other=0.0), tl.load(KR + oo, mask=mask, other=0.0),
                                      nk, c, s, eps, D, BT)
        tl.store(GQKV + xo + H * D, gx.to(GQKV.dtype.element_ty), mask=mask)
        gc += gc1 + gc2
        gs += gs1 + gs2
        tl.store(GQKV + xo + 2 * H * D, tl.load(GV + oo, mask=mask, other=0.0).to(GQKV.dtype.element_ty), mask=mask)
    co = pid_c * H * T * (D // 2) + to                    # this lane chunk's partial table gradients
    tl.store(GCOS + co, gc, mask=mask)
    tl.store(GSIN + co, gs, mask=mask)


@torch.library.custom_op('arc::qk_rope', mutates_args=(), device_types='cuda')
def qk_rope(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, heads: int, eps: float
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """qkv [B,T,3HD] (BF16 or FP32), cos, sin [1,H,T,D/2] FP32 -> qr, kr, v [B,H,T,D] FP32 and the norms |q|, |k|
    [B,H,T] (kept for the backward pass)."""
    B, T, E = qkv.shape
    H = heads
    D = E // 3 // H
    qkv, cos, sin = qkv.contiguous(), cos.contiguous(), sin.contiguous()
    qr, kr, v = (qkv.new_empty(B, H, T, D, dtype=torch.float32) for _ in range(3))
    nq, nk = (qkv.new_empty(B, H, T, dtype=torch.float32) for _ in range(2))
    _qk_rope_fwd[(triton.cdiv(T, BT), B * H)](qkv, cos, sin, qr, kr, v, nq, nk, T, H, eps, D=D, BT=BT,
                                               num_warps=WARPS)
    return qr, kr, v, nq, nk


@qk_rope.register_fake
def _(qkv, cos, sin, heads, eps):
    B, T, E = qkv.shape
    D = E // 3 // heads
    f = lambda *s: qkv.new_empty(*s, dtype=torch.float32)
    return f(B, heads, T, D), f(B, heads, T, D), f(B, heads, T, D), f(B, heads, T), f(B, heads, T)


@torch.library.custom_op('arc::qk_rope_bwd', mutates_args=(), device_types='cuda')
def qk_rope_bwd(gqr: torch.Tensor | None, gkr: torch.Tensor | None, gv: torch.Tensor | None, qr: torch.Tensor,
                kr: torch.Tensor, nq: torch.Tensor, nk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                dtype: torch.dtype, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = qr.shape
    z = lambda g: torch.zeros_like(qr) if g is None else g.contiguous()
    gqr, gkr, gv = z(gqr), z(gkr), z(gv)
    cos, sin = cos.contiguous(), sin.contiguous()
    gqkv = qr.new_empty(B, T, 3 * H * D, dtype=dtype)
    nc = triton.cdiv(B, NB)
    gcos, gsin = (cos.new_empty(nc, *cos.shape[1:]) for _ in range(2))
    _qk_rope_bwd[(triton.cdiv(T, BT_BWD), H, nc)](gqr, gkr, gv, qr, kr, nq, nk, cos, sin, gqkv, gcos, gsin, B, T, H,
                                                  eps, NB, D=D, BT=BT_BWD, num_warps=WARPS_BWD)
    return gqkv, gcos.sum(0, keepdim=True), gsin.sum(0, keepdim=True)


@qk_rope_bwd.register_fake
def _(gqr, gkr, gv, qr, kr, nq, nk, cos, sin, dtype, eps):
    B, H, T, D = qr.shape
    return qr.new_empty(B, T, 3 * H * D, dtype=dtype), cos.new_empty(cos.shape), sin.new_empty(sin.shape)


def _setup(ctx, inputs, output):
    qkv, cos, sin, heads, eps = inputs
    qr, kr, v, nq, nk = output
    ctx.save_for_backward(qr, kr, nq, nk, cos, sin)
    ctx.dtype, ctx.eps = qkv.dtype, eps


def _backward(ctx, gqr, gkr, gv, gnq, gnk):
    qr, kr, nq, nk, cos, sin = ctx.saved_tensors
    gqkv, gcos, gsin = qk_rope_bwd(gqr, gkr, gv, qr, kr, nq, nk, cos, sin, ctx.dtype, ctx.eps)
    return gqkv, gcos, gsin, None, None


qk_rope.register_autograd(_backward, setup_context=_setup)
