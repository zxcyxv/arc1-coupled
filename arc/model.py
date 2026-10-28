"""Coupled model: bands perception + a per-task lasso rule head, one segment per call.

Refactored from zxcyxv/something (commit 093911e): the bands path of lt/model.py (LT.block_step, bands_step) and
lt/coupled.py, with lt/coupled.py's Gamma term (frames) of 2026-10-25. Creation order and every operation are the
same, so with the same seed the initial weights and outputs equal the old code's (tests/test_model.py reductions;
the bit check against the old repository is recorded in docs/decisions.md).

A lane is one pair: the input canvas and the output canvas (C x C each, C = 30) stacked along rows, T = 2 C^2 tokens:
0 unknown (every output-canvas cell, demonstrations and queries alike), 1 pad, 2..11 colours 0..9, 12 EOS. Lanes never
mix in perception; the lanes of one task version form a group, and the demonstrations reach a query only through the
rule head's Theta (and Gamma).

One block (shared weights; two residual RMSNorms, AGENTS.md):
    u = h + sqrt(d) inj;  q, k, v = heads(W_q u), heads(W_k u), heads(W_v u);  q~, k~ = RoPE(q / |q|), RoPE(k / |k|)
    bands r = 1..R, per head: E_K,r <- z_r E_K,r + gamma_r k~;  D_V,r = z_r E_V,r;  E_V,r <- D_V,r + gamma_r v
    S = sum_r Re(a_r v^T E_K,r + b_r D_V,r^T k~);  W <- (1 - eta) W + eta S;  read = q~ (sum_t v k~^T + W)^T
    h' = RMSNorm(u + W_O read);  h = RMSNorm(h' + W_D(silu(g) * a)), [g; a] = W_GU h'
with z_r = exp(-e^nu + i theta) learned per head and band, gamma_r = sqrt(1 - |z_r|^2), eta learned per head (capped).

Rule head after every block: phi(o) = unit(W_phi h(o)) on the output-canvas cells; per lane, leave-one-out statistics
of the other demonstration lanes of its group (G = sum phi phi^T / n, S = sum onehot(y) phi^T / n) and one proximal step
Theta <- soft(Theta - (Theta G - S), lam) (valid at step 1: lambda_max(G) <= tr G = 1); read r(o) = Theta phi(o), fed
back through W_fb into the next block's output-canvas injection; logits = classifier(h) + tau r.
rule_heads = H > 1: heads 1..H-1 are further rule synapses of the same form, each with its own phi_h = unit(W_phi,h h),
lasso Theta_h (lambda_h, step 1) and feedback W_fb,h r_h added to the injection; logits keep head 0's tau r.

frames = K > 0 (cmap's Gamma term): F = K + 1 frame colours per output cell, kappa_0 = the colour histogram of the
lane's input grid, kappa_k = a softmax read of the input grid's colour cells from the output cell (learned 2D RoPE on
canvas coordinates), all / sqrt(F) so sum_k |kappa_k|^2 <= 1; r(o) = Theta phi(o) + sum_k Gamma_k kappa_k(o), one
lasso over (Theta, Gamma), one proximal step on each block in turn. Gamma has no colour index.

A segment runs `blocks` blocks from the carry, the last `grad_blocks` with gradients, and returns the detached carry.
With fused_bands (CUDA, FP32 forward) the trace update runs as one Triton kernel each way (arc/bands_kernel.py), and so
do the unit q/k, their RoPE and the head split (arc/rope_kernel.py): the same formulas with less memory traffic. The RoPE
tables (cos, sin of the learned angles) and the frames' tables are computed once per segment. The token sums (S, W, read) are an autograd
Function (BandsFused / BandsRead) whose backward pass stays FP32 inside the compiled, autocast block; on CUDA in FP32 they
are 3xTF32 tensor-core Triton kernels (arc/sums_kernel.py): FP32-level accuracy, FP32 accumulation. The trainer compiles
one layer (block_step and the rule head after it, `layer`) with static shapes, and captures each micro-batch's segment,
loss and backward pass as one CUDA graph (arc/train.py).
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .rope_kernel import qk_rope
    from .sums_kernel import bands_sums_bwd, bands_sums_fwd, sums_bwd, sums_fwd
except ImportError:                                      # no Triton: the PyTorch path only
    qk_rope = bands_sums_bwd = bands_sums_fwd = sums_bwd = sums_fwd = None

VOCAB = 13
UNKNOWN, PAD, EOS = 0, 1, 12                         # colours are 2..11


@dataclass
class ModelConfig:
    canvas: int = 30                  # C: each lane is two C x C canvases
    d: int = 512
    heads: int = 8
    blocks: int = 8                   # per segment
    grad_blocks: int = 4              # the last blocks of a segment run with gradients
    mlp_expansion: float = 4.0
    bands: int = 4
    band_periods: tuple = (2.0, 6.0, 20.0, 64.0)   # initial periods p_r: theta_r = 2 pi / p_r, |z_r| = e^{-1/p_r}
    band_coef_std: float = 0.1        # std of each real component of the initial a_r, b_r
    outer_eta: float = 0.125          # initial EMA rate of W, learned per head and capped at this value
    eps: float = 1e-4                 # q, k normalisation x / (|x| + eps)
    phi: int = 64                     # rule-head feature width
    lam_init: float = 1e-3            # lasso threshold (learned, softplus): lam_floor + softplus(lam_raw)
    lam_floor: float = 0.0            # lower bound of the lasso threshold (the training loss alone drives it to 0)
    lam_learn: bool = True            # False: lambda = lam_floor exactly (no learned part; lam_floor 0 = no sparse LTD)
    rope_1d: bool = False             # True: positions = the lane's sequence index (else (row, col) on the canvas)
    rope_fixed: Optional[bool] = None # True: TRM's fixed frequencies, a buffer (None: = rope_1d). 1D: rope_base^(-2i / dh)
                                      # per pair; 2D: axial, the first dh / 4 pairs on rows, the rest on columns, each
                                      # rope_base^(-4i / dh). False: learned angles (initialised uniform, as before)
    rope_base: float = 10000.0
    tau_init: float = 1.0             # read gain
    tau_learn: bool = True            # False: tau stays tau_init (a buffer, no gradient)
    direct_head: bool = True          # False: logits = bias + tau r (the rule head's read only; no classifier(h) path from
                                      # the query's own perception to the output, as in cmap)
    out_ffn: bool = False             # True: logits = classifier(FFN_out(base + sqrt(d) W_fb r)) on the output canvas after
                                      # the last head (a dedicated post-norm SwiGLU FFN, no attention), no tau r term;
                                      # base = h with direct_head, else sqrt(d) emb(unknown) (a constant: r alone decides)
    rule_heads: int = 1               # independent rule synapses: head 0 as above (with Gamma), heads 1.. a Theta each
                                      # (own phi, lambda, feedback W_fb); Theta [L,13,H p] and r [L,O,13 H] carry them all
    frames: int = 0                   # K > 0: Gamma over K softmax frames + the grid histogram
    frame_dim: int = 32               # q/k width of a frame (frame_dim / 4 frequencies per axis)
    frame_fixed: bool = False         # True: the frames' frequencies stay at their initial values (a buffer, no gradient)
    frame_bf16: bool = True          # with amp on CUDA: the frames' softmax read as BF16 flash attention
    trace_dtype: str = 'float32'      # E_K, E_V between blocks with an FP32 forward: float32, bfloat16 or float16;
                                      # every update is computed in FP32
    fused_bands: bool = False         # CUDA FP32: the trace update as one Triton kernel each way (arc/bands_kernel.py)
    amp: bool = True                  # BF16 autocast for the projections and the FFN on CUDA
    dtype: str = 'float32'            # float64 for tests

    def __post_init__(self):
        self.band_periods = tuple(float(p) for p in self.band_periods)
        if self.d % self.heads or (self.d // self.heads) % 2:
            raise ValueError('d / heads must be an even integer')
        if not 0 < self.grad_blocks <= self.blocks:
            raise ValueError('0 < grad_blocks <= blocks')
        if len(self.band_periods) != self.bands or min(self.band_periods) <= 0:
            raise ValueError('one positive initial period per band')
        if not 0 < self.outer_eta < 1:
            raise ValueError('0 < outer_eta < 1')
        if self.frames and self.frame_dim % 4:
            raise ValueError('frame_dim must be a multiple of 4')
        if self.trace_dtype not in ('float32', 'bfloat16', 'float16'):
            raise ValueError('trace_dtype: float32, bfloat16 or float16')


def trunc_normal_(tensor, std=1.0, lower=-2.0, upper=2.0):
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        a, b = math.erf(lower / math.sqrt(2)), math.erf(upper / math.sqrt(2))
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pu, pl = c * math.exp(-0.5 * upper ** 2), c * math.exp(-0.5 * lower ** 2)
        cs = std / math.sqrt(1 - (upper * pu - lower * pl) / z - ((pu - pl) / z) ** 2)
        tensor.uniform_(a, b).erfinv_().mul_(math.sqrt(2) * cs)
        return tensor.clip_(lower * cs, upper * cs)


def rms_norm(x, eps=1e-5):
    """URM RMSNorm: FP32 mean of squares over all hidden channels, no scale or bias, input dtype out."""
    dtype = x.dtype
    xf = x if dtype == torch.float64 else x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(dtype)


def soft(x, t):
    return torch.sign(x) * F.relu(x.abs() - t)


def _sums(qr, kr, v, k_mix, v_mix, w, eta):
    s = v.mT @ k_mix + v_mix.mT @ kr
    w_new = (1 - eta) * w + eta * s
    m = v.mT @ kr + w_new
    return qr @ m.mT, w_new, s, m


def _sums_bwd(g_read, g_w, qr, w, eta, s, m):
    """Gradients from (read, W'): those of q~, m, S, the incoming W and eta."""
    g_read = g_read.contiguous()                        # one copy of the head-split gradient serves both GEMMs
    g_m = g_read.mT @ qr
    g_qr = g_read @ m
    g_w = g_m if g_w is None else g_w + g_m            # W' enters m and the output
    g_s = eta * g_w
    g_eta = ((s - w) * g_w).sum((0, 2, 3), keepdim=True)
    return g_qr, g_m, g_s, (1 - eta) * g_w, g_eta


class BandsRead(torch.autograd.Function):
    """The token sums of bands_step, FP32 (FP64) both ways:
        S = v^T k_mix + v_mix^T k~;  W' = (1 - eta) W + eta S;  read = q~ (v^T k~ + W')^T.
    On CUDA in FP32 they are 3xTF32 tensor-core kernels (arc/sums_kernel.py: FP32-level accuracy); otherwise
    PyTorch GEMMs. The backward pass is written out and runs with autocast off: under torch.compile, AOTAutograd traces
    the backward of plain matmuls with the caller's autocast state, so in the trainer's compiled block (BF16 autocast)
    their gradients were BF16 GEMMs despite the FP32 forward."""

    @staticmethod
    def forward(ctx, qr, kr, v, k_mix, v_mix, w, eta):
        ctx.kernel = sums_fwd is not None and qr.is_cuda and qr.dtype == torch.float32
        if ctx.kernel:
            read, w_new, s, m = sums_fwd(qr, kr, v, k_mix, v_mix, w, eta)
        else:
            read, w_new, s, m = _sums(qr, kr, v, k_mix, v_mix, w, eta)
        ctx.save_for_backward(qr, kr, v, k_mix, v_mix, w, eta, s, m)
        return read, w_new

    @staticmethod
    def backward(ctx, g_read, g_w):
        qr, kr, v, k_mix, v_mix, w, eta, s, m = ctx.saved_tensors
        if ctx.kernel:
            return sums_bwd(g_read, g_w, qr, kr, v, k_mix, v_mix, w, eta, s, m)
        with torch.autocast(qr.device.type, enabled=False):
            g_qr, g_m, g_s, g_w, g_eta = _sums_bwd(g_read, g_w, qr, w, eta, s, m)
            g_v = kr @ g_m.mT + k_mix @ g_s.mT
            g_kr = v @ g_m + v_mix @ g_s
            return g_qr, g_kr, g_v, v @ g_s, kr @ g_s.mT, g_w, g_eta


class BandsFused(torch.autograd.Function):
    """The trace update (arc/bands_kernel.py's formulas) with the token sums of BandsRead on the same token tiles (one
    kernel, arc/sums_kernel.py, 3xTF32 tensor-core products), then the read; backward: the sums' gradients on tensor
    cores, added by the trace gradient kernel to its own (no separate passes for the sums of the gradients). k_mix and
    v_mix are written only when a backward pass will need them."""

    @staticmethod
    def forward(ctx, qr, kr, v, ek, ev, P, w, eta, store):
        save = any(ctx.needs_input_grad)
        read, ek_new, ev_new, w_new, s, m, k_mix, v_mix = bands_sums_fwd(qr, kr, v, ek, ev, P, w, eta, store, save)
        ctx.save_for_backward(qr, kr, v, ek, ev, P, k_mix, v_mix, w, eta, s, m)
        return read, ek_new, ev_new, w_new

    @staticmethod
    def backward(ctx, g_read, g_ek, g_ev, g_w):
        g_qr, g_kr, g_v, g_ek, g_ev, g_P, g_w, g_eta = bands_sums_bwd(g_read, g_w, g_ek, g_ev, *ctx.saved_tensors)
        return g_qr, g_kr, g_v, g_ek, g_ev, g_P, g_w, g_eta, None


class Block(nn.Module):
    """The shared bands block's parameters (creation order = lt.model.Block for the bands config)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, H = cfg.d, cfg.heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        inter = int(cfg.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
        self.ffn_gate_up = nn.Linear(d, 2 * inter, bias=False)
        self.ffn_down = nn.Linear(inter, d, bias=False)
        with torch.no_grad():
            self.ffn_down.weight.zero_()
        self.rope_theta = nn.Parameter(torch.zeros(H, d // H // 2, 2))          # drawn in Coupled.__init__
        self.eta_raw = nn.Parameter(torch.full((H,), math.log(cfg.outer_eta / (1 - cfg.outer_eta))))
        p = torch.tensor(cfg.band_periods, dtype=torch.float64)
        self.band_pole = nn.Parameter(torch.stack((-p.log(), 2 * math.pi / p))[:, None].repeat(1, H, 1).float())
        self.band_coef = nn.Parameter(torch.zeros(4, H, cfg.bands))            # Re a, Im a, Re b, Im b (drawn later)

    def bands(self):
        """Per head and band [H,R]: Re z, Im z, gamma = sqrt(1 - |z|^2), Re a, Im a, Re b, Im b."""
        nu, theta = self.band_pole
        mag = torch.exp(-torch.exp(nu))
        return (mag * torch.cos(theta), mag * torch.sin(theta), torch.sqrt(-torch.expm1(-2 * torch.exp(nu))),
                *self.band_coef)

    def eta(self):
        return torch.sigmoid(self.eta_raw)

    @torch.no_grad()
    def project_eta_(self, cfg):
        """eta <= outer_eta: the W window is never shorter than 1 / outer_eta blocks."""
        self.eta_raw.clamp_(max=math.log(cfg.outer_eta / (1 - cfg.outer_eta)))


class Coupled(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = c = cfg
        self.forward_dtype = getattr(torch, c.dtype)
        C, d, H = c.canvas, c.d, c.heads
        self.C, self.O, self.T = C, C * C, 2 * C * C
        self.d, self.H, self.dh = d, H, d // H
        self.embed_scale = math.sqrt(d)
        # Creation order fixes the random draws (as lt.model.LT, then lt.coupled.Coupled).
        self.embed = nn.Embedding(VOCAB, d)
        trunc_normal_(self.embed.weight, std=1.0 / self.embed_scale)
        self.classifier = nn.Linear(d, VOCAB)
        self.block = Block(c)
        self.init_hidden = nn.Buffer(trunc_normal_(torch.empty(d, dtype=self.forward_dtype), std=1.0))
        with torch.no_grad():
            self.block.rope_theta.uniform_(-math.pi / 2, math.pi / 2)
            self.block.band_coef.normal_(0.0, c.band_coef_std)
        rope_dtype = torch.float64 if self.forward_dtype == torch.float64 else torch.float32
        t = torch.arange(self.T)
        fixed = c.rope_1d if c.rope_fixed is None else c.rope_fixed
        if fixed:                         # TRM's RoPE frequencies, no parameter
            dh = d // H
            th = torch.zeros_like(self.block.rope_theta)                       # [H, dh / 2, (row, col)]
            if c.rope_1d:
                th[..., 0] = 1.0 / (c.rope_base ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh)).to(th.dtype)
            else:                         # axial 2D: each axis gets half the pairs, a 1D RoPE of width dh / 2
                q = dh // 4
                f = 1.0 / (c.rope_base ** (torch.arange(0, dh // 2, 2, dtype=torch.float64) / (dh // 2))).to(th.dtype)
                th[:, :q, 0], th[:, q:, 1] = f, f
            del self.block.rope_theta                                          # after its draw: later draws unchanged
            self.block.register_buffer('rope_theta', th)
        if c.rope_1d:
            self.register_buffer('pos_row', t.to(rope_dtype), persistent=False)
            self.register_buffer('pos_col', torch.zeros(self.T, dtype=rope_dtype), persistent=False)
        else:
            self.register_buffer('pos_row', (t // C).to(rope_dtype), persistent=False)     # rows 0..2C-1 over the lane
            self.register_buffer('pos_col', (t % C).to(rope_dtype), persistent=False)
        self.w_phi = nn.Linear(d, c.phi, bias=False)
        self.w_fb = nn.Linear(VOCAB, d, bias=False)
        nn.init.zeros_(self.w_fb.weight)                       # the feedback path opens by learning
        self.lam_raw = nn.Parameter(torch.tensor(math.log(math.expm1(c.lam_init))))
        if not c.lam_learn:                                    # same creation order; the learned part is dropped
            del self.lam_raw
            self.register_buffer('lam_raw', torch.tensor(float('-inf')))
        if c.tau_learn:
            self.tau = nn.Parameter(torch.tensor(c.tau_init))
        else:
            self.register_buffer('tau', torch.tensor(c.tau_init))
        if c.frames:
            K, E = c.frames, c.frame_dim
            self.w_fq = nn.Linear(d, K * E, bias=False)
            self.w_fk = nn.Linear(d, K * E, bias=False)
            p = torch.logspace(math.log10(2.0), math.log10(4.0 * C), E // 4)     # periods 2 .. 4 C cells
            omega = (2 * math.pi / p)[None, None].repeat(K, 2, 1)                  # [K, axis, freq]
            if c.frame_fixed:
                self.register_buffer('frame_omega', omega)
            else:
                self.frame_omega = nn.Parameter(omega)
            r, cc = torch.meshgrid(torch.arange(C), torch.arange(C), indexing='ij')
            self.register_buffer('frame_pos', torch.stack([r, cc], -1).reshape(-1, 2).float(), persistent=False)
        if c.out_ffn:                                          # created last: the draws above are unchanged
            inter = int(c.mlp_expansion * d * 2 / 3 + 255) // 256 * 256
            self.out_gate_up = nn.Linear(d, 2 * inter, bias=False)
            self.out_down = nn.Linear(inter, d, bias=False)
            with torch.no_grad():
                self.out_down.weight.zero_()
        if c.rule_heads > 1:                                   # created last: the draws above are unchanged
            X = c.rule_heads - 1
            self.w_phi_x = nn.Linear(d, X * c.phi, bias=False)
            self.w_fb_x = nn.Linear(X * VOCAB, d, bias=False)
            nn.init.zeros_(self.w_fb_x.weight)
            self.lam_raw_x = nn.Parameter(torch.full((X,), math.log(math.expm1(c.lam_init))))
        self.to(self.forward_dtype)
        self.layer_fn = None                                   # compiled layer (trainer, evaluation)

    # ------------------------------------------------------------------ perception
    def amp(self, device):
        if not (self.cfg.amp and device.type == 'cuda' and self.forward_dtype == torch.float32):
            return False
        return torch.cuda.is_bf16_supported(including_emulation=False)

    def rope_angles(self):
        """[H, T, P] learned 2D RoPE angles theta_row * row + theta_col * col."""
        th = self.block.rope_theta
        return (th[..., 0, None] * self.pos_row + th[..., 1, None] * self.pos_col).transpose(-1, -2)

    @staticmethod
    def rope(x, tables):
        cos, sin = tables
        x0, x1 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).reshape_as(x)

    def rope_tables(self):
        """cos, sin [1,H,T,P] of rope_angles(): they depend only on rope_theta, so a segment computes them once for all
        its blocks (gradients reach rope_theta through them). Evaluated in FP64 and rounded once to the forward dtype:
        the angles reach |theta| (2C - 1) rad, where FP32 rounding of the angle itself is the largest table error."""
        th = self.block.rope_theta.double()
        angles = (th[..., 0, None] * self.pos_row.double() + th[..., 1, None] * self.pos_col.double()).transpose(-1, -2)
        angles = angles[None].contiguous()
        dt = self.block.rope_theta.dtype
        return angles.cos().to(dt), angles.sin().to(dt)

    def bands_step(self, qkv, ek, ev, w, tables=None):
        """One block of the bands memory in FP32 (FP64 on the test path). qkv [B,T,3d] the q, k, v projections; ek, ev
        [B,H,T,2RD] hold (Re, Im) x R bands of the k~ and v traces; w [B,H,D,D]; tables = rope_tables() (None:
        computed here). The bands are merged before the token sums, so S is two [D,T] @ [T,D] products for any R.
        Returns read [B,H,T,D] and the new (ek, ev, w)."""
        cfg = self.cfg
        with torch.autocast(device_type=qkv.device.type, enabled=False):
            dtype = torch.float64 if qkv.dtype == torch.float64 else torch.float32
            fused = cfg.fused_bands and qkv.is_cuda and dtype == torch.float32
            if tables is None:
                tables = self.rope_tables()
            cos, sin = (t.to(dtype) for t in tables)
            if fused:                                      # unit q, k + RoPE + head split in one pass
                qr, kr, v, _, _ = qk_rope(qkv, cos, sin, self.H, cfg.eps)
            else:
                unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
                q, k, v = qkv.view(*qkv.shape[:2], 3, self.H, self.dh).permute(2, 0, 3, 1, 4).unbind(0)
                qr, kr = self.rope(unit(q.to(dtype)), (cos, sin)), self.rope(unit(k.to(dtype)), (cos, sin))
                v = v.to(dtype)
            B, H, T, D = v.shape
            store = getattr(torch, cfg.trace_dtype) if dtype == torch.float32 else dtype
            eta = self.block.eta().to(dtype)[None, :, None, None]
            if fused:                                      # the trace kernel and the token sums
                return BandsFused.apply(qr, kr, v, ek, ev, torch.stack(self.block.bands()), w.to(dtype), eta, store)
            else:
                zr, zi, gamma, ar, ai, br, bi = (x.to(dtype)[None, :, None, :, None] for x in self.block.bands())
                ek_re, ek_im = ek.to(dtype).reshape(B, H, T, 2, cfg.bands, D).unbind(3)
                ev_re, ev_im = ev.to(dtype).reshape(B, H, T, 2, cfg.bands, D).unbind(3)
                ek_re, ek_im = zr * ek_re - zi * ek_im + gamma * kr.unsqueeze(3), zr * ek_im + zi * ek_re  # block in
                dv_re, dv_im = zr * ev_re - zi * ev_im, zr * ev_im + zi * ev_re                             # block out
                k_mix = (ar * ek_re - ai * ek_im).sum(3)
                v_mix = (br * dv_re - bi * dv_im).sum(3)
                ev_re = dv_re + gamma * v.unsqueeze(3)
                ek = torch.stack((ek_re, ek_im), 3).reshape(B, H, T, -1).to(store)
                ev = torch.stack((ev_re, dv_im), 3).reshape(B, H, T, -1).to(store)
                read, w = BandsRead.apply(qr, kr, v, k_mix, v_mix, w.to(dtype), eta)
        return read, ek, ev, w

    def block_step(self, h, emb, fb, ek, ev, w, tables=None):
        """One recurrent block: the new hidden state and bands state. The block's input is emb [B,T,d] with the rule
        head's feedback fb [B,O,d] added on the output canvas (fb None: emb alone); tables = rope_tables()."""
        L = self.block
        O = self.O
        inj = emb if fb is None else torch.cat([emb[:, :O], emb[:, O:] + fb], 1)
        u = h + self.embed_scale * inj
        B, T, _ = u.shape
        qkv = F.linear(u, torch.cat([L.q_proj.weight, L.k_proj.weight, L.v_proj.weight]))   # one GEMM for q, k, v
        read, ek, ev, w = self.bands_step(qkv, ek, ev, w, tables)
        read = read.transpose(1, 2).reshape(B, T, self.d).to(u.dtype)
        h = rms_norm(u + L.out_proj(read))
        g, a = L.ffn_gate_up(h).chunk(2, dim=-1)
        out = L.ffn_down(F.silu(g) * a)
        return rms_norm(h + out), ek, ev, w

    # ------------------------------------------------------------------ rule head
    def features(self, h):
        """phi [L,O,phi] of the output-canvas cells (unit length) from h [L,T,d]."""
        x = self.w_phi(h[:, self.O:].to(self.w_phi.weight.dtype))
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def features_x(self, h):
        """phi of rule heads 1.. [L,O,H-1,phi], each unit length."""
        x = self.w_phi_x(h[:, self.O:].to(self.w_phi_x.weight.dtype)).unflatten(-1, (-1, self.cfg.phi))
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def feedback(self, r):
        """W_fb r of every rule head, [L,O,d] from r [L,O,13 H]."""
        fb = self.w_fb(r[..., :VOCAB])
        return fb if self.cfg.rule_heads == 1 else fb + self.w_fb_x(r[..., VOCAB:])

    def frame_tables(self, dtype=None):
        """cos, sin [K,O,E] of the frames' angles omega_k,axis,f times the cell's canvas row / column, each repeated for
        the two channels of its pair. They depend only on frame_omega, so a segment computes them once (eager) and the
        compiled layer takes them as inputs: torch 2.11's inductor miscompiled the frame_omega gradient (the reduction
        through cos/sin over the cells) at O = 121 (canvas 11, FP32; off by ~100%, depending on the lane count)."""
        dt = dtype or self.frame_omega.dtype
        K, E = self.cfg.frames, self.cfg.frame_dim
        ang = torch.einsum('oa,kaf->koaf', self.frame_pos.to(dt), self.frame_omega.to(dt)).reshape(K, self.O, E // 2)
        return ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)

    def frame_rope(self, x, ftab=None):
        """x [L,K,O,E] rotated per (axis, frequency) pair: x cos + rot(x) sin (the products and sums of the pair form
        (x1 cos - x2 sin, x1 sin + x2 cos), so the same values); ftab = frame_tables() (None: computed here)."""
        cos, sin = self.frame_tables(x.dtype) if ftab is None else (t.to(x.dtype) for t in ftab)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return x * cos + torch.stack([-x2, x1], -1).flatten(-2) * sin

    def frame_colours(self, h, x_in, amp=None, ftab=None):
        """kappa [L,F,O,13] of the output-canvas cells from h [L,T,d] and the input-canvas tokens x_in [L,O]; colour
        cells are tokens 2..11 (pad and EOS are never read). The softmax read runs through SDPA (no [O,O] tensor kept
        for the backward pass on CUDA). amp: self.amp(h.device), passed by segment (None: computed here); ftab:
        frame_tables(), passed by segment (None: computed here)."""
        c = self.cfg
        L, O, K, E = h.shape[0], self.O, c.frames, c.frame_dim
        dt = self.w_fq.weight.dtype
        live = (x_in >= 2) & (x_in <= 11)
        col = F.one_hot(x_in, VOCAB).to(dt) * live[..., None]
        bf16 = (self.amp(h.device) if amp is None else amp) and c.frame_bf16
        proj = ((lambda lin, x: F.linear(x.bfloat16(), lin.weight.bfloat16()).to(dt)) if bf16
                else (lambda lin, x: lin(x.to(dt))))
        if ftab is None:
            ftab = self.frame_tables(dt)
        q = self.frame_rope(proj(self.w_fq, h[:, O:]).view(L, O, K, E).transpose(1, 2), ftab)
        k = self.frame_rope(proj(self.w_fk, h[:, :O]).view(L, O, K, E).transpose(1, 2), ftab)
        bias = torch.where(live, 0.0, -1e4).to(dt)
        if bf16:
            # BF16 flash attention: the key mask rides in one extra q/k channel (q 1, k 0 or -1e4) and q carries the
            # 1/sqrt(E) scale, so no mask tensor is passed; widths padded to a multiple of 8 (values too).
            W = -(-(E + 1) // 8) * 8
            qx = F.pad(torch.cat([q / math.sqrt(E), torch.ones_like(q[..., :1])], -1), (0, W - E - 1))
            kx = F.pad(torch.cat([k, bias[:, None, :, None].expand(L, K, O, 1)], -1), (0, W - E - 1))
            val = F.pad(col, (0, W - VOCAB))[:, None].expand(L, K, O, W)
            kap = F.scaled_dot_product_attention(qx.bfloat16(), kx.bfloat16(), val.bfloat16(), scale=1.0)
            kap = kap[..., :VOCAB].to(dt)
        else:
            Ev = max(E, VOCAB)                                                     # values padded to the q/k width
            val = F.pad(col, (0, Ev - VOCAB))[:, None].expand(L, K, O, Ev)
            kap = F.scaled_dot_product_attention(q, k, val, attn_mask=bias[:, None, None, :])[..., :VOCAB]
        hist = col.sum(1) / live.sum(1, keepdim=True).clamp_min(1).to(dt)
        return torch.cat([hist[:, None, None].expand(L, 1, O, VOCAB), kap], 1) / math.sqrt(K + 1)

    def statistics(self, phi, y, m, demo, group, ng=None, kap=None, phx=None):
        """Leave-one-out statistics per lane over the OTHER demonstration lanes of its group: G [L,p,p], S [L,13,p]
        and, with frame colours kap [L,F,O,13], C [L,F,13,p], H [L,F,F], s [L,F]. phi [L,O,p], y one-hot [L,O,13],
        m [L,O] labelled cells, demo [L], group [L] (any integer labels; ng is not needed and ignored). The sums over
        the other lanes of the group are one 0/1 [L,L] matmul (exact products, no cancellation of the own lane, static
        shapes, no atomics). The contractions with a small output (H, s here; the Gamma terms in step and rule_head)
        are products and sums, which the compiled head fuses into reductions (as einsums they ran as slow cuBLAS
        GEMVs); without compile H materialises an [L,F,F,O,13] product."""
        w = m * demo[:, None]
        parts = [torch.einsum('lo,lop,loq->lpq', w, phi, phi), torch.einsum('lo,loc,lop->lcp', w, y, phi)]
        if kap is not None:
            wk = w[:, None, :, None] * kap
            parts += [torch.einsum('lkoc,lop->lkcp', wk, phi), (wk[:, :, None] * kap[:, None]).sum((-2, -1)),
                      (wk * y[:, None]).sum((-2, -1))]
        if phx is not None:                                    # rule heads 1..: Gx [L,H-1,p,p], Sx [L,H-1,13,p] last
            parts += [torch.einsum('lo,lohp,lohq->lhpq', w, phx, phx), torch.einsum('lo,loc,lohp->lhcp', w, y, phx)]
        nl = w.sum(-1)
        L = group.shape[0]
        others = (group[:, None] == group[None, :]) & ~torch.eye(L, dtype=torch.bool, device=group.device)
        others = others.to(nl.dtype)                           # [L,L]: the other lanes of each lane's group
        n = (others @ nl).clamp_min(1)                         # own lane left out (a non-demonstration adds nothing)
        return [(others @ P.reshape(L, -1)).view_as(P) / n.view(-1, *[1] * (P.dim() - 1)) for P in parts]

    def step(self, theta, G, S, gam=None, ext=None):
        """One proximal step of the lasso on Theta (step 1); with frames (gam [L,F], ext = (C, H, s)) then one on
        Gamma (also valid at step 1, since sum_k |kappa_k|^2 <= 1)."""
        lam = self.cfg.lam_floor + F.softplus(self.lam_raw)
        g = theta @ G - S
        if gam is not None:
            g = g + (gam[..., None, None] * ext[0]).sum(1)
        theta = soft(theta - g, lam)
        if gam is None:
            return theta
        Ck, H, sv = ext
        gg = (theta[:, None] * Ck).sum((-2, -1)) + (H * gam[:, None]).sum(-1) - sv
        return theta, soft(gam - gg, lam)

    def rule_head(self, h, x_in, y, m, demo, group, theta, gam, amp, ftab=None):
        """The rule head after one block, in FP32 (FP64 on the test path): phi, kappa, the leave-one-out statistics,
        one proximal step and the read. Returns the new theta, gam and r [L,O,13]. Static shapes and no host sync, so
        the trainer compiles it (inside layer_fn)."""
        c = self.cfg
        with torch.autocast('cuda', enabled=False):
            phi = self.features(h)
            kap = self.frame_colours(h, x_in, amp, ftab) if c.frames else None
            phx = self.features_x(h) if c.rule_heads > 1 else None
            G, S, *ext = self.statistics(phi, y, m, demo, group, kap=kap, phx=phx)
            if phx is not None:
                Gx, Sx = ext[-2:]
                ext = ext[:-2]
                theta, thx = theta[..., :c.phi], theta[..., c.phi:]
            if c.frames:
                theta, gam = self.step(theta, G, S, gam, ext)
            else:
                theta = self.step(theta, G, S)
            r = torch.einsum('lcp,lop->loc', theta, phi)
            if c.frames:
                r = r + (gam[..., None, None] * kap).sum(1)
            if phx is not None:                                # heads 1..: one proximal step each (step 1), no Gamma
                th = thx.unflatten(-1, (-1, c.phi)).transpose(1, 2)               # [L,H-1,13,phi]
                th = soft(th - (th @ Gx - Sx), c.lam_floor + F.softplus(self.lam_raw_x)[:, None, None])
                theta = torch.cat([theta, th.transpose(1, 2).flatten(-2)], -1)
                r = torch.cat([r, torch.einsum('lhcp,lohp->lohc', th, phx).flatten(-2)], -1)
        return theta, gam, r

    def layer(self, h, emb, ek, ev, w, tables, theta, gam, r, x_in, y, m, demo, group, amp, head=True):
        """One block and the rule head after it: the feedback fb = W_fb r of the previous head (FP32, as the caller's
        autocast state), block_step under BF16 autocast (amp), rule_head (FP32). Returns the new h, ek, ev, w, theta,
        gam, r. The trainer compiles this whole layer as one region (layer_fn): h feeds both the head and the next
        layer, so the backward pass sums its two gradients inside the compiled graph instead of a separate add.
        tables = rope_tables() (+ frame_tables() with frames), computed once per segment."""
        fb = self.feedback(r).to(emb.dtype) if head else None
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            h, ek, ev, w = self.block_step(h, emb, fb, ek, ev, w, tables[:2])
        if head:
            theta, gam, r = self.rule_head(h, x_in, y, m, demo, group, theta, gam, amp, tables[2:] or None)
        return h, ek, ev, w, theta, gam, r

    # ------------------------------------------------------------------ segments
    def initial_carry(self, L, device):
        """Fresh lanes: h = init_hidden, zero traces, W, Theta, Gamma and read."""
        c, dt = self.cfg, self.forward_dtype
        z = lambda *s: torch.zeros(L, *s, device=device, dtype=dt)
        tdt = getattr(torch, c.trace_dtype) if dt == torch.float32 else dt
        zt = lambda *s: torch.zeros(L, *s, device=device, dtype=tdt)
        return dict(h=self.init_hidden.to(device).expand(L, self.T, self.d).clone(),
                    ek=zt(self.H, self.T, 2 * self.dh * c.bands), ev=zt(self.H, self.T, 2 * self.dh * c.bands),
                    w=z(self.H, self.dh, self.dh), theta=z(VOCAB, c.phi * c.rule_heads),
                    gam=z(c.frames + 1) if c.frames else None, r=z(self.O, VOCAB * c.rule_heads))

    def segment(self, carry, batch, head=True):
        """One segment from `carry` (None: fresh lanes). batch: inputs, labels [L,T] (labels only on the output
        canvas, pad elsewhere), group [L] (0..ng-1), demo [L] (lanes whose labels enter their group's statistics).
        Returns logits [L,O,13] of the output canvases and the new carry (detached)."""
        c = self.cfg
        x, labels, group = batch['inputs'].long(), batch['labels'], batch['group'].long()
        L = x.shape[0]
        dev, dt = x.device, self.forward_dtype
        demo = batch['demo'].to(dt)
        yl = labels[:, self.O:].long()
        m = (yl != PAD).to(dt)
        y = F.one_hot(yl, VOCAB).to(dt)
        x_in = x[:, :self.O]
        if carry is None:
            carry = self.initial_carry(L, dev)
        h, ek, ev, w = carry['h'], carry['ek'], carry['ev'], carry['w']
        theta, gam, r = carry['theta'], carry['gam'], carry['r']
        amp = self.amp(dev)
        first_grad = c.blocks - c.grad_blocks
        fn = self.layer_fn or self.layer
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            emb = self.embed(x)
        tables = self.rope_tables() + (self.frame_tables() if c.frames else ())   # once per segment
        for i in range(c.blocks):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= first_grad):
                if i == first_grad:                            # the no-grad blocks' head state enters as a constant
                    theta, r = theta.detach(), r.detach()
                    gam = None if gam is None else gam.detach()
                h, ek, ev, w, theta, gam, r = fn(h, emb, ek, ev, w, tables, theta, gam, r, x_in, y, m, demo, group,
                                                 amp, head)
        if c.out_ffn and head:                                 # the last read through one FFN, as a block injects it
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                base = h[:, self.O:] if c.direct_head else self.embed_scale * emb[:, self.O:].to(h.dtype)
                v = rms_norm(base + self.embed_scale * self.feedback(r).to(h.dtype))
                g, a = self.out_gate_up(v).chunk(2, dim=-1)
                h_out = rms_norm(v + self.out_down(F.silu(g) * a))
        with torch.autocast('cuda', enabled=False):
            if c.out_ffn and head:
                logits = self.classifier(h_out.to(dt))
            elif c.direct_head or not head:
                logits = self.classifier(h[:, self.O:].to(dt))
            else:
                logits = self.classifier.bias.expand(h.shape[0], self.O, VOCAB)
            if head and not c.out_ffn:
                logits = logits + self.tau * r[..., :VOCAB]
        new = dict(h=h, ek=ek, ev=ev, w=w, theta=theta, gam=gam, r=r)
        return logits, {k: (None if v is None else v.detach()) for k, v in new.items()}
