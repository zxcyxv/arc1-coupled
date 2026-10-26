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

frames = K > 0 (cmap's Gamma term): F = K + 1 frame colours per output cell, kappa_0 = the colour histogram of the
lane's input grid, kappa_k = a softmax read of the input grid's colour cells from the output cell (learned 2D RoPE on
canvas coordinates), all / sqrt(F) so sum_k |kappa_k|^2 <= 1; r(o) = Theta phi(o) + sum_k Gamma_k kappa_k(o), one
lasso over (Theta, Gamma), one proximal step on each block in turn. Gamma has no colour index.

A segment runs `blocks` blocks from the carry, the last `grad_blocks` with gradients, and returns the detached carry.
With fused_bands (CUDA, FP32 forward) the trace update runs as one Triton kernel each way (arc/bands_kernel.py): the
same formulas with less memory traffic. The trainer compiles block_step and the rule head after it (rule_head,
inject) separately, each with static shapes.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .bands_kernel import bands_trace
except ImportError:                                      # no Triton: the PyTorch path only
    bands_trace = None

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
    lam_init: float = 1e-3            # lasso threshold (learned, softplus)
    tau_init: float = 1.0             # read gain (learned)
    frames: int = 0                   # K > 0: Gamma over K softmax frames + the grid histogram
    frame_dim: int = 32               # q/k width of a frame (frame_dim / 4 frequencies per axis)
    frame_bf16: bool = True           # with amp on CUDA: the frames' softmax read as BF16 flash attention
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
        self.register_buffer('pos_row', (t // C).to(rope_dtype), persistent=False)     # rows 0..2C-1 over the lane
        self.register_buffer('pos_col', (t % C).to(rope_dtype), persistent=False)
        self.w_phi = nn.Linear(d, c.phi, bias=False)
        self.w_fb = nn.Linear(VOCAB, d, bias=False)
        nn.init.zeros_(self.w_fb.weight)                       # the feedback path opens by learning
        self.lam_raw = nn.Parameter(torch.tensor(math.log(math.expm1(c.lam_init))))
        self.tau = nn.Parameter(torch.tensor(c.tau_init))
        if c.frames:
            K, E = c.frames, c.frame_dim
            self.w_fq = nn.Linear(d, K * E, bias=False)
            self.w_fk = nn.Linear(d, K * E, bias=False)
            p = torch.logspace(math.log10(2.0), math.log10(4.0 * C), E // 4)     # periods 2 .. 4 C cells
            self.frame_omega = nn.Parameter((2 * math.pi / p)[None, None].repeat(K, 2, 1))   # [K, axis, freq]
            r, cc = torch.meshgrid(torch.arange(C), torch.arange(C), indexing='ij')
            self.register_buffer('frame_pos', torch.stack([r, cc], -1).reshape(-1, 2).float(), persistent=False)
        self.to(self.forward_dtype)
        self.block_fn = None                                   # compiled block_step (trainer)
        self.head_fn = None                                    # compiled rule_head (trainer)
        self.inject_fn = None                                  # compiled inject (trainer)

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

    def bands_step(self, q, k, v, ek, ev, w):
        """One block of the bands memory in FP32 (FP64 on the test path). ek, ev [B,H,T,2RD] hold (Re, Im) x R bands
        of the k~ and v traces; w [B,H,D,D]. The bands are merged before the token sums, so S is two [D,T] @ [T,D]
        products for any R. Returns read [B,H,T,D] and the new (ek, ev, w)."""
        cfg = self.cfg
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            unit = lambda x: x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + cfg.eps)
            angles = self.rope_angles().to(dtype)[None]
            tables = angles.cos(), angles.sin()
            qr, kr = self.rope(unit(q.to(dtype)), tables), self.rope(unit(k.to(dtype)), tables)
            v = v.to(dtype)
            B, H, T, D = v.shape
            store = getattr(torch, cfg.trace_dtype) if dtype == torch.float32 else dtype
            if cfg.fused_bands and v.is_cuda and dtype == torch.float32:
                ek, ev, k_mix, v_mix = bands_trace(kr, v, ek, ev, torch.stack(self.block.bands()), store)
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
            s = v.transpose(-1, -2) @ k_mix + v_mix.transpose(-1, -2) @ kr
            eta = self.block.eta().to(dtype)[None, :, None, None]
            w = (1 - eta) * w.to(dtype) + eta * s
            read = qr @ (v.transpose(-1, -2) @ kr + w).transpose(-1, -2)
        return read, ek, ev, w

    def block_step(self, h, inj, ek, ev, w):
        """One recurrent block: the new hidden state and bands state."""
        L = self.block
        u = h + self.embed_scale * inj
        B, T, _ = u.shape
        heads = lambda x: x.reshape(B, T, self.H, self.dh).transpose(1, 2)
        q, k, v = heads(L.q_proj(u)), heads(L.k_proj(u)), heads(L.v_proj(u))
        read, ek, ev, w = self.bands_step(q, k, v, ek, ev, w)
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

    def frame_rope(self, x):
        """x [L,K,O,E] rotated per (axis, frequency) pair by omega_k,axis,f times the cell's canvas row / column."""
        K, O, E = x.shape[1], x.shape[2], x.shape[3]
        ang = torch.einsum('oa,kaf->koaf', self.frame_pos.to(x.dtype), self.frame_omega.to(x.dtype)).reshape(K, O, E // 2)
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

    def frame_colours(self, h, x_in, amp=None):
        """kappa [L,F,O,13] of the output-canvas cells from h [L,T,d] and the input-canvas tokens x_in [L,O]; colour
        cells are tokens 2..11 (pad and EOS are never read). The softmax read runs through SDPA (no [O,O] tensor kept
        for the backward pass on CUDA). amp: self.amp(h.device), passed by segment (None: computed here)."""
        c = self.cfg
        L, O, K, E = h.shape[0], self.O, c.frames, c.frame_dim
        dt = self.w_fq.weight.dtype
        live = (x_in >= 2) & (x_in <= 11)
        col = F.one_hot(x_in, VOCAB).to(dt) * live[..., None]
        bf16 = (self.amp(h.device) if amp is None else amp) and c.frame_bf16
        proj = ((lambda lin, x: F.linear(x.bfloat16(), lin.weight.bfloat16()).to(dt)) if bf16
                else (lambda lin, x: lin(x.to(dt))))
        q = self.frame_rope(proj(self.w_fq, h[:, O:]).view(L, O, K, E).transpose(1, 2))
        k = self.frame_rope(proj(self.w_fk, h[:, :O]).view(L, O, K, E).transpose(1, 2))
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

    def statistics(self, phi, y, m, demo, group, ng=None, kap=None):
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
        nl = w.sum(-1)
        L = group.shape[0]
        others = (group[:, None] == group[None, :]) & ~torch.eye(L, dtype=torch.bool, device=group.device)
        others = others.to(nl.dtype)                           # [L,L]: the other lanes of each lane's group
        n = (others @ nl).clamp_min(1)                         # own lane left out (a non-demonstration adds nothing)
        return [(others @ P.reshape(L, -1)).view_as(P) / n.view(-1, *[1] * (P.dim() - 1)) for P in parts]

    def step(self, theta, G, S, gam=None, ext=None):
        """One proximal step of the lasso on Theta (step 1); with frames (gam [L,F], ext = (C, H, s)) then one on
        Gamma (also valid at step 1, since sum_k |kappa_k|^2 <= 1)."""
        lam = F.softplus(self.lam_raw)
        g = theta @ G - S
        if gam is not None:
            g = g + (gam[..., None, None] * ext[0]).sum(1)
        theta = soft(theta - g, lam)
        if gam is None:
            return theta
        Ck, H, sv = ext
        gg = (theta[:, None] * Ck).sum((-2, -1)) + (H * gam[:, None]).sum(-1) - sv
        return theta, soft(gam - gg, lam)

    def rule_head(self, h, x_in, y, m, demo, group, theta, gam, amp):
        """The rule head after one block, in FP32 (FP64 on the test path): phi, kappa, the leave-one-out statistics,
        one proximal step and the read. Returns the new theta, gam and r [L,O,13]. Static shapes and no host sync, so
        the trainer compiles it (head_fn, like block_fn)."""
        c = self.cfg
        with torch.autocast('cuda', enabled=False):
            phi = self.features(h)
            kap = self.frame_colours(h, x_in, amp) if c.frames else None
            G, S, *ext = self.statistics(phi, y, m, demo, group, kap=kap)
            if c.frames:
                theta, gam = self.step(theta, G, S, gam, ext)
            else:
                theta = self.step(theta, G, S)
            r = torch.einsum('lcp,lop->loc', theta, phi)
            if c.frames:
                r = r + (gam[..., None, None] * kap).sum(1)
        return theta, gam, r

    def inject(self, emb, r):
        """The next block's injection: the embeddings, plus W_fb r on the output canvas (compiled as inject_fn)."""
        return torch.cat([emb[:, :self.O], emb[:, self.O:] + self.w_fb(r).to(emb.dtype)], 1)

    # ------------------------------------------------------------------ segments
    def initial_carry(self, L, device):
        """Fresh lanes: h = init_hidden, zero traces, W, Theta, Gamma and read."""
        c, dt = self.cfg, self.forward_dtype
        z = lambda *s: torch.zeros(L, *s, device=device, dtype=dt)
        tdt = getattr(torch, c.trace_dtype) if dt == torch.float32 else dt
        zt = lambda *s: torch.zeros(L, *s, device=device, dtype=tdt)
        return dict(h=self.init_hidden.to(device).expand(L, self.T, self.d).clone(),
                    ek=zt(self.H, self.T, 2 * self.dh * c.bands), ev=zt(self.H, self.T, 2 * self.dh * c.bands),
                    w=z(self.H, self.dh, self.dh), theta=z(VOCAB, c.phi),
                    gam=z(c.frames + 1) if c.frames else None, r=z(self.O, VOCAB))

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
        fn = self.block_fn or self.block_step
        head_fn = self.head_fn or self.rule_head
        inject_fn = self.inject_fn or self.inject
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            emb = self.embed(x)
        for i in range(c.blocks):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= first_grad):
                if i == first_grad:                            # the no-grad blocks' head state enters as a constant
                    theta, r = theta.detach(), r.detach()
                    gam = None if gam is None else gam.detach()
                inj = inject_fn(emb, r) if head else emb
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                    h, ek, ev, w = fn(h, inj, ek, ev, w)
                if head:
                    theta, gam, r = head_fn(h, x_in, y, m, demo, group, theta, gam, amp)
        with torch.autocast('cuda', enabled=False):
            logits = self.classifier(h[:, self.O:].to(dt))
            if head:
                logits = logits + self.tau * r
        new = dict(h=h, ek=ek, ev=ev, w=w, theta=theta, gam=gam, r=r)
        return logits, {k: (None if v is None else v.detach()) for k, v in new.items()}
