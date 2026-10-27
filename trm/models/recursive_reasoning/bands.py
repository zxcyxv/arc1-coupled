"""The bands perception (arc1-coupled arc/model.py Coupled.block_step, unchanged) with TRM's puzzle embedding, in TRM's
harness: one sequence per example, 16 puzzle-embedding tokens + the 30 x 30 input canvas, the answer read at the input
cells (TRM's layout). No rule head, no ACT: every lane runs halt_max_steps segments and is then replaced (all lanes of a
batch together, as arc.train's loops). A segment is `blocks` blocks of the shared bands block, the last `grad_blocks`
with gradients; the carry is the bands state (h, the two traces, W), detached between segments.

Positions: rope '2d' (the bands block's learned angles), input cell i at (i // 30, i % 30), puzzle token j at (-1, j);
rope '1d', TRM's fixed RoPE (base 10000) over the sequence index 0..915, puzzle tokens first.
Tokens: TRM's (0 pad, 1 EOS, 2..11 colours) mapped to the bands model's (1 pad, 12 EOS, 2..11); the classifier's 13
outputs are read back in TRM's 12 classes. The puzzle embedding is TRM's sparse table (zero init, sign-SGD); in training
each grad-accumulation chunk fills its own rows of the per-step local table (chunk_offset), so the chunks' gradients
add up in one local table for TRM's optimizer."""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # the repository root: arc.*
from arc.model import Coupled, ModelConfig  # noqa: E402

TRM_TO_MODEL = (1, 12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)     # TRM token -> model token; also the logits' read-back order
_BLOCK = {}                                                # compiled Coupled.block_step (one per process)


def block_fn(compile_):
    if not compile_:
        return Coupled.block_step
    if 'fn' not in _BLOCK:
        import torch._dynamo.config as dc
        import torch._inductor.config as ic
        ic.triton.persistent_reductions = False
        dc.recompile_limit = max(dc.recompile_limit, 32)  # one recompile per module instance (the EMA copy)
        _BLOCK['fn'] = torch.compile(Coupled.block_step, dynamic=False)
    return _BLOCK['fn']


@dataclass
class BandsInnerCarry:
    h: torch.Tensor
    ek: torch.Tensor
    ev: torch.Tensor
    w: torch.Tensor


@dataclass
class BandsCarry:
    inner_carry: BandsInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class BandsConfig(BaseModel):
    batch_size: int
    seq_len: int
    num_puzzle_identifiers: int
    vocab_size: int
    puzzle_emb_ndim: int = 512
    puzzle_emb_len: int = 16
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4.0
    blocks: int = 16
    grad_blocks: int = 10
    bands: int = 4
    trace_dtype: str = 'bfloat16'
    rope: str = '2d'                  # '2d': learned angles over (row, col); '1d': TRM's fixed RoPE over the sequence index
    rope_base: float = 10000.0        # '1d': frequencies rope_base^(-2i / head_dim), as TRM's RotaryEmbedding
    halt_max_steps: int = 16
    compile: bool = True
    causal: bool = False


class SparsePuzzleEmbedding(nn.Module):
    """TRM's CastedSparseEmbedding (same buffers, so TRM's CastedSparseEmbeddingSignSGD_Distributed updates it) with a
    chunk offset into the local table."""
    def __init__(self, num, dim, batch_size):
        super().__init__()
        self.weights = nn.Buffer(torch.zeros(num, dim), persistent=True)
        self.local_weights = nn.Buffer(torch.zeros(batch_size, dim, requires_grad=True), persistent=False)
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, ids, offset=0):
        if not self.training:
            return self.weights[ids]
        n = ids.shape[0]
        with torch.no_grad():
            self.local_weights[offset:offset + n].copy_(self.weights[ids])
            self.local_ids[offset:offset + n].copy_(ids)
        return self.local_weights[offset:offset + n]


class Bands(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        c = self.config = BandsConfig(**config_dict)
        assert c.seq_len == 900 and c.vocab_size == 12 and c.puzzle_emb_ndim == c.hidden_size
        self.core = Coupled(ModelConfig(canvas=30, d=c.hidden_size, heads=c.num_heads, mlp_expansion=c.expansion,
                                        blocks=c.blocks, grad_blocks=c.grad_blocks, bands=c.bands, frames=0,
                                        fused_bands=True, trace_dtype=c.trace_dtype))
        P = c.puzzle_emb_len
        self.core.T = T = P + c.seq_len
        if c.rope == '1d':                # TRM: position = sequence index (puzzle tokens first), fixed frequencies
            dh = c.hidden_size // c.num_heads
            th = self.core.block.rope_theta                                   # [H, dh / 2, (row, col)]
            with torch.no_grad():
                th[..., 0] = 1.0 / (c.rope_base ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh)).to(th.dtype)
                th[..., 1] = 0.0
            th.requires_grad_(False)                                         # no gradient, no EMA, no update
            row, col = torch.arange(T).float(), torch.zeros(T)
        elif c.rope == '2d':              # learned angles; input cell i at (i // 30, i % 30), puzzle token j at (-1, j)
            t = torch.arange(c.seq_len)
            row = torch.cat([torch.full((P,), -1.0), (t // 30).float()])
            col = torch.cat([torch.arange(P).float(), (t % 30).float()])
        else:
            raise ValueError(f'rope {c.rope}')
        self.core.register_buffer('pos_row', row, persistent=False)
        self.core.register_buffer('pos_col', col, persistent=False)
        self.puzzle_emb = SparsePuzzleEmbedding(c.num_puzzle_identifiers, c.puzzle_emb_ndim, c.batch_size)
        self.register_buffer('to_model', torch.tensor(TRM_TO_MODEL), persistent=False)

    # ---------------------------------------------------------------- carry
    def _fresh(self, B, device):
        k = self.core.initial_carry(B, device)
        return BandsInnerCarry(k['h'], k['ek'], k['ev'], k['w'])

    def initial_carry(self, batch):
        B = batch['inputs'].shape[0]
        dev = batch['inputs'].device
        return BandsCarry(self._fresh(B, dev), torch.zeros(B, dtype=torch.int32, device=dev),
                          torch.ones(B, dtype=torch.bool, device=dev), {k: torch.empty_like(v) for k, v in batch.items()})

    def _reset(self, halted, carry):
        f = self._fresh(halted.shape[0], halted.device)
        pick = lambda new, old: torch.where(halted.view(-1, *[1] * (old.dim() - 1)), new, old)
        return BandsInnerCarry(*(pick(getattr(f, n), getattr(carry, n)) for n in ('h', 'ek', 'ev', 'w')))

    # ---------------------------------------------------------------- one segment
    def forward(self, carry: BandsCarry, batch: Dict[str, torch.Tensor], chunk_offset: int = 0):
        c, m = self.config, self.core
        inner = self._reset(carry.halted, carry.inner_carry)
        steps = torch.where(carry.halted, 0, carry.steps)
        data = {k: torch.where(carry.halted.view(-1, *[1] * (batch[k].dim() - 1)), batch[k], v)
                for k, v in carry.current_data.items()}

        x = self.to_model[data['inputs'].long()]
        amp = m.amp(x.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            emb = m.embed(x)
        pe = self.puzzle_emb(data['puzzle_identifiers'].long(), chunk_offset).to(emb.dtype)
        pe = F.pad(pe, (0, c.puzzle_emb_len * c.hidden_size - pe.shape[-1])).view(-1, c.puzzle_emb_len, c.hidden_size)
        emb = torch.cat([pe, emb], 1)
        tables = m.rope_tables()
        fn = block_fn(c.compile)
        h, ek, ev, w = inner.h, inner.ek, inner.ev, inner.w
        first_grad = c.blocks - c.grad_blocks
        for i in range(c.blocks):
            with torch.set_grad_enabled(torch.is_grad_enabled() and i >= first_grad):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                    h, ek, ev, w = fn(m, h, emb, None, ek, ev, w, tables)
        with torch.autocast('cuda', enabled=False):
            logits = m.classifier(h[:, c.puzzle_emb_len:].float())[..., self.to_model]
        new_inner = BandsInnerCarry(h.detach(), ek.detach(), ev.detach(), w.detach())
        with torch.no_grad():
            steps = steps + 1
            halted = steps >= c.halt_max_steps
        q = torch.zeros(h.shape[0], device=h.device)
        return BandsCarry(new_inner, steps, halted, data), dict(logits=logits, q_halt_logits=q, q_continue_logits=q)
