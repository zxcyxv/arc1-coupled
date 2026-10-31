"""CPU/FP64 checks of lt's three-factor W (ModelConfig.three_factor) and learned initial W (w0_learned).

- beta = 0 (the start) and W0 = 0 (the start) give the bands EMA model bit for bit, gradients of the shared parameters
  included.
- w0_learned is the EMA started at W0: same logits as the plain model whose carry holds W = W0, and the same gradient
  of a one-segment loss; later segments still reach W0.
- Two segments of 3 blocks are one segment of 6 (C, the previous (C^, V), the block count and W~ carry exactly), the
  head losses add up, and delta_n = -(1 - g) C^(h_{n-2}) + g V(h_{n-1}) - V(h_{n-2}) with delta = 0 at a puzzle's
  first block.
- The head losses reach only the heads; the CE reaches beta but not the heads.
- The harness trains it and resumes exactly (real data, tiny model).
Run: python -m unittest tests.test_lt_three_factor"""
import contextlib
import io
import math
import tempfile
import unittest

import torch

from lt import train
from lt.model import LT, ModelConfig


def cfg(**kw):
    base = dict(batch_size=3, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, blocks_per_seg=3, loops=2, amp=False,
                forward_dtype='float64', kernel='torch', bands=4, tag_bound=0.0, outer_eta=0.125, outer_eta_learn=True,
                outer_eta_cap=True, tf_head_dim=8)
    base.update(kw)
    return ModelConfig(**base)


def batch(seed, n=3):
    g = torch.Generator().manual_seed(seed)
    return dict(inputs=torch.randint(1, 11, (n, 81), generator=g), labels=torch.randint(2, 11, (n, 81), generator=g),
                puzzle_identifiers=torch.zeros(n, dtype=torch.int32))


def run(m, segs, seed=0):
    """segs segments (loops = 2: lanes are replaced every other segment); logits and the sum of their squares."""
    carry = m.initial_carry(batch(seed))
    logits, loss = [], 0
    for s in range(segs):
        carry, out = m(carry, batch(seed + s))
        logits.append(out)
        loss = loss + out.square().mean()
    return logits, loss, carry


def perturb(m, seed=1):
    """Nonzero beta, lam away from its init, random heads and W0: every three-factor path is live."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if m.config.three_factor:
            m.block.tf_beta.copy_(torch.randn(m.H, generator=g, dtype=torch.float64) * 30)
            m.block.tf_lam_raw.add_(torch.randn(m.H, generator=g, dtype=torch.float64))
            for p in m.tf_heads.parameters():
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * 0.5)
        if m.config.w0_learned:
            m.w0_table.copy_(torch.randn(m.w0_table.shape, generator=g, dtype=torch.float64) * 0.3)


class StartsUnchanged(unittest.TestCase):
    def check_equal(self, base=None, **flags):
        out = []
        for kw in (base or {}, dict(base or {}, **flags)):
            torch.manual_seed(0)
            m = LT(cfg(**kw)).double().train()
            logits, loss, _ = run(m, 5)
            loss.backward()
            out.append((logits, {n: p.grad for n, p in m.named_parameters()}))
        (la, ga), (lb, gb) = out
        for x, y in zip(la, lb):
            self.assertTrue(torch.equal(x, y))
        for n, g in ga.items():
            self.assertTrue(torch.equal(g, gb[n]), n)

    def test_beta0_is_the_bands_model(self):
        self.check_equal(three_factor=True)

    def test_beta0_with_grad_blocks(self):
        self.check_equal(dict(grad_blocks=2), three_factor=True)

    def test_w0_zero_is_the_bands_model(self):
        self.check_equal(w0_learned=True)

    def test_delta_starts_at_zero(self):
        torch.manual_seed(0)
        m = LT(cfg(three_factor=True)).double().train()
        run(m, 3)
        self.assertLess(float(m.tf_stats['delta_abs'].max()), 1e-9)   # the bias -ln 2 is rounded in FP32


class LearnedW0(unittest.TestCase):
    def test_is_the_ema_started_at_w0(self):
        torch.manual_seed(0)
        m = LT(cfg(w0_learned=True, loops=50)).double().train()
        perturb(m)
        torch.manual_seed(0)
        ref = LT(cfg(loops=50)).double().train()
        ref.load_state_dict({k: v for k, v in m.state_dict().items() if k != 'w0_table'})
        b = batch(0)
        B, H, D = 3, m.H, m.dh
        w0 = m.w0_table[0].detach().view(1, H, D, D).expand(B, H, D, D).clone().requires_grad_(True)
        z = ref.initial_state(B, torch.device('cpu'), torch.float64)
        h0 = ref.init_hidden.expand(B, ref.n_pos, ref.d).clone()
        rc = ref.initial_carry(b)
        rc.halted = torch.zeros(B, dtype=torch.bool)            # not fresh: the carried state is used as it is
        rc.data, rc.hidden = b, h0
        rc.clock, rc.key_trace, rc.value_trace, rc.stdp_w, rc.prev = z[0], z[1], z[2], w0, z[4]
        mc = m.initial_carry(b)
        mc, lm = m(mc, b)
        rc, lr = ref(rc, b)
        torch.testing.assert_close(lm, lr, rtol=1e-10, atol=1e-12)
        lm.square().mean().backward()
        lr.square().mean().backward()
        torch.testing.assert_close(m.w0_table.grad[0].view(H, D, D), w0.grad.sum(0), rtol=1e-9, atol=1e-14)
        for _ in range(2):                                        # later segments: same logits
            mc, lm = m(mc, b)
            rc, lr = ref(rc, b)
            torch.testing.assert_close(lm, lr, rtol=1e-10, atol=1e-12)
        m.w0_table.grad = None
        lm.square().mean().backward()                             # segment 3 alone still reaches W0
        self.assertGreater(float(m.w0_table.grad.abs().max()), 0.0)


class Concatenation(unittest.TestCase):
    def models(self, **kw):
        torch.manual_seed(0)
        a = LT(cfg(three_factor=True, w0_learned=True, loops=50, **kw)).double().train()
        perturb(a)
        b = LT(cfg(three_factor=True, w0_learned=True, loops=50, **dict(kw, blocks_per_seg=6))).double().train()
        b.load_state_dict(a.state_dict())
        return a, b

    def test_two_segments_are_one(self):
        a, b = self.models()
        x = batch(0)
        ca = a.initial_carry(x)
        ca, _ = a(ca, x)
        loss_a = a.tf_loss
        ca, la = a(ca, x)
        loss_a = loss_a + a.tf_loss
        cb, lb = b(b.initial_carry(x), x)
        torch.testing.assert_close(la, lb, rtol=1e-11, atol=1e-12)
        for f in ('stdp_w', 'tf_elig', 'tf_prev', 'key_trace', 'value_trace', 'hidden'):
            torch.testing.assert_close(getattr(ca, f), getattr(cb, f), rtol=1e-11, atol=1e-12, msg=f)
        torch.testing.assert_close(loss_a, 2 * b.tf_loss, rtol=1e-11, atol=1e-12)
        self.assertGreater(float(b.tf_stats['delta_abs'].min()), 0.0)

    def test_delta_is_the_td_error_of_the_previous_block(self):
        _, b = self.models()
        seen = []
        orig = b.tf_input

        def spy(tf, h, i, store):
            out = orig(tf, h, i, store)
            seen.append((h.detach().clone(), out[1].clone()))
            return out
        b.tf_input = spy
        b(b.initial_carry(batch(0)), batch(0))
        g = b.config.tf_gamma
        board = lambda h: tuple(x.mean(-1) for x in b.tf_head(h))
        self.assertTrue(torch.equal(seen[0][1], torch.zeros(3, dtype=torch.float64)))   # fresh lanes
        for n in range(1, len(seen)):
            (cp, vp), (_, vc) = board(seen[n - 1][0]), board(seen[n][0])
            torch.testing.assert_close(seen[n][1], -(1 - g) * cp + g * vc - vp, rtol=1e-12, atol=1e-14)

    def test_write_formula(self):
        a, _ = self.models()
        g = torch.Generator().manual_seed(3)
        B, H, D = 3, a.H, a.dh
        r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
        w_prev, s, elig, w0 = r(B, H, D, D), r(B, H, D, D), r(B, H, D, D), r(B, H, D, D)
        delta, n = r(B), torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
        eta = a.block.eta(a.config).detach().view(1, H, 1, 1)
        beta, lam = a.block.tf_beta.detach().view(1, H, 1, 1), a.block.tf_lam().detach().view(1, H, 1, 1)
        with torch.no_grad():
            w, m, (c, *_) = a.tf_write((1 - eta) * w_prev + eta * s, s, (elig, delta, n, w0, None), torch.float64)
        w_ref = (1 - eta) * w_prev + eta * s + eta * beta * delta.view(B, 1, 1, 1) * elig
        torch.testing.assert_close(w, w_ref, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(m, w_ref + (1 - eta) ** n.view(B, 1, 1, 1) * w0, rtol=1e-12, atol=1e-13)
        torch.testing.assert_close(c, lam * elig + (1 - lam) * s, rtol=1e-12, atol=1e-13)


class GradientPaths(unittest.TestCase):
    def test_heads_and_main_path_are_separate(self):
        torch.manual_seed(0)
        m = LT(cfg(three_factor=True, loops=50)).double().train()
        perturb(m)
        x = batch(0)
        c, _ = m(m.initial_carry(x), x)
        c, logits = m(c, x)
        heads = {n for n, _ in m.named_parameters() if n.startswith('tf_heads.')}
        m.tf_loss.backward(retain_graph=True)
        for n, p in m.named_parameters():
            if n in heads:
                self.assertGreater(float(p.grad.abs().max()), 0.0, n)
            else:
                self.assertTrue(p.grad is None or float(p.grad.abs().max()) == 0.0, n)
        m.zero_grad(set_to_none=True)
        logits.square().mean().backward()
        for n, p in m.named_parameters():
            if n in heads:
                self.assertTrue(p.grad is None or float(p.grad.abs().max()) == 0.0, n)
        self.assertGreater(float(m.block.tf_beta.grad.abs().min()), 0.0)
        self.assertGreater(float(m.block.tf_lam_raw.grad.abs().min()), 0.0)


def tiny_run(out, steps, **over):
    c = dict(out_dir=out, max_steps=steps, hidden_size=16, num_heads=2, puzzle_emb_ndim=16, blocks_per_seg=2,
             loops=3, global_batch_size=4, test_size=8, epochs=2, eval_interval=1, lr_warmup_steps=2, compile=False,
             dataloader_workers=0, save_every_steps=0, log_every=1000, amp=False, kernel='torch', bands=4,
             tag_bound=0.0, outer_eta=0.125, outer_eta_learn=True, outer_eta_cap=True, three_factor=True,
             w0_learned=True, tf_head_dim=8, tf_noise=0.1)
    c.update(over)
    with contextlib.redirect_stdout(io.StringIO()):
        return train.main(c)


class Harness(unittest.TestCase):
    def test_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            tiny_run(a, 7)
            tiny_run(b, 4)
            tiny_run(b, 7)
            ca = torch.load(f'{a}/step_7.pt', weights_only=False)
            cb = torch.load(f'{b}/step_7.pt', weights_only=False)
            for key in ('raw_model', 'model'):
                for name, value in ca[key].items():
                    self.assertTrue(torch.equal(value, cb[key][name]), f'{key}.{name}')
            self.assertIn('tf_heads.0.weight', ca['raw_model'])
            self.assertIn('w0_table', ca['raw_model'])


if __name__ == '__main__':
    unittest.main()
