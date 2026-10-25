"""The coupled model's reductions and information paths. python -m unittest tests.test_model"""
import unittest

import torch
import torch.nn.functional as F

from arc.model import PAD, VOCAB, Coupled, ModelConfig


def small(**kw):
    return ModelConfig(**dict(dict(canvas=3, d=16, heads=2, blocks=4, grad_blocks=2, phi=8, bands=2,
                                   band_periods=(2.0, 6.0), dtype='float64'), **kw))


def batch(cfg, groups=2, demos=3, seed=0):
    """groups x (demos demonstration lanes + 1 query lane, the last of its group)."""
    g = torch.Generator().manual_seed(seed)
    L, O = demos + 1, cfg.canvas ** 2
    n = groups * L
    inputs = torch.cat([torch.randint(1, VOCAB, (n, O), generator=g), torch.zeros(n, O, dtype=torch.long)], -1)
    labels = torch.cat([torch.full((n, O), PAD), torch.randint(1, VOCAB, (n, O), generator=g)], -1)
    query = (torch.arange(n) % L) == L - 1
    return dict(inputs=inputs, labels=labels, group=torch.arange(n) // L, demo=~query, query=query)


def model(cfg, seed=1):
    torch.manual_seed(seed)
    m = Coupled(cfg).eval()
    with torch.no_grad():
        m.w_fb.weight.normal_(0, 0.5)                          # open the feedback path
        m.lam_raw.fill_(-6.0)
    return m


class Reductions(unittest.TestCase):
    def test_no_demonstrations_is_bands(self):
        for frames in (0, 3):
            cfg = small(frames=frames, frame_dim=8)
            m = model(cfg)
            b = batch(cfg)
            b['demo'][:] = False
            with torch.no_grad():
                a, carry = m.segment(None, b)
                ref, _ = m.segment(None, b, head=False)
            self.assertEqual(float(carry['theta'].abs().max()), 0.0)
            if frames:
                self.assertEqual(float(carry['gam'].abs().max()), 0.0)
            torch.testing.assert_close(a, ref, rtol=0, atol=0)

    def test_query_answer_never_read(self):
        for frames in (0, 3):
            cfg = small(frames=frames, frame_dim=8)
            m = model(cfg)
            b = batch(cfg)
            b2 = {k: v.clone() for k, v in b.items()}
            q = b['query']
            b2['labels'][q] = torch.randint(1, VOCAB, b2['labels'][q].shape)
            with torch.no_grad():
                torch.testing.assert_close(m.segment(None, b)[0][q], m.segment(None, b2)[0][q], rtol=0, atol=0)

    def test_demonstrations_reach_the_query_only_through_the_head(self):
        cfg = small(frames=3, frame_dim=8)
        m = model(cfg)
        b = batch(cfg)
        b2 = {k: v.clone() for k, v in b.items()}
        q = b['query']
        b2['inputs'][0] = torch.randint(1, VOCAB, b2['inputs'][0].shape)     # demonstration 0 of group 0
        b2['labels'][0] = torch.randint(1, VOCAB, b2['labels'][0].shape)
        with torch.no_grad():
            self.assertFalse(torch.allclose(m.segment(None, b)[0][q][0], m.segment(None, b2)[0][q][0]))
            torch.testing.assert_close(m.segment(None, b)[0][q][1], m.segment(None, b2)[0][q][1], rtol=0, atol=0)
            m.w_fb.weight.zero_()
            m.tau.zero_()
            torch.testing.assert_close(m.segment(None, b)[0][q], m.segment(None, b2)[0][q], rtol=0, atol=0)

    def test_carry_continues_the_recurrence(self):
        """Two segments of 4 blocks from the carry are one segment of 8 blocks (the carry holds every state)."""
        cfg4, cfg8 = small(frames=3, frame_dim=8, blocks=4), small(frames=3, frame_dim=8, blocks=8)
        m4, m8 = model(cfg4), model(cfg8)
        m8.load_state_dict(m4.state_dict())
        b = batch(cfg4)
        with torch.no_grad():
            ref, _ = m8.segment(None, b)
            _, c = m4.segment(None, b)
            out, _ = m4.segment(c, b)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_gradients_only_through_the_last_blocks(self):
        cfg = small(frames=3, frame_dim=8)
        m = model(cfg)
        m.train()
        b = batch(cfg)
        logits, carry = m.segment(None, b)
        logits.sum().backward()
        self.assertTrue(all(v is None or not v.requires_grad for v in carry.values()))
        self.assertGreater(float(m.block.q_proj.weight.grad.abs().sum()), 0.0)


class Lasso(unittest.TestCase):
    def test_leave_one_out(self):
        cfg = small()
        m = model(cfg)
        g = torch.Generator().manual_seed(3)
        b = batch(cfg, groups=3)
        n, O = len(b['group']), cfg.canvas ** 2
        phi = F.normalize(torch.randn(n, O, cfg.phi, generator=g, dtype=torch.float64), dim=-1)
        y = F.one_hot(torch.randint(0, VOCAB, (n, O), generator=g), VOCAB).double()
        mk = (torch.rand(n, O, generator=g) < 0.7).double()
        demo = b['demo'].double()
        G, S = m.statistics(phi, y, mk, demo, b['group'], 3)
        for j in range(n):
            keep = demo * (b['group'] == b['group'][j]).double()
            keep[j] = 0
            w = mk * keep[:, None]
            tot = w.sum()
            torch.testing.assert_close(G[j], torch.einsum('lo,lop,loq->pq', w, phi, phi) / tot)
            torch.testing.assert_close(S[j], torch.einsum('lo,loc,lop->cp', w, y, phi) / tot)
            self.assertAlmostEqual(float(torch.diagonal(G[j]).sum()), 1.0, places=12)   # tr G = 1

    def test_unit_step_decreases_the_lasso(self):
        cfg = small()
        m = model(cfg)
        with torch.no_grad():
            m.lam_raw.fill_(-7.0)
        g = torch.Generator().manual_seed(4)
        phi = F.normalize(torch.randn(50, cfg.phi, generator=g, dtype=torch.float64), dim=-1)
        y = F.one_hot(torch.randint(0, VOCAB, (50,), generator=g), VOCAB).double()
        G, S = phi.T @ phi / 50, y.T @ phi / 50
        lam = float(F.softplus(m.lam_raw))
        obj = lambda th: float(0.5 * ((y - phi @ th.T) ** 2).sum(-1).mean() + lam * th.abs().sum())
        th = torch.zeros(VOCAB, cfg.phi, dtype=torch.float64)
        prev = obj(th)
        with torch.no_grad():
            for _ in range(40):
                th = m.step(th, G, S)
                cur = obj(th)
                self.assertLessEqual(cur, prev + 1e-12)
                prev = cur
        self.assertLess(prev, obj(torch.zeros_like(th)))

    def test_block_steps_decrease_the_joint_lasso(self):
        cfg = small(frames=3, frame_dim=8)
        m = model(cfg)
        with torch.no_grad():
            m.lam_raw.fill_(-7.0)
        g = torch.Generator().manual_seed(5)
        n, Fn = 60, cfg.frames + 1
        phi = F.normalize(torch.randn(n, cfg.phi, generator=g, dtype=torch.float64), dim=-1)
        kap = torch.softmax(3 * torch.randn(Fn, n, VOCAB, generator=g, dtype=torch.float64), -1) / Fn ** 0.5
        y = F.one_hot(torch.randint(0, VOCAB, (n,), generator=g), VOCAB).double()
        G, S = (phi.T @ phi / n)[None], (y.T @ phi / n)[None]
        ext = (torch.einsum('koc,op->kcp', kap, phi)[None] / n, torch.einsum('koc,joc->kj', kap, kap)[None] / n,
               torch.einsum('oc,koc->k', y, kap)[None] / n)
        lam = float(F.softplus(m.lam_raw))
        obj = lambda th, ga: float(0.5 * ((y - phi @ th[0].T - torch.einsum('k,koc->oc', ga[0], kap)) ** 2).sum(-1).mean()
                                   + lam * (th.abs().sum() + ga.abs().sum()))
        th, ga = torch.zeros(1, VOCAB, cfg.phi, dtype=torch.float64), torch.zeros(1, Fn, dtype=torch.float64)
        prev = obj(th, ga)
        with torch.no_grad():
            for _ in range(40):
                th, ga = m.step(th, G, S, ga, ext)
                cur = obj(th, ga)
                self.assertLessEqual(cur, prev + 1e-12)
                prev = cur
        self.assertGreater(float(ga.abs().max()), 0.0)

    def test_frame_colours_are_bounded_and_colour_only(self):
        cfg = small(frames=3, frame_dim=8)
        m = model(cfg)
        b = batch(cfg)
        O = cfg.canvas ** 2
        with torch.no_grad():
            kap = m.frame_colours(torch.randn(len(b['group']), 2 * O, cfg.d, dtype=torch.float64), b['inputs'][:, :O])
        self.assertLessEqual(float(kap.square().sum((1, 3)).max()), 1.0 + 1e-12)
        self.assertEqual(float(kap[..., [0, 1, 12]].abs().max()), 0.0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA')
class FusedBands(unittest.TestCase):
    def test_fused_trace_step_equals_pytorch(self):
        """arc/bands_kernel.py against the PyTorch path in FP32 (T = 50: a full and a partial token tile): the second
        segment's logits, carry and every parameter gradient."""
        dev = torch.device('cuda')
        out = {}
        for fused in (False, True):
            cfg = small(canvas=5, d=32, frames=3, frame_dim=8, dtype='float32', amp=False, fused_bands=fused)
            m = model(cfg).to(dev)
            b = {k: v.to(dev) for k, v in batch(cfg).items()}
            _, carry = m.segment(None, b)
            logits, carry = m.segment(carry, b)
            logits.square().sum().backward()
            out[fused] = [logits, carry['ek'], carry['ev'], carry['w']] + [p.grad for p in m.parameters()
                                                                           if p.grad is not None]
        self.assertEqual(len(out[False]), len(out[True]))
        for a, b in zip(out[False], out[True]):
            self.assertLessEqual(float((a - b).abs().max()), 1e-4 * float(a.abs().max()))


if __name__ == '__main__':
    unittest.main()
