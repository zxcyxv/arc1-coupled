"""The small ARC-1 <= 10 harness (arc.small): episodes, loss, a CPU run. python -m unittest tests.test_small (CPU)"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from arc.model import EOS, PAD, UNKNOWN, VOCAB, Coupled, ModelConfig
from arc.small import DATA, Episodes, main, query_loss, scores
from arc.train import lm_loss

TINY = dict(d=16, heads=2, blocks=2, grad_blocks=1, phi=8, bands=1, band_periods=[2.0], frames=2, frame_dim=8,
            fused_bands=False)


@unittest.skipUnless(DATA.exists(), 'data/arc1_ctx10_trans.npz missing')
class Small(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = Episodes(DATA, seed=0)

    def test_store(self):
        ep = self.ep
        self.assertEqual((ep.N, ep.C, len(ep.tasks), ep.n_test), (10, 11, 255, 278))

    def test_train_batch(self):
        ep, C = self.ep, self.ep.C
        b = ep.train_batch(64, np.random.default_rng(1))
        x, y, g = b['inputs'], b['labels'], b['group']
        self.assertEqual(tuple(x.shape), (64, 2 * C * C))
        self.assertTrue(b['demo'].all() and b['query'].all())
        self.assertTrue((x[:, C * C:] == UNKNOWN).all())                 # the output canvas never leaks
        self.assertTrue((y[:, :C * C] == PAD).all())                     # labels only on the output canvas
        sizes = torch.bincount(g)
        self.assertTrue((sizes <= ep.D + 1).all() and sizes.sum() == 64)
        # every output canvas: colours in one rectangle, EOS just outside it, pad elsewhere
        for lane in y[:, C * C:].view(-1, C, C):
            col = (lane >= 2) & (lane <= 11)
            rows, cols = col.any(1).nonzero().flatten(), col.any(0).nonzero().flatten()
            h, w = len(rows), len(cols)
            r, c = int(rows[0]), int(cols[0])
            self.assertTrue(col[r:r + h, c:c + w].all() and col.sum() == h * w)
            if r + h < C:
                self.assertTrue((lane[r + h, c:min(c + w + 1, C)] == EOS).all())
            self.assertEqual(int((lane == EOS).sum()), int((lane != PAD).sum()) - h * w)

    def test_versions(self):
        k0, p0 = self.ep.version(3, 0)
        self.assertEqual(k0, 0)
        np.testing.assert_array_equal(p0, np.arange(10))
        k, p = self.ep.version(3, 7)
        self.assertEqual(p[0], 0)                                        # black fixed
        self.assertEqual(sorted(p.tolist()), list(range(10)))
        k2, p2 = Episodes(DATA, seed=0).version(3, 7)                   # fixed by (seed, task, version)
        self.assertEqual((k, p.tolist()), (k2, p2.tolist()))

    def test_eval_batches(self):
        n, lanes = 0, set()
        for b in self.ep.eval_batches(16):
            lanes.add(b['inputs'].shape[0])
            q = b['query']
            n += int(q.sum())
            self.assertFalse((q & b['demo']).any())
            for gid in b['group'][q]:                                    # each query: its group's other lanes are demos
                grp = b['group'] == gid
                self.assertEqual(int((grp & q).sum()), 1)
                self.assertTrue(b['demo'][grp & ~q].all())
        self.assertEqual(n, 278)
        self.assertEqual(lanes, {16 * (self.ep.D + 1)})                  # padded: one shape

    def test_loss_and_scores(self):
        torch.manual_seed(0)
        b = self.ep.train_batch(8, np.random.default_rng(0))
        logits = torch.randn(8, 121, VOCAB, dtype=torch.float64)
        loss, st = lm_loss(logits, b['labels'], 121)
        torch.testing.assert_close(query_loss(logits, b['labels'], b['query']), loss / 8)   # all lanes are queries
        y = b['labels'][:, 121:]
        perfect = torch.nn.functional.one_hot(y, VOCAB).double() * 10
        s = scores(perfect, b['labels'], b['query'])
        self.assertEqual((s['exact'], s['cell'], s['cell_eos']), (8, s['n_cell'], s['n_eos']))
        self.assertEqual(s['n_eos'], int((y != PAD).sum()))
        # pad cells never count: a wrong prediction on pad changes neither the loss nor exact
        bad = perfect.clone()
        bad[y == PAD] = torch.nn.functional.one_hot(torch.tensor(5), VOCAB).double() * 10
        self.assertEqual(scores(bad, b['labels'], b['query'])['exact'], 8)
        torch.testing.assert_close(query_loss(bad, b['labels'], b['query']),
                                   query_loss(perfect, b['labels'], b['query']))

    def test_padding_lanes_change_nothing(self):
        torch.manual_seed(0)
        m = Coupled(ModelConfig(canvas=11, dtype='float64', **TINY)).eval()
        groups = [self.ep.group(0, list(self.ep.train_pairs[0][:3]) + [self.ep.test_pairs[0][0]], 0, None,
                                [True] * 3 + [False], [False] * 3 + [True])]
        a, b = self.ep.collate(groups), self.ep.collate(groups, pad_to=7)
        with torch.no_grad():
            la, lb = m.segment(None, a)[0], m.segment(None, b)[0]
        torch.testing.assert_close(la, lb[:4], rtol=0, atol=0)

    def test_cpu_run_and_resume(self):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        with tempfile.TemporaryDirectory() as tmp:
            sets = ['batch=8', 'eval_groups=8', 'log_every=2', 'eval_every=2', 'model=' + json.dumps(TINY)]
            main(['--out', tmp, '--device', 'cpu', '--set', 'steps=2', *sets])
            main(['--out', tmp, '--device', 'cpu', '--set', 'steps=4', *sets])     # resumes from step 2
            ev = [json.loads(l) for l in (Path(tmp) / 'eval.jsonl').read_text().splitlines()]
            self.assertEqual([e['step'] for e in ev], [2, 4])
            self.assertTrue(all(e['n'] == 278 for e in ev))
            with self.assertRaises(ValueError):                                     # another config is refused
                main(['--out', tmp, '--device', 'cpu', '--set', 'steps=6', 'lr=1e-3', *sets])


if __name__ == '__main__':
    unittest.main()
