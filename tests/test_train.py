"""Training harness: exact resume, DDP = one process, the TRM loss. python -m unittest tests.test_train (CPU, ~1 min)"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from arc.data import default_store_path
from arc.train import lm_loss, main
from arc.model import PAD

ROOT = Path(__file__).resolve().parents[1]
TINY = dict(model=dict(canvas=30, d=16, heads=2, blocks=2, grad_blocks=1, phi=8, bands=1, band_periods=[2.0], frames=2,
                       frame_dim=8, dtype='float64', amp=False),
            global_batch_size=8, grad_accum=2, loops=4, epochs=1, lr=1e-3, lr_warmup_steps=2,
            quick_eval_epochs=0, vote_eval_epochs=0, final_eval=False, save_every_steps=4, keep_last=3, log_every=4,
            compile=False)


def weights(out, step):
    d = torch.load(Path(out) / f'step_{step}.pt', map_location='cpu', weights_only=False)
    return d['model'], d['ema'], d['cursor']


@unittest.skipUnless(default_store_path().exists(), 'data/arc1_trm.npz not built (scripts/prepare_data.py)')
class Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['ARC_DEVICE'] = 'cpu'
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = Path(cls.tmp.name) / 'tiny.json'
        cls.cfg.write_text(json.dumps(TINY))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_main(self, out, *sets):
        main(['--config', str(self.cfg), '--out', str(out), '--set', *sets])

    def test_resume_is_exact(self):
        a, b = Path(self.tmp.name) / 'a', Path(self.tmp.name) / 'b'
        self.run_main(a, 'max_steps=12')
        self.run_main(b, 'max_steps=4')
        self.run_main(b, 'max_steps=12')                               # resumes from step_4.pt
        (ma, ea, ca), (mb, eb, cb) = weights(a, 12), weights(b, 12)
        self.assertEqual(ca, cb)
        for k in ma:
            torch.testing.assert_close(ma[k], mb[k], rtol=0, atol=0)
        for k in ea:
            torch.testing.assert_close(ea[k], eb[k], rtol=0, atol=0)
        # the milestone written at the end holds the EMA weights, and evaluation loads them like the full checkpoint
        from arc.evaluate import load_weights
        from arc.train import build_model, load_config
        cfg = load_config(self.cfg)
        full = torch.load(a / 'step_12.pt', map_location='cpu', weights_only=False)
        mile = torch.load(a / 'milestones' / 'step_12.pt', map_location='cpu', weights_only=False)
        m1, m2 = build_model(cfg, torch.device('cpu')), build_model(cfg, torch.device('cpu'))
        load_weights(m1, full, 'ema')
        load_weights(m2, mile, 'ema')
        for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            torch.testing.assert_close(p1, p2, rtol=0, atol=0)
            torch.testing.assert_close(p1, ea[n], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            load_weights(m2, mile, 'raw')

    def test_ddp_equals_one_process(self):
        one, two = Path(self.tmp.name) / 'one', Path(self.tmp.name) / 'two'
        self.run_main(one, 'max_steps=8', 'grad_accum=2')
        env = dict(os.environ, ARC_DEVICE='cpu', OMP_NUM_THREADS='1')
        r = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                            '-m', 'arc.train', '--config', str(self.cfg), '--out', str(two), '--set', 'max_steps=8',
                            'grad_accum=1'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        (m1, e1, c1), (m2, e2, c2) = weights(one, 8), weights(two, 8)
        self.assertEqual(c1, c2)
        for k in m1:
            torch.testing.assert_close(m1[k], m2[k], rtol=0, atol=1e-12)


class Optimizer(unittest.TestCase):
    def test_adam_atan2_is_trms_kernel(self):
        """Three steps against adam-atan2 0.0.3's adam_math written out per element (FP64)."""
        from arc.train import AdamATan2
        g = torch.Generator().manual_seed(0)
        p = torch.randn(50, dtype=torch.float64, generator=g)
        ref, m, v = p.clone(), torch.zeros(50, dtype=torch.float64), torch.zeros(50, dtype=torch.float64)
        w = torch.nn.Parameter(p.clone())
        opt = AdamATan2([w], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
        for t in range(1, 4):
            grad = torch.randn(50, dtype=torch.float64, generator=g)
            w.grad = grad.clone()
            opt.step()
            ref = ref * (1 - 1e-2 * 0.1)
            m = m + (1 - 0.9) * (grad - m)
            v = v + (1 - 0.95) * (grad * grad - v)
            ref = ref - 1e-2 / (1 - 0.9 ** t) * torch.atan2(m, v.sqrt() / (1 - 0.95 ** t) ** 0.5)
            torch.testing.assert_close(w.detach(), ref, rtol=0, atol=1e-15)


class Loss(unittest.TestCase):
    def test_trm_lm_loss(self):
        """Per lane the mean stablemax CE over labelled cells, summed over lanes (TRM ACTLossHead)."""
        g = torch.Generator().manual_seed(0)
        O = 4
        logits = torch.randn(3, O, 13, generator=g, dtype=torch.float64)
        y = torch.randint(2, 13, (3, O), generator=g)
        y[0, :2] = PAD
        labels = torch.cat([torch.full((3, O), PAD), y], 1)
        loss, st = lm_loss(logits, labels, O)
        s = torch.where(logits < 0, 1 / (1 - logits + 1e-30), logits + 1)
        lp = torch.log(s / s.sum(-1, keepdim=True))
        ref = sum(float(-lp[i, j, y[i, j]].mean()) for i in range(3)
                  for j in [torch.nonzero(y[i] != PAD).flatten()])
        self.assertAlmostEqual(float(loss), ref, places=10)
        self.assertEqual(int(st[1]), 3)


if __name__ == '__main__':
    unittest.main()
