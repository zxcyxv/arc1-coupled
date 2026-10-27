"""Voting, pass@k and the distributed evaluation path. python -m unittest tests.test_evaluate"""
import os
import unittest

import numpy as np
import torch

from arc.data import Store, default_store_path, grid_hash
from arc.evaluate import eval_plan, run, score_records
from arc.model import Coupled, ModelConfig


class Stub:
    def __init__(self, answers, n_test):
        self.ans = answers
        self.eval_tasks = np.array(sorted(n_test))
        self.z = {'task_test_n': np.array([n_test[t] for t in sorted(n_test)])}

    def answer(self, t, j):
        return self.ans[(t, j)]


class Voting(unittest.TestCase):
    def test_pass_at_k_and_tie_break(self):
        A, B, C = (np.full((2, 2), c, np.uint8) for c in (1, 2, 3))
        hA, hB, hC = (grid_hash(g) for g in (A, B, C))
        store = Stub({(0, 0): A, (0, 1): B, (1, 0): C}, {0: 2, 1: 1})
        recs = [(0, 0, 0, hA, 0.5, -1.0), (0, 0, 1, hA, 0.5, -1.0),       # answer A: 2 votes, demo score 0.5
                (0, 0, 2, hB, 1.0, -1.0), (0, 0, 3, hB, 1.0, -1.0),       # B: 2 votes, demo score 1.0 -> ranked first
                (0, 1, 0, hB, 0.0, 0.0), (0, 1, 1, hB, 0.0, 0.0), (0, 1, 2, hA, 1.0, 0.0),
                (1, 0, 0, hA, 1.0, 0.0)]
        m = score_records(store, recs, [0, 1, 2, 3])
        self.assertAlmostEqual(m['pass@1'], (0.5 + 0.0) / 2)                # task-weighted (TRM)
        self.assertAlmostEqual(m['pass@2'], (1.0 + 0.0) / 2)
        self.assertEqual((m['identity_exact'], m['identity_n']), (2, 3))    # version 0: A right, B right, A wrong

    def test_plan_keeps_groups_whole(self):
        g = [(0, 0, 4, 1), (1, 0, 3, 1), (2, 0, 2, 1)]
        self.assertEqual(eval_plan(g, 8), [[g[0]], [g[1], g[2]]])
        with self.assertRaises(ValueError):
            eval_plan([(0, 0, 9, 1)], 8)


@unittest.skipUnless(default_store_path().exists(), 'data/arc1_trm.npz not built (scripts/prepare_data.py)')
class EndToEnd(unittest.TestCase):
    def test_runs_on_two_tasks(self):
        store = Store(default_store_path())
        store.eval_tasks = store.eval_tasks[:2]
        torch.manual_seed(0)
        model = Coupled(ModelConfig(canvas=30, d=16, heads=2, blocks=2, grad_blocks=1, phi=8, bands=1,
                                    band_periods=(2.0,), frames=2, frame_dim=8, dtype='float64'))
        metrics, recs, grids = run(model, store, [0, 1], lanes=16, segments=2, log=lambda m: None)
        n_tests = int(store.z['task_test_n'][store.eval_tasks].sum())
        self.assertEqual(metrics['identity_n'], n_tests)
        self.assertEqual(len(recs), 2 * n_tests)
        self.assertTrue(all(0.0 <= metrics[f'pass@{k}'] <= 1.0 for k in (1, 2)))
        self.assertTrue(all(0.0 <= metrics[k] <= 1.0 for k in ('identity_cell', 'identity_cell_eos', 'view_cell',
                                                                 'view_cell_eos')))
        self.assertEqual(len(recs[0]), 11)
        self.assertTrue(all(r[3] in grids for r in recs))


if __name__ == '__main__':
    unittest.main()
