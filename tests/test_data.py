"""The store reproduces TRM's canvases; augmentation, crop and the training stream. python -m unittest tests.test_data"""
import unittest
from pathlib import Path

import numpy as np

from arc.data import (ARC_MAX, TO_MODEL, Store, TrainStream, augment, canvas, crop, default_store_path, dihedral,
                      inverse)
from arc.model import PAD, UNKNOWN

ROOT = Path(__file__).resolve().parents[1]
STORE = default_store_path()
FIXTURE = ROOT / 'tests' / 'fixtures' / 'trm_sample.npz'


def trm_crop(grid):
    """TRM evaluators/arc.py _crop, line for line (without numba)."""
    grid = grid.reshape(30, 30)
    max_area, max_size = 0, (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area, max_size = area, (num_r, num_c)
    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class Augmentation(unittest.TestCase):
    def test_inverse_undoes_every_version(self):
        rng = np.random.default_rng(0)
        g = rng.integers(0, 10, (4, 7)).astype(np.uint8)
        for tid in range(8):
            mapping = np.concatenate([[0], 1 + rng.permutation(9)]).astype(np.uint8)
            np.testing.assert_array_equal(inverse(augment(g, tid, mapping), tid, mapping), g)

    def test_transposed_ids(self):
        g = np.zeros((2, 5), np.uint8)
        self.assertEqual([dihedral(g, t).shape for t in range(8)],
                         [(2, 5), (5, 2), (2, 5), (5, 2), (2, 5), (2, 5), (5, 2), (5, 2)])

    def test_crop_is_trms(self):
        rng = np.random.default_rng(1)
        for _ in range(300):
            h, w = (int(x) for x in rng.integers(1, 31, 2))
            c = canvas(rng.integers(0, 10, (h, w)).astype(np.uint8), 0, 0)
            if rng.random() < 0.5:                                  # corrupt a few cells
                c.reshape(-1)[rng.integers(0, 900, int(rng.integers(1, 6)))] = rng.integers(0, 12)
            mine = crop(TO_MODEL[c][None])[0]
            np.testing.assert_array_equal(mine, trm_crop(c))


@unittest.skipUnless(STORE.exists(), 'data/arc1_trm.npz not built (scripts/prepare_data.py)')
class StoreMatchesTRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Store(STORE)

    def test_counts(self):
        s = self.store
        self.assertEqual((s.n_tasks, len(s.eval_tasks), s.train_pairs, s.test_pairs), (960, 400, 3988, 419))
        self.assertEqual(int(s.n_versions.sum()), 876405)               # TRM: 876,406 identifiers incl. blank

    @unittest.skipUnless(FIXTURE.exists(), 'no TRM fixture')
    def test_trm_canvases(self):
        f = np.load(FIXTURE)
        for t, v, j, x, y in zip(f['task'], f['version'], f['example'], f['inputs'], f['labels']):
            ci, co = self.store.trm_canvases(int(t), int(v), int(j))
            np.testing.assert_array_equal(ci.ravel(), x)
            np.testing.assert_array_equal(co.ravel(), y)

    def test_lane_layout(self):
        x, y = self.store.lane(0, 3, 1)
        O = ARC_MAX * ARC_MAX
        self.assertTrue((x[O:] == UNKNOWN).all() and (y[:O] == PAD).all())
        ci, co = self.store.trm_canvases(0, 3, 1)
        np.testing.assert_array_equal(x[:O], TO_MODEL[ci.ravel()])
        np.testing.assert_array_equal(y[O:], TO_MODEL[co.ravel()])

    def test_stream_packs_full_micro_batches(self):
        st = TrainStream(self.store, seed=0, lanes=96, n_mb=8)
        plan, cur = st.plan(0)
        self.assertEqual(len(plan), 8)
        n_tr = self.store.z['task_train_n']
        for mb in plan:
            self.assertEqual(sum(len(ex) for _, _, ex in mb), 96)
            for t, v, ex in mb[:-1]:                                     # only the last group may be cut
                self.assertEqual(len(ex), n_tr[t])
        again, cur2 = TrainStream(self.store, seed=0, lanes=96, n_mb=8).plan(0)
        self.assertEqual(cur, cur2)
        self.assertEqual([[(t, v, list(e)) for t, v, e in mb] for mb in plan],
                         [[(t, v, list(e)) for t, v, e in mb] for mb in again])
        nxt, _ = st.plan(cur)
        self.assertNotEqual(nxt[0][0][:2], plan[0][0][:2])

    def test_epoch_visits_every_task_once(self):
        st = TrainStream(self.store, seed=0, lanes=96, n_mb=1)
        self.assertEqual(sorted(st.visit(i)[0] for i in range(960)), list(range(960)))


ARC2 = default_store_path('arc2')
FIXTURE2 = ROOT / 'tests' / 'fixtures' / 'trm_sample_arc2.npz'


@unittest.skipUnless(ARC2.exists(), 'data/arc2_trm.npz not built (scripts/prepare_data.py --benchmark arc2)')
class Arc2StoreMatchesTRM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = Store(ARC2)

    def test_counts(self):
        s = self.store
        self.assertEqual((s.n_tasks, len(s.eval_tasks), s.train_pairs, s.test_pairs), (1280, 120, 5573, 172))

    @unittest.skipUnless(FIXTURE2.exists(), 'no TRM fixture')
    def test_trm_canvases(self):
        f = np.load(FIXTURE2)
        for t, v, j, x, y in zip(f['task'], f['version'], f['example'], f['inputs'], f['labels']):
            ci, co = self.store.trm_canvases(int(t), int(v), int(j))
            np.testing.assert_array_equal(ci.ravel(), x)
            np.testing.assert_array_equal(co.ravel(), y)


if __name__ == '__main__':
    unittest.main()
