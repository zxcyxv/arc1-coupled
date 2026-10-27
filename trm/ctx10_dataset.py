"""TRM's PuzzleDataset over the ARC-1 <= 10 harness data (arc1-coupled data/arc1_ctx10_trans.npz, arc/small.py Episodes):
255 evaluation tasks, their train pairs as the train split (groups = tasks, puzzles = the 1001 versions of arc.small,
examples = the task's train pairs), their 278 test pairs in version 0 untranslated as the test split (the <= 10
harness's evaluation). Canvases 11 x 11 in TRM tokens (0 pad, 1 EOS, 2..11 colours). As in TRM's own builder, each
(version, example) has one fixed translation (example 0 of a version untranslated). One puzzle identifier per
(task, version), shared by both splits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: arc.*
from arc.small import Episodes  # noqa: E402

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from puzzle_dataset import PuzzleDataset  # noqa: E402

TO_TRM = np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1])   # model token (0 unknown, 1 pad, 2..11, 12 EOS) -> TRM


class LazyCanvases:
    def __init__(self, ep, task, ver, pair, shift, which):
        self.ep, self.task, self.ver, self.pair, self.shift, self.which = ep, task, ver, pair, shift, which

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        idx = np.arange(len(self.task))[idx] if isinstance(idx, slice) else np.asarray(idx)
        C = self.ep.C
        out = np.empty((len(idx), C * C), np.int32)
        for r, i in enumerate(idx):
            k, perm = self.ep.version(self.task[i], self.ver[i])
            g = self.ep.grids(self.pair[i])
            g = [np.rot90(perm[x], k % 4) for x in g]
            if k >= 4:
                g = [x.T for x in g]
            fr, fc = self.shift[i]
            hr, hc = max(g[0].shape[0], g[1].shape[0]), max(g[0].shape[1], g[1].shape[1])
            pr, pc = int(fr * (C - hr + 1)), int(fc * (C - hc + 1))
            out[r] = TO_TRM[self.ep.canvas(g[self.which], pr, pc)].ravel()
        return out


class Ctx10PuzzleDataset(PuzzleDataset):
    def _load_metadata(self, dataset_path):
        self._build(dataset_path)
        d = self._index
        n_puz = len(d['puzzle_indices']) - 1
        C = self.ep.C
        return PuzzleDatasetMetadata(seq_len=C * C, vocab_size=12, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                                     num_puzzle_identifiers=1 + self.ep.n_tasks * self.ep.nv,
                                     total_groups=len(d['group_indices']) - 1,
                                     mean_puzzle_examples=len(d['task']) / n_puz, total_puzzles=n_puz, sets=['all'])

    def _build(self, path):
        if getattr(self, '_index', None) is not None:
            return
        ep = self.ep = Episodes(path, seed=0, num_aug=1000, D=5)
        test = self.split == 'test'
        rng = np.random.default_rng(42)
        task, ver, pair, shift, pid, p_idx, g_idx = [], [], [], [], [], [0], [0]
        for t in ep.tasks:
            pairs = ep.test_pairs[t] if test else ep.train_pairs[t]
            if len(pairs) == 0:
                continue
            for v in ([0] if test else range(ep.nv)):
                n = len(pairs)
                task.append(np.full(n, t)); ver.append(np.full(n, v)); pair.append(np.asarray(pairs))
                s = np.zeros((n, 2)) if test else rng.random((n, 2))
                if not test:
                    s[0] = 0.0                                    # one untranslated example per version (TRM)
                shift.append(s)
                pid.append(1 + int(t) * ep.nv + v)
                p_idx.append(p_idx[-1] + n)
            g_idx.append(len(p_idx) - 1)
        self._index = dict(task=np.concatenate(task), ver=np.concatenate(ver), pair=np.concatenate(pair),
                           shift=np.concatenate(shift), puzzle_identifiers=np.array(pid, np.int32),
                           puzzle_indices=np.array(p_idx, np.int32), group_indices=np.array(g_idx, np.int32))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        d = self._index
        mk = lambda w: LazyCanvases(self.ep, d['task'], d['ver'], d['pair'], d['shift'], w)
        self._data = {'all': dict(inputs=mk(0), labels=mk(1), puzzle_identifiers=d['puzzle_identifiers'],
                                  puzzle_indices=d['puzzle_indices'], group_indices=d['group_indices'])}
