"""TRM's PuzzleDataset over our compact store (arc/data.py Store): the same groups (tasks), puzzles (versions) and
examples as TRM's materialised arrays, with the 30 x 30 canvases built on the fly by Store.trm_canvases (TRM tokens:
0 pad, 1 EOS, 2..11 colours). Sampling, collate and everything downstream are TRM's own code. A data_paths entry is a
directory holding store.npz (a link to data/<benchmark>_trm.npz) and make_eval_meta.py's evaluator files."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: arc.*
from arc.data import Store  # noqa: E402

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from puzzle_dataset import PuzzleDataset  # noqa: E402


class LazyCanvases:
    """inputs or labels [N, 900] int32, built only for the indices asked."""
    def __init__(self, store, task, ver, ex, test, which):
        self.store, self.task, self.ver, self.ex, self.test, self.which = store, task, ver, ex, test, which

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        idx = np.arange(len(self.task))[idx] if isinstance(idx, slice) else np.asarray(idx)
        out = np.empty((len(idx), 900), np.int32)
        for r, i in enumerate(idx):
            out[r] = self.store.trm_canvases(self.task[i], self.ver[i], self.ex[i], self.test)[self.which].ravel()
        return out


class StorePuzzleDataset(PuzzleDataset):
    def _load_metadata(self, dataset_path):
        self._build(dataset_path)
        d = self._index
        n_puz = len(d['puzzle_indices']) - 1
        return PuzzleDatasetMetadata(seq_len=900, vocab_size=12, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                                     num_puzzle_identifiers=self._n_ids, total_groups=len(d['group_indices']) - 1,
                                     mean_puzzle_examples=len(d['task']) / n_puz, total_puzzles=n_puz, sets=['all'])

    def _build(self, path):
        if getattr(self, '_index', None) is not None:
            return
        if os.path.isdir(path):                              # a directory: store.npz + the evaluator's json files
            path = os.path.join(path, 'store.npz')
        s = self.store = Store(path)
        z = s.z
        test = self.split == 'test'
        n_ex = z['task_test_n'] if test else z['task_train_n']
        tasks = np.flatnonzero(n_ex > 0)
        ver_off = z['ver_off']
        task, ver, ex, pid, p_idx, g_idx = [], [], [], [], [0], [0]
        for t in tasks:
            nv = int(s.n_versions[t])
            if test and self.config.eval_versions:               # versions 0..N-1 (arc.evaluate version_set N)
                nv = min(nv, self.config.eval_versions)
            for v in range(nv):
                n = int(n_ex[t])
                task.append(np.full(n, t, np.int32)); ver.append(np.full(n, v, np.int32)); ex.append(np.arange(n, dtype=np.int32))
                pid.append(1 + int(ver_off[t]) + v)               # one identifier per version, shared by both splits
                p_idx.append(p_idx[-1] + n)
            g_idx.append(len(p_idx) - 1)
        self._n_ids = 1 + int(ver_off[-1])
        self._index = dict(task=np.concatenate(task), ver=np.concatenate(ver), ex=np.concatenate(ex),
                           puzzle_identifiers=np.array(pid, np.int32), puzzle_indices=np.array(p_idx, np.int32),
                           group_indices=np.array(g_idx, np.int32))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        d = self._index
        mk = lambda w: LazyCanvases(self.store, d['task'], d['ver'], d['ex'], self.split == 'test', w)
        self._data = {'all': dict(inputs=mk(0), labels=mk(1), puzzle_identifiers=d['puzzle_identifiers'],
                                  puzzle_indices=d['puzzle_indices'], group_indices=d['group_indices'])}
