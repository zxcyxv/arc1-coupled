"""ARC-AGI-1 in TRM's split and augmentation, stored compactly, and the lanes the coupled model reads.

TRM (TinyRecursiveModels dataset/build_arc_dataset.py: training + evaluation + concept, --test-set-name evaluation,
num_aug 1000, seed 42) materialises 4M x 900 canvases. `build_store` replays its random stream instead (one
np.random.RandomState(42), the same calls in the same order) and keeps only what the stream decides: every task's
versions (dihedral id, colour map; duplicates removed by TRM's puzzle hash) and every training example's fixed
translation (TRM leaves one random example per version untranslated). `lane_canvases` rebuilds TRM's 30 x 30 canvases
from these exactly (scripts/prepare_data.py --verify compares them with TRM's own arrays).

Split (TRM): the train split of a training or concept task is its demonstrations + its test pairs; of an evaluation task,
its demonstrations; the test split is the evaluation tasks' test pairs (no translation). Each task version is a group:
its train-split examples are lanes that are demonstrations for each other (leave-one-out) and queries for themselves.

Tokens: TRM canvases use 0 pad, 1 EOS, 2..11 colours; the model's lanes use 0 unknown (the whole output canvas),
1 pad, 2..11 colours, 12 EOS. A lane is [input canvas, output canvas], labels only on the output canvas.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import EOS, PAD, UNKNOWN

ARC_MAX = 30
BENCHMARKS = {                                            # TRM README: --subsets ... --test-set-name ...
    'arc1': (('training', 'evaluation', 'concept'), 'evaluation'),
    'arc2': (('training2', 'evaluation2', 'concept'), 'evaluation2'),
}
RETRIES = 5                                               # TRM ARCAugmentRetriesFactor
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]
TRANSPOSED = (1, 3, 6, 7)                                 # dihedral ids that swap height and width
TO_MODEL = np.array([PAD, EOS] + list(range(2, 12)), dtype=np.int64)   # TRM token -> model token


# ---------------------------------------------------------------------------------------------- TRM's functions
def dihedral(arr, tid):
    """TRM dataset/common.py dihedral_transform."""
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    return arr


def grid_hash(grid):
    assert grid.ndim == 2 and grid.dtype == np.uint8
    return hashlib.sha256(b''.join([x.to_bytes(1, byteorder='big') for x in grid.shape] + [grid.tobytes()])).hexdigest()


def puzzle_hash(dests):
    """TRM puzzle_hash: dests = list of example lists (one per destination split)."""
    hashes = sorted(f'{grid_hash(i)}|{grid_hash(o)}' for examples in dests for i, o in examples)
    return hashlib.sha256('|'.join(hashes).encode()).hexdigest()


def augment(grid, tid, mapping):
    """TRM aug(): colours mapped first, then the dihedral transform."""
    return dihedral(mapping[grid], tid)


def inverse(grid, tid, mapping):
    """TRM inverse_aug(): the inverse dihedral transform, then the inverse colour map."""
    return np.argsort(mapping).astype(np.uint8)[dihedral(grid, DIHEDRAL_INVERSE[tid])]


def canvas(grid, pr, pc):
    """TRM np_grid_to_seq_translational_augment for one grid at offset (pr, pc): colour + 2, EOS (1) on the row and
    column just outside the grid (not the corner), 0 elsewhere."""
    h, w = grid.shape
    out = np.zeros((ARC_MAX, ARC_MAX), dtype=np.uint8)
    out[pr:pr + h, pc:pc + w] = grid + 2
    if pr + h < ARC_MAX:
        out[pr + h, pc:pc + w] = 1
    if pc + w < ARC_MAX:
        out[pr:pr + h, pc + w] = 1
    return out


def crop(tokens):
    """TRM evaluators/arc.py _crop, vectorised over [N, 30, 30] model (or TRM) tokens: the largest top-left rectangle
    of colour cells (tokens 2..11), returned as colours 0..9 (list of uint8 arrays)."""
    tokens = np.asarray(tokens).reshape(-1, ARC_MAX, ARC_MAX)
    colour = (tokens >= 2) & (tokens <= 11)
    first_bad = np.where(colour.all(-1), ARC_MAX, np.argmin(colour, -1))          # leading colour run per row
    cols = np.minimum.accumulate(first_bad, axis=1)                               # TRM's shrinking num_c
    area = cols * np.arange(1, ARC_MAX + 1)[None]
    rows = area.argmax(1) + 1
    best = area.max(1)
    out = []
    for n in range(len(tokens)):
        r, c = (int(rows[n]), int(cols[n, rows[n] - 1])) if best[n] > 0 else (0, 0)
        out.append((tokens[n, :r, :c] - 2).astype(np.uint8))
    return out


# ---------------------------------------------------------------------------------------------- store
def load_kaggle(prefix, subsets):
    """{subset: [(name, {'train': [...], 'test': [...]}), ...]} in file order, test outputs filled in (TRM)."""
    out = {}
    for s in subsets:
        with open(f'{prefix}_{s}_challenges.json') as f:
            puzzles = json.load(f)
        with open(f'{prefix}_{s}_solutions.json') as f:
            sols = json.load(f)
        for pid in puzzles:
            for i, g in enumerate(sols[pid]):
                puzzles[pid]['test'][i]['output'] = g
        out[s] = list(puzzles.items())
    return out


def build_store(prefix, benchmark='arc1', seed=42, num_aug=1000, log=print):
    """Replay TRM's builder stream; returns the compact store (dict of numpy arrays)."""
    subsets, test_subset = BENCHMARKS[benchmark]
    rs = np.random.RandomState(seed)
    kaggle = load_kaggle(prefix, subsets)
    tasks = []
    for s in subsets:
        puzzles = kaggle[s]
        rs.shuffle(puzzles)
        for name, puzzle in puzzles:
            dests = {}                                    # TRM dest_mapping: 'train' or 'test' split per example type
            for kind, examples in puzzle.items():
                dest = 'test' if (s == test_subset and kind == 'test') else 'train'
                dests.setdefault(dest, []).extend(
                    (np.array(e['input'], dtype=np.uint8), np.array(e['output'], dtype=np.uint8)) for e in examples)
            versions = [(0, np.arange(10, dtype=np.uint8))]
            hashes = {puzzle_hash(list(dests.values()))}
            for _ in range(RETRIES * num_aug):
                tid = rs.randint(0, 8)
                mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8),
                                          rs.permutation(np.arange(1, 10, dtype=np.uint8))])
                h = puzzle_hash([[(augment(i, tid, mapping), augment(o, tid, mapping)) for i, o in ex]
                                 for ex in dests.values()])
                if h not in hashes:
                    hashes.add(h)
                    versions.append((tid, mapping))
                if len(versions) >= num_aug + 1:
                    break
            if len(versions) < num_aug + 1:
                log(f'[data] {name}: {len(versions)} versions (TRM: augmentation not full)')
            tasks.append(dict(name=name, subset=subsets.index(s), train=dests['train'], test=dests.get('test', []),
                              versions=versions))
    # TRM convert_dataset, train split: per group, per version, one untranslated example, the others translated
    pads = []
    for t in tasks:
        n = len(t['train'])
        for tid, _ in t['versions']:
            keep = rs.randint(0, n)
            for j, (i, o) in enumerate(t['train']):
                if j == keep:
                    pads.append((0, 0))
                    continue
                hi, wi, ho, wo = *i.shape, *o.shape
                if tid in TRANSPOSED:
                    hi, wi, ho, wo = wi, hi, wo, ho
                pads.append((rs.randint(0, ARC_MAX - max(hi, ho) + 1), rs.randint(0, ARC_MAX - max(wi, wo) + 1)))
    # (the test split's per-version draws come after these and decide nothing)
    cells, off, hw, pos = [], [], [], 0
    tr_start, te_start = [], []
    for t in tasks:
        for kind, starts in (('train', tr_start), ('test', te_start)):
            starts.append(len(off))
            for i, o in t[kind]:
                off.append(pos)
                hw.append((*i.shape, *o.shape))
                cells += [i.ravel(), o.ravel()]
                pos += i.size + o.size
    nv = np.array([len(t['versions']) for t in tasks], np.int64)
    ntr = np.array([len(t['train']) for t in tasks], np.int64)
    return dict(
        cells=np.concatenate(cells), pair_off=np.array(off, np.int64), pair_hw=np.array(hw, np.int16),
        task_names=np.array([t['name'] for t in tasks]), task_subset=np.array([t['subset'] for t in tasks], np.int8),
        task_train_start=np.array(tr_start, np.int64), task_train_n=ntr,
        task_test_start=np.array(te_start, np.int64), task_test_n=np.array([len(t['test']) for t in tasks], np.int64),
        ver_off=np.concatenate([[0], np.cumsum(nv)]).astype(np.int64),
        ver_tid=np.array([v[0] for t in tasks for v in t['versions']], np.uint8),
        ver_map=np.stack([v[1] for t in tasks for v in t['versions']]).astype(np.uint8),
        pad_off=np.concatenate([[0], np.cumsum(nv * ntr)]).astype(np.int64), pads=np.array(pads, np.int8),
        seed=np.int64(seed), num_aug=np.int64(num_aug), benchmark=np.array(benchmark),
        subsets=np.array(subsets))


class Store:
    """The compact store (data/arc1_trm.npz) with lane construction."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.z = {k: z[k] for k in z.files}
        z = self.z
        self.names, self.subset = z['task_names'], z['task_subset']
        self.n_tasks = len(self.names)
        self.n_versions = np.diff(z['ver_off'])
        self.eval_tasks = np.flatnonzero(z['task_test_n'] > 0)
        self.train_pairs = int(z['task_train_n'].sum())
        self.test_pairs = int(z['task_test_n'].sum())

    def grids(self, p):
        hi, wi, ho, wo = (int(x) for x in self.z['pair_hw'][p])
        o = int(self.z['pair_off'][p])
        c = self.z['cells']
        return c[o:o + hi * wi].reshape(hi, wi), c[o + hi * wi:o + hi * wi + ho * wo].reshape(ho, wo)

    def version(self, t, v):
        k = int(self.z['ver_off'][t]) + int(v)
        return int(self.z['ver_tid'][k]), self.z['ver_map'][k]

    def pad(self, t, v, j):
        return self.z['pads'][int(self.z['pad_off'][t]) + int(v) * int(self.z['task_train_n'][t]) + int(j)]

    def trm_canvases(self, t, v, j, test=False):
        """TRM's (input, label) canvases of example j of task t's train (or test) split in version v."""
        z = self.z
        p = int(z['task_test_start' if test else 'task_train_start'][t]) + int(j)
        tid, mapping = self.version(t, v)
        i, o = (augment(g, tid, mapping) for g in self.grids(p))
        pr, pc = (0, 0) if test else (int(x) for x in self.pad(t, v, j))
        return canvas(i, pr, pc), canvas(o, pr, pc)

    def lane(self, t, v, j, test=False):
        """Model lane (inputs [T], labels [T]) of example j of task t, version v."""
        ci, co = self.trm_canvases(t, v, j, test)
        O = ARC_MAX * ARC_MAX
        x = np.concatenate([TO_MODEL[ci.ravel()], np.full(O, UNKNOWN)])
        y = np.concatenate([np.full(O, PAD), TO_MODEL[co.ravel()]])
        return x, y

    def answer(self, t, j):
        """Ground-truth output grid (colours 0..9) of task t's test pair j, original frame."""
        return self.grids(int(self.z['task_test_start'][t]) + int(j))[1].astype(np.uint8)


def collate(lanes):
    """lanes: [(inputs, labels, group, demo, query)] -> tensors."""
    return dict(inputs=torch.as_tensor(np.stack([l[0] for l in lanes])),
                labels=torch.as_tensor(np.stack([l[1] for l in lanes])),
                group=torch.as_tensor(np.array([l[2] for l in lanes], np.int64)),
                demo=torch.as_tensor(np.array([l[3] for l in lanes], bool)),
                query=torch.as_tensor(np.array([l[4] for l in lanes], bool)))


# ---------------------------------------------------------------------------------------------- training stream
class TrainStream:
    """TRM's sampler with all lanes replaced together. Visit i (epoch i // n_tasks) takes task perm_e[i % n_tasks], one
    of its versions and all its train-split examples in random order (Philox keyed by (seed, i)). A global batch is
    `n_mb` micro-batches of `lanes` lanes packed with visits in order; a group that does not fit the rest of a
    micro-batch is cut there (TRM cuts at the end of its batch). The cursor (next visit) fixes everything after it."""
    def __init__(self, store, seed, lanes, n_mb):
        self.s, self.seed, self.lanes, self.n_mb = store, int(seed), int(lanes), int(n_mb)
        self._perm = {}

    def _rng(self, kind, i):
        return np.random.Generator(np.random.Philox(key=np.array([self.seed, (kind << 56) + int(i)], dtype=np.uint64)))

    def visit(self, i):
        e, k = divmod(int(i), self.s.n_tasks)
        if e not in self._perm:
            self._perm = {e: self._rng(0, e).permutation(self.s.n_tasks)}
        t = int(self._perm[e][k])
        g = self._rng(1, i)
        v = int(g.integers(self.s.n_versions[t]))
        return t, v, g.permutation(int(self.s.z['task_train_n'][t]))

    def plan(self, cursor):
        """Global batch from `cursor`: [n_mb][(t, v, [example indices])], the next cursor."""
        mbs, cur = [], int(cursor)
        for _ in range(self.n_mb):
            mb, room = [], self.lanes
            while room:
                t, v, ex = self.visit(cur)
                cur += 1
                mb.append((t, v, ex[:room]))
                room -= min(room, len(ex))
            mbs.append(mb)
        return mbs, cur

    def micro_batch(self, groups):
        lanes = [(*self.s.lane(t, v, j), g, True, True) for g, (t, v, ex) in enumerate(groups) for j in ex]
        return collate(lanes)


# ---------------------------------------------------------------------------------------------- evaluation
def eval_groups(store, versions):
    """Every evaluation task in each version: its demonstrations (train split, stored translation) + its test pairs
    (untranslated). [(t, v, n_demos, n_tests)]."""
    z = store.z
    return [(int(t), int(v), int(z['task_train_n'][t]), int(z['task_test_n'][t]))
            for v in versions for t in store.eval_tasks if v < store.n_versions[t]]


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def default_store_path(benchmark='arc1'):
    return Path(__file__).resolve().parents[1] / 'data' / f'{benchmark}_trm.npz'
