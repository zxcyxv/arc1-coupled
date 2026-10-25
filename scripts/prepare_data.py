"""Build data/arc1_trm.npz or data/arc2_trm.npz (arc.data.build_store) and, optionally, check it against TRM's arrays.

    python scripts/prepare_data.py                                   # data/kaggle/arc-agi_* -> data/arc1_trm.npz
    python scripts/prepare_data.py --benchmark arc2                  # -> data/arc2_trm.npz
    python scripts/prepare_data.py --verify TRM_DATA_DIR --fixture tests/fixtures/trm_sample.npz

TRM_DATA_DIR is the output of TinyRecursiveModels `python -m dataset.build_arc_dataset --input-file-prefix
kaggle/combined/arc-agi --output-dir DIR --subsets training evaluation concept --test-set-name evaluation` (6.8 GB).
--verify rebuilds every one of its train and test canvases from the store and compares them byte for byte, checks the
group and puzzle boundaries and mean_puzzle_examples, and with --fixture saves a small random sample of TRM's canvases
for tests/test_data.py (so the check can be repeated without TRM's arrays).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arc.data import Store, build_store, fingerprint  # noqa: E402


def verify(store, trm, fixture=None, sample=256, seed=0):
    ok = True
    for split in ('train', 'test'):
        d = Path(trm) / split
        meta = json.loads((d / 'dataset.json').read_text())
        x = np.load(d / 'all__inputs.npy', mmap_mode='r')
        y = np.load(d / 'all__labels.npy', mmap_mode='r')
        pidx = np.load(d / 'all__puzzle_indices.npy')
        gidx = np.load(d / 'all__group_indices.npy')
        test = split == 'test'
        tasks = store.eval_tasks if test else np.arange(store.n_tasks)
        n_ex = store.z['task_test_n' if test else 'task_train_n']
        if len(gidx) - 1 != len(tasks):
            print(f'[verify] {split}: {len(gidx) - 1} TRM groups, store {len(tasks)}')
            return False
        bad = 0
        t0 = time.time()
        for g, t in enumerate(tasks):
            nv = int(store.n_versions[t])
            if gidx[g + 1] - gidx[g] != nv:
                print(f'[verify] {split} task {store.names[t]}: {gidx[g + 1] - gidx[g]} TRM versions, store {nv}')
                return False
            for v in range(nv):
                p = gidx[g] + v
                if pidx[p + 1] - pidx[p] != n_ex[t]:
                    print(f'[verify] {split} task {store.names[t]} v{v}: example count differs')
                    return False
                for j in range(int(n_ex[t])):
                    ci, co = store.trm_canvases(t, v, j, test=test)
                    e = pidx[p] + j
                    if not (np.array_equal(ci.ravel(), x[e]) and np.array_equal(co.ravel(), y[e])):
                        bad += 1
                        if bad <= 5:
                            print(f'[verify] {split} mismatch: task {store.names[t]} version {v} example {j}')
            if g % 100 == 0:
                print(f'[verify] {split} {g}/{len(tasks)} tasks, {bad} mismatches, {time.time() - t0:.0f}s', flush=True)
        mpe = float((n_ex[tasks] * store.n_versions[tasks]).sum() / store.n_versions[tasks].sum())
        print(f'[verify] {split}: {len(x)} canvases, {bad} mismatches; mean_puzzle_examples store {mpe:.10f} '
              f'TRM {meta["mean_puzzle_examples"]:.10f}', flush=True)
        ok = ok and bad == 0 and abs(mpe - meta['mean_puzzle_examples']) < 1e-9
        if fixture and split == 'train':
            rng = np.random.default_rng(seed)
            picks = []
            for _ in range(sample):
                g = int(rng.integers(len(tasks)))
                t = int(tasks[g])
                v = int(rng.integers(store.n_versions[t]))
                j = int(rng.integers(n_ex[t]))
                e = pidx[gidx[g] + v] + j
                picks.append((t, v, j, e))
            sel = np.array([p[3] for p in picks])
            np.savez_compressed(fixture, task=np.array([p[0] for p in picks]), version=np.array([p[1] for p in picks]),
                                example=np.array([p[2] for p in picks]), inputs=x[sel], labels=y[sel])
            print(f'[verify] wrote {fixture} ({sample} TRM canvases)')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='arc1', choices=('arc1', 'arc2'),
                    help='arc1: training + evaluation + concept; arc2: training2 + evaluation2 + concept (TRM README)')
    ap.add_argument('--kaggle', default=str(ROOT / 'data' / 'kaggle' / 'arc-agi'), help='prefix of the JSON files')
    ap.add_argument('--out', default=None, help='default data/<benchmark>_trm.npz')
    ap.add_argument('--verify', default=None, help="TRM's built dataset directory (arc1concept-aug-1000 / arc2concept-...)")
    ap.add_argument('--fixture', default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / 'data' / f'{args.benchmark}_trm.npz')
    if not out.exists():
        t0 = time.time()
        np.savez_compressed(out, **build_store(args.kaggle, args.benchmark))
        print(f'[data] wrote {out} ({out.stat().st_size / 2 ** 20:.1f} MiB, {time.time() - t0:.0f}s)', flush=True)
    store = Store(out)
    print(f'[data] {store.n_tasks} tasks, {int(store.n_versions.sum())} versions, {store.train_pairs} train pairs, '
          f'{store.test_pairs} test pairs; sha256 {fingerprint(out)[:16]}', flush=True)
    if args.verify:
        ok = verify(store, args.verify, args.fixture)
        print('[verify] ' + ('IDENTICAL to TRM' if ok else 'DIFFERS from TRM'), flush=True)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
