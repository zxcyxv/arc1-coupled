"""identifiers.json and test_puzzles.json (TRM's evaluator inputs) for a directory holding our store as store.npz:
identifier 1 + ver_off[t] + v names task t's version v as TRM's builder does (the original name for the identity,
else name|||t{dihedral}|||{colour map}); test_puzzles holds every evaluation task's test pairs in the original frame.
usage: python trm/make_eval_meta.py DIR   (DIR/store.npz: a link to data/<benchmark>_trm.npz)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the repository root: arc.*
from arc.data import Store

d = sys.argv[1]
s = Store(os.path.join(d, 'store.npz'))
z = s.z
names = ['<blank>'] + [None] * int(z['ver_off'][-1])
for t in range(s.n_tasks):
    nm = str(s.names[t])
    for v in range(int(s.n_versions[t])):
        tid, mp = s.version(t, v)
        ident = tid == 0 and np.array_equal(mp, np.arange(10))
        names[1 + int(z['ver_off'][t]) + v] = nm if ident else f"{nm}|||t{tid}|||{''.join(str(int(x)) for x in mp)}"
assert None not in names
tests = {}
for t in s.eval_tasks:
    pairs = []
    for j in range(int(z['task_test_n'][t])):
        gi, go = s.grids(int(z['task_test_start'][t]) + j)
        pairs.append(dict(input=np.asarray(gi).tolist(), output=np.asarray(go).tolist()))
    tests[str(s.names[t])] = dict(train=[], test=pairs)
json.dump(names, open(os.path.join(d, 'identifiers.json'), 'w'))
json.dump(tests, open(os.path.join(d, 'test_puzzles.json'), 'w'))
print(len(names), 'identifiers;', len(tests), 'test puzzles;', sum(len(v['test']) for v in tests.values()), 'test pairs')
