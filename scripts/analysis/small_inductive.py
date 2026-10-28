"""A <= 10 harness checkpoint (arc.small) on ARC-1 evaluation tasks it never trained on: every evaluation task whose
grids all fit 10 x 10, its train pairs as demonstrations (the first D), each test pair as the query, version 0 (no
augmentation), as arc.small's test evaluation. The harness's own test set is evaluated too, as a check (it must give
the logged number).

    python scripts/analysis/small_inductive.py runs/small/b8g4_rope1d_100k/last.pt
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from arc.model import Coupled, ModelConfig  # noqa: E402
from arc.small import Episodes, evaluate  # noqa: E402


def build(challenges, solutions, out, side=10):
    """The tasks whose grids all fit side x side, as an arc.small store (train pairs trainable, test pairs not)."""
    ch, so = json.load(open(challenges)), json.load(open(solutions))
    cells, off, hw, task, trainable, names = [], [], [], [], [], []
    n = 0
    for name in sorted(ch):
        pairs = [(p['input'], p['output'], True) for p in ch[name]['train']]
        pairs += [(p['input'], o, False) for p, o in zip(ch[name]['test'], so[name])]
        grids = [np.array(g) for a, b, _ in pairs for g in (a, b)]
        if max(max(g.shape) for g in grids) > side:
            continue
        t = len(names)
        names.append(name)
        for a, b, tr in pairs:
            a, b = np.array(a, np.uint8), np.array(b, np.uint8)
            off.append(n)
            hw.append((*a.shape, *b.shape))
            cells += [a.ravel(), b.ravel()]
            n += a.size + b.size
            task.append(t)
            trainable.append(tr)
    np.savez(out, pair_cells=np.concatenate(cells), pair_off=np.array(off), pair_hw=np.array(hw, np.int16),
             pair_task=np.array(task), pair_trainable=np.array(trainable), task_names=np.array(names))
    return len(names)


def main():
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    meta = ck['meta']
    device = torch.device('cuda')
    model = Coupled(ModelConfig(**meta['model'])).to(device)
    missing = model.load_state_dict(ck['ema'], strict=False)
    assert not missing.unexpected_keys and all(k not in dict(model.named_parameters()) for k in missing.missing_keys), missing
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False                     # as arc.small
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    import torch._inductor.config as ic
    ic.triton.persistent_reductions = False
    model.layer_fn = torch.compile(model.layer, dynamic=False)
    side = meta['model']['canvas'] - 1
    print(f"checkpoint step {ck['step']}, canvas {meta['model']['canvas']}")
    res = evaluate(model, Episodes(meta['data'], N=side, D=meta['D']), device, meta['eval_groups'])
    print(f"own test set: exact {res['exact']}/{res['n']} cell {res['cell'] / res['n_cell']:.4f} ce {res['test_ce']:.4f}")
    with tempfile.TemporaryDirectory() as d:
        k = ROOT / 'data/kaggle'
        for split in ('evaluation', 'training'):
            path = Path(d) / f'{split}.npz'
            nt = build(k / f'arc-agi_{split}_challenges.json', k / f'arc-agi_{split}_solutions.json', path, side)
            res = evaluate(model, Episodes(path, N=side, D=meta['D']), device, meta['eval_groups'])
            print(f"ARC-1 {split} tasks with grids <= {side}: {nt} tasks, exact {res['exact']}/{res['n']} "
                  f"({res['exact'] / res['n']:.3f}) cell {res['cell'] / res['n_cell']:.4f} ce {res['test_ce']:.4f}")


if __name__ == '__main__':
    main()
