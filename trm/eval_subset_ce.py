"""A TRM-harness checkpoint on the ARC-1 evaluation tasks whose grids all fit 10 x 10 (the <= 10 harness's evaluation
tasks, arc.small with data/arc1_ctx10_trans.npz): their test pairs in version 0, TRM's own inference (halt_max_steps
segments, eval mode). Reports what arc.small's evaluation reports on the same pairs: exact pairs, cell accuracy inside the
true grid (pooled over cells), and test CE (per pair the mean stablemax CE over the labelled cells, grid and EOS, then the
mean over pairs).

    cd trm && ../.venv-trm/bin/python eval_subset_ce.py CKPT_DIR DATA_DIR KAGGLE_DIR

CKPT_DIR, DATA_DIR as eval_checkpoint.py; KAGGLE_DIR holds arc-agi_evaluation_{challenges,solutions}.json."""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.losses import IGNORE_LABEL_ID, log_stablemax  # noqa: E402
from puzzle_dataset import PuzzleDatasetConfig  # noqa: E402
from store_dataset import StorePuzzleDataset  # noqa: E402
from utils.functions import load_model_class  # noqa: E402

ckpt_dir, data_dir, kaggle = sys.argv[1], sys.argv[2], sys.argv[3]
B = 256
dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29934', rank=0, world_size=1)

ch = json.load(open(os.path.join(kaggle, 'arc-agi_evaluation_challenges.json')))
so = json.load(open(os.path.join(kaggle, 'arc-agi_evaluation_solutions.json')))
small_tasks = set()
for name, t in ch.items():
    grids = [p[x] for p in t['train'] for x in ('input', 'output')] + [p['input'] for p in t['test']] + so[name]
    if max(max(len(g), len(g[0])) for g in grids) <= 10:
        small_tasks.add(name)

cfg = yaml.safe_load(open(os.path.join(ckpt_dir, 'all_config.yaml')))
ds = StorePuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data_dir], global_batch_size=B, test_set_mode=True,
                                            epochs_per_iter=1, rank=0, num_replicas=1, eval_versions=1),
                        split='test')
meta = ds.metadata
arch = dict(cfg['arch'])
loss_cfg = arch.pop('loss')
arch_name = arch.pop('name')
model_cfg = dict(arch, batch_size=B, vocab_size=meta.vocab_size, seq_len=meta.seq_len,
                 num_puzzle_identifiers=meta.num_puzzle_identifiers, causal=False)
with torch.device('cuda'):
    model = load_model_class(arch_name)(model_cfg)
    head = load_model_class(loss_cfg.pop('name'))(model, **loss_cfg)
path = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*')))[-1]
sd = torch.load(path, map_location='cuda', weights_only=False)
sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
missing, unexpected = head.load_state_dict(sd, strict=False)
assert not unexpected and all('cos_cached' in k or 'sin_cached' in k for k in missing), (missing, unexpected)
head.eval()

ds._lazy_load_dataset()
idx = ds._index
names = ds.store.z['task_names']
keep = np.array([str(names[t]) in small_tasks for t in idx['task']])
exact, ce, right, cells = [], [], 0, 0
pos = 0
with torch.inference_mode():
    for _, batch, _ in ds:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device('cuda'):
            carry = head.initial_carry(batch)
        while True:
            carry, _, _, out, done = head(carry=carry, batch=batch, return_keys={'preds', 'logits'})
            if done:
                break
        n = int((batch['puzzle_identifiers'] != meta.blank_identifier_id).sum())
        lab, pred = batch['labels'][:n], out['preds'][:n]
        valid = lab != IGNORE_LABEL_ID
        lp = log_stablemax(out['logits'][:n].double(), dim=-1)
        nll = -torch.gather(lp, -1, torch.where(valid, lab, 0).long()[..., None]).squeeze(-1)
        pair_ce = (nll * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        grid = valid & (lab >= 2)                                     # TRM tokens: 0 pad, 1 EOS, 2..11 colours
        sel = torch.as_tensor(keep[pos:pos + n], device=lab.device)
        exact += (((pred == lab) | ~valid).all(-1))[sel].tolist()
        ce += pair_ce[sel].tolist()
        right += int(((pred == lab) & grid)[sel].sum())
        cells += int(grid[sel].sum())
        pos += n
assert pos == len(keep)
print(f'{path}: ARC-1 evaluation tasks with grids <= 10: {len(small_tasks)} tasks, {len(exact)} test pairs, version 0')
print(f'exact {sum(exact)}/{len(exact)} ({np.mean(exact):.3f}) cell {right / cells:.4f} test_ce {np.mean(ce):.4f}')
