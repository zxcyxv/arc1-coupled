"""A TRM-harness checkpoint (e.g. ARC Prize's public TRM, arcprize/trm_arc_prize_verification) on our store's test split:
the ARC-1 evaluation test pairs in versions 0..N-1, TRM's own inference (halt_max_steps segments, eval mode), TRM's ARC
evaluator for the vote. Reports our harness's numbers on the same footing: view_exact (exact per (pair, version),
averaged over the N versions), identity exact (version 0), both split by grid size (input and output sides <= 10 or not),
and pass@k of the N-version vote.

    cd trm && ../.venv-trm/bin/python eval_checkpoint.py CKPT_DIR DATA_DIR [N]

CKPT_DIR holds all_config.yaml and the state dict file step_*; DATA_DIR is make_eval_meta.py's directory (store.npz,
identifiers.json, test_puzzles.json). Our store's puzzle identifiers equal TRM's builder's (both number every version
of every task in the builder's processing order, from 1), so the checkpoint's puzzle embedding rows line up."""
import glob
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluators.arc import ARC  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from puzzle_dataset import PuzzleDatasetConfig  # noqa: E402
from store_dataset import StorePuzzleDataset  # noqa: E402
from utils.functions import load_model_class  # noqa: E402

ckpt_dir, data_dir = sys.argv[1], sys.argv[2]
n_versions = int(sys.argv[3]) if len(sys.argv) > 3 else 8
B = 256
dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29933', rank=0, world_size=1)

cfg = yaml.safe_load(open(os.path.join(ckpt_dir, 'all_config.yaml')))
ds = StorePuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[data_dir], global_batch_size=B, test_set_mode=True,
                                            epochs_per_iter=1, rank=0, num_replicas=1, eval_versions=n_versions),
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
print(f'loaded {path}: puzzle embedding {tuple(model.puzzle_emb.weights.shape)}, identifiers {meta.num_puzzle_identifiers}')
head.eval()

ev = ARC(data_dir, meta)
ev.begin_eval()
ds._lazy_load_dataset()
idx = ds._index
store = ds.store
small = []
for t, j in zip(idx['task'], idx['ex']):
    gi, go = store.grids(int(store.z['task_test_start'][t]) + int(j))
    small.append(max(*gi.shape, *go.shape) <= 10)
small = np.array(small)
right = np.zeros(len(idx['task']), bool)
pos = 0
with torch.inference_mode():
    for _, batch, _ in ds:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device('cuda'):
            carry = head.initial_carry(batch)
        while True:
            carry, _, _, preds, done = head(carry=carry, batch=batch, return_keys={'preds', 'q_halt_logits'})
            if done:
                break
        lab = batch['labels']
        ok = ((preds['preds'] == lab) | (lab == IGNORE_LABEL_ID)).all(-1)
        n = int((batch['puzzle_identifiers'] != meta.blank_identifier_id).sum())
        right[pos:pos + n] = ok[:n].cpu().numpy()
        pos += n
        ev.update_batch({k: v for k, v in batch.items()}, preds)
assert pos == len(right)
v0 = idx['ver'] == 0
res = ev.result(None, rank=0, world_size=1)
print(f'test pairs {int(v0.sum())} (<= 10: {int((v0 & small).sum())}, > 10: {int((v0 & ~small).sum())}), versions {n_versions}, '
      f'examples {len(right)}')
print(f'view_exact {right.mean():.4f}   <= 10 {right[small].mean():.4f}   > 10 {right[~small].mean():.4f}')
print(f'identity exact {int(right[v0].sum())}/{int(v0.sum())} = {right[v0].mean():.4f}   '
      f'<= 10 {int(right[v0 & small].sum())}/{int((v0 & small).sum())}   > 10 {int(right[v0 & ~small].sum())}/{int((v0 & ~small).sum())}')
print('vote', {k: round(v, 4) for k, v in res.items()})
