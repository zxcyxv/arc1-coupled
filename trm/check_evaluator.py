"""TRM's ARC evaluator on our adapter's test split with the labels as predictions: every pass@k must be 1.0 (the
identifiers' augmentation names invert our versions exactly). usage (from trm/): python check_evaluator.py DATA_DIR"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch, torch.distributed as dist
from store_dataset import StorePuzzleDataset
from puzzle_dataset import PuzzleDatasetConfig
from evaluators.arc import ARC
dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29911', rank=0, world_size=1)
d = StorePuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[sys.argv[1]], global_batch_size=768, test_set_mode=True, eval_versions=8,
                                          epochs_per_iter=1, rank=0, num_replicas=1), split='test')
ev = ARC(sys.argv[1], d.metadata)
ev.begin_eval()
n = 0
for _, batch, _ in d:
    lab = batch['labels'].clone(); lab[lab < 0] = 0                   # IGNORE -> pad
    ev.update_batch(batch, dict(preds=lab, q_halt_logits=torch.zeros(len(lab))))
    n += int((batch['puzzle_identifiers'] != 0).sum())
print('examples', n, ev.result(None, rank=0, world_size=1))
tr = StorePuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[sys.argv[1]], global_batch_size=768, test_set_mode=False,
                                           epochs_per_iter=1, rank=0, num_replicas=1), split='train')
m = tr.metadata
print('train groups', m.total_groups, 'puzzles', m.total_puzzles, 'mean ex', round(m.mean_puzzle_examples, 4),
      'ids', m.num_puzzle_identifiers, 'steps per 100k epochs', int(100000 * m.total_groups * m.mean_puzzle_examples / 768))
