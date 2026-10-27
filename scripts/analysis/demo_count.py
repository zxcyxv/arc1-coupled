"""Training-episode errors of a trained <=10 run by the number of demonstrations per query. Episodes drawn as in
training (task uniform, up to D + 1 train pairs, every lane demo and query with leave-one-out statistics, a random
version, random translation), R per task. usage: demo_count.py CKPT [ema|live] [R]"""
import sys
from pathlib import Path, collections
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repository root
import numpy as np, torch
from arc.small import Episodes
from arc.model import Coupled, ModelConfig, PAD
ckp, which, R = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'ema', int(sys.argv[3]) if len(sys.argv) > 3 else 16
ck = torch.load(ckp, map_location='cuda', weights_only=False)
meta = ck['meta']
model = Coupled(ModelConfig(**meta['model'])).cuda()
sd = dict(ck['model'])
if which == 'ema':
    sd.update(ck['ema'])
model.load_state_dict(sd); model.eval()
data = Episodes(meta['data'], seed=meta['seed'], num_aug=meta['num_aug'], D=meta['D'])
rng = np.random.default_rng(12345)
items = []
for t in data.tasks:
    for _ in range(R):
        pairs = rng.permutation(data.train_pairs[t])[:data.D + 1]
        v = int(rng.integers(data.nv))
        items.append((t, len(pairs), data.group(t, pairs, v, rng, [True] * len(pairs), [True] * len(pairs))))
res = collections.defaultdict(list)          # n_demos -> per-query exact
per_task = collections.defaultdict(list)
with torch.no_grad():
    for i in range(0, len(items), 24):
        chunk = items[i:i + 24]
        b = data.collate([g for _, _, g in chunk], pad_to=24 * (data.D + 1))
        b = {k: v.cuda() for k, v in b.items()}
        logits, _ = model.segment(None, b)
        y = b['labels'][:, -logits.shape[1]:]
        ok = ((logits.argmax(-1) == y) | (y == PAD)).all(-1).cpu().numpy()
        lane = 0
        for t, n, g in chunk:
            for _ in range(n):
                res[n - 1].append(ok[lane]); per_task[t].append(ok[lane]); lane += 1
print(ckp.split('/')[-1], which, 'step', ck['step'])
tot = np.concatenate([np.array(v) for v in res.values()])
print(f'  all queries: exact {tot.mean():.3f}  (n {len(tot)})')
for k in sorted(res):
    a = np.array(res[k])
    print(f'  {k} demos: exact {a.mean():.3f}  queries {len(a):5d}  share of all errors {(~a).sum() / (~tot).sum():.2f}')
pt = {t: np.mean(v) for t, v in per_task.items()}
vals = np.array(list(pt.values()))
print('  per-task exact: tasks at 0: %d, (0,0.5]: %d, (0.5,0.9]: %d, (0.9,1): %d, at 1: %d' % (
    (vals == 0).sum(), ((vals > 0) & (vals <= .5)).sum(), ((vals > .5) & (vals <= .9)).sum(), ((vals > .9) & (vals < 1)).sum(), (vals == 1).sum()))
npairs = {t: len(data.train_pairs[t]) for t in pt}
for k in sorted(set(npairs.values())):
    ts = [t for t in pt if npairs[t] == k]
    print(f'    tasks with {k} train pairs ({len(ts)}): mean exact {np.mean([pt[t] for t in ts]):.3f}, at 0: {sum(pt[t] == 0 for t in ts)}, below 0.5: {sum(pt[t] <= 0.5 for t in ts)}')
