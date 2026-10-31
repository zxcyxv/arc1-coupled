"""Step 1 of TD + search: train only the C^ / V heads of the handoff's three-factor design (lt/model.py tf_heads) on a
frozen checkpoint. Nothing else is trained.

- Body frozen (no gradient anywhere outside tf_heads). beta = 0, so the third factor adds nothing and the trajectories
  are the frozen model's own.
- Head losses exactly as LT.tf_finish: per cell, C^(s_j) -> CE(s_j) (the classifier's stablemax CE on the true labels)
  and V(s_j) -> -(1 - g) C^(s_j) + g V(s_{j+1}) (targets detached, g = tf_gamma), j over the block inputs of a segment.
- Training as lt.train runs it: training puzzles with the same augmentation, 128 lanes, 4 segments of 4 blocks per
  puzzle, v noise 0.1 (training mode), bf16 autocast, AdamATan2 lr 1e-4 (1,000-step warmup, weight decay 1.0 on
  matrices, betas 0.9 / 0.95), EMA 0.999 of the head weights (saved as 'model', like lt.train).
Usage: python heads_train.py CKPT OUT_DIR [STEPS=30000]
"""
import json, os, sys, time
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data, SudokuTrainDataset
from lt.train import AdamATan2, EMA, lr_at

DEV = 'cuda'
CK, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
os.makedirs(OUT, exist_ok=True)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
gbs = cfg['global_batch_size']
torch.manual_seed(cfg['seed'])
tr_x, tr_y, _, _, _, _ = load_data(cfg['data_npz'], cfg['test_size'])
mcfg = dict(cfg, batch_size=gbs, seq_len=cfg['num_grids'] * cfg['grid'] ** 2, vocab_size=cfg['grid'] + 2,
            three_factor=True, tf_beta_init=0.0, act_ckpt=False)
m = LT(ModelConfig.from_dict(mcfg)).to(DEV)
missing, unexpected = m.load_state_dict(ck['model'], strict=False)
assert not unexpected and sorted(missing) == sorted(['block.tf_beta', 'block.tf_lam_raw', 'tf_heads.0.weight',
                                                     'tf_heads.0.bias', 'tf_heads.2.weight', 'tf_heads.2.bias']), missing
for n, p in m.named_parameters():
    p.requires_grad = n.startswith('tf_heads.')
assert float(m.block.tf_beta.abs().max()) == 0.0
m.train()
heads = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
opt = AdamATan2([{'params': [p for n, p in heads if p.ndim < 2], 'weight_decay': 0.0},
                 {'params': [p for n, p in heads if p.ndim >= 2], 'weight_decay': cfg['weight_decay']}],
                lr=0, betas=(cfg['beta1'], cfg['beta2']))
ema = EMA(m, cfg['ema_rate'])
total_iters = cfg['epochs'] // cfg['eval_interval']
ds = SudokuTrainDataset(tr_x, tr_y, seed=cfg['seed'], num_aug=cfg['num_aug'], global_batch_size=gbs, rank=0, world_size=1,
                        epochs_per_iter=cfg['eval_interval'], start_iter=0, total_iters=total_iters)
loader = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=True,
                    generator=torch.Generator().manual_seed(cfg['seed']))       # in-process (no spawn worker)
print(f'[HEADS] frozen {CK} (step {ck["step"]}); trainable {[n for n, _ in heads]}; {STEPS} steps, {gbs} lanes, '
      f'{cfg["loops"]} x {cfg["blocks_per_seg"]} blocks, noise {m.config.tf_noise}, g {m.config.tf_gamma}', flush=True)
log = open(os.path.join(OUT, 'train.jsonl'), 'a')
carry, step, t0, acc = None, 0, time.time(), None
for _, batch in loader:
    batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
    if carry is None:
        carry = m.initial_carry(batch)
    carry, _ = m(carry, batch)
    st = m.tf_stats
    (m.tf_loss / gbs).backward()
    for g in opt.param_groups:
        g['lr'] = lr_at(step, cfg['lr'], cfg['lr_warmup_steps'])
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema.update(m)
    step += 1
    v = torch.stack([st['c_loss'].mean(), st['v_loss'].mean(), st['delta_abs'].mean()]).double()
    acc = v if acc is None else acc + v
    if step % 250 == 0:
        c, vl, da = (acc / 250).tolist()
        acc = None
        rec = dict(step=step, c_loss=c, v_loss=vl, delta_abs=da, elapsed=time.time() - t0)
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if step % 2500 == 0:
            print(f'[HEADS] step {step} c_loss {c:.5f} v_loss {vl:.6f} |delta| {da:.5f} ({time.time() - t0:.0f}s)', flush=True)
    if step >= STEPS:
        break
raw = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
with ema.swap(m):
    ema_sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
path = os.path.join(OUT, f'heads_{step}.pt')
torch.save(dict(model=ema_sd, raw_model=raw, cfg=dict(mcfg), step=step, frozen_from=CK, frozen_step=ck['step']), path)
print(f'[HEADS] saved {path} ({time.time() - t0:.0f}s)', flush=True)
