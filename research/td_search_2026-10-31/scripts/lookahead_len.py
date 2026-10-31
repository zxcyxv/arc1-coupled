"""How short can a hypothesis rollout be? Frozen 6x6-hard model, root at block B0. Branches: least confident blank cell x
top-3 digits as hypothetical clues, rolled out R blocks either (fresh) from a new state or (continue) from a copy of the
root's state with only the input changed. For each R: does the lowest-entropy branch hold the true digit (among puzzles
whose true digit is in the top-3), and is the selected branch's board exact. Label-free selection; labels only score.
Usage: python lookahead_len.py CKPT [B0=16]
"""
import sys, dataclasses
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK = sys.argv[1]
B0 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = 3
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
N = len(X)
cfg.update(batch_size=N * K, seq_len=36, vocab_size=8, amp=False, act_ckpt=False, blocks_per_seg=1)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval(); m.config.loops = 10 ** 6; m.config.tf_noise_eval = False
tok0 = torch.tensor(X.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)


def bt(t):
    return dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))


def probs(lg):
    x = lg.float(); sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return sm / sm.sum(-1, keepdim=True)


def ent(P, blank):
    pd = P[..., 2:8].clamp_min(1e-9); e = -(pd * pd.log()).sum(-1)
    return (e * blank).sum(-1) / blank.sum(-1)


with torch.no_grad():
    b = bt(tok0); c = m.initial_carry(b)
    for s in range(B0):                       # blocks_per_seg = 1: one block per call
        c, lg = m(c, b)
    P0 = probs(lg)
    pd = P0[..., 2:8]
    conf = pd.max(-1).values.masked_fill(tok0 != 1, 2.0)
    cell = conf.argmin(-1)
    top = pd[torch.arange(N), cell].topk(K, -1).indices
    truec = lab[torch.arange(N), cell] - 2
    has = (top == truec[:, None]).any(1)
    br = tok0[:, None, :].repeat(1, K, 1).clone()
    br[torch.arange(N)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    brf = br.view(-1, 36)
    blank_rep = (tok0 == 1).repeat_interleave(K, 0)
    print(f'[{CK.split("/")[-2:]}] root block {B0}: exact {int((P0.argmax(-1) == lab).all(-1).sum())}/{N}; true digit in top-3 {has.float().mean():.3f}')
    for mode in ('fresh', 'continue'):
        bb = bt(brf)
        if mode == 'fresh':
            cb = m.initial_carry(bb)
        else:                                  # copy the root's state into each branch lane, change only the input
            rep = lambda x: None if x is None else (x.repeat_interleave(K, 0) if torch.is_tensor(x) else x)
            cb = dataclasses.replace(c, **{f.name: rep(getattr(c, f.name)) for f in dataclasses.fields(c) if f.name != 'data'})
            cb = dataclasses.replace(cb, data={k: v.repeat_interleave(K, 0) for k, v in c.data.items()})
            cb = dataclasses.replace(cb, data=dict(cb.data, inputs=brf))
        row = []
        for r in range(1, 17):
            cb, lgb = m(cb, bb)
            if r in (1, 2, 4, 8, 16):
                Pb = probs(lgb)
                e = ent(Pb, blank_rep).view(N, K)
                pick = e.argmin(1)
                right = (top[torch.arange(N), pick] == truec) & has
                exb = (Pb.argmax(-1) == lab.repeat_interleave(K, 0)).all(-1).view(N, K)[torch.arange(N), pick]
                row.append(f'R={r}: picks true digit {float(right.sum() / has.sum()):.3f}, selected board exact {int(exb.sum())}')
        print(f'   {mode:8s} ' + ' | '.join(row), flush=True)
