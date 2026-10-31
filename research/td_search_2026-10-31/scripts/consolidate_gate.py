"""Three-factor consolidation gate (frozen 6x6-hard model, rule-free, label-free at inference).
Root: plain recursion 16 blocks. Branches: least confident cell x top-K digits as hypothetical clues, re-solved R blocks.
Third factor: delta_k = -(entropy_k - mean_k entropy) of each branch's settled board (the model's own confidence).
Consolidation: the root keeps its own hidden state and its ORIGINAL input (no hypothetical clue); only its fast memory
(W and/or the bands traces) is written: M_root <- M_root + eta * sum_k delta_k (M_k - mean_k M), or replaced by the memory of
the most confident branch (eta = inf, hard). The root then continues R2 blocks. Labels only score.
Usage: python consolidate_gate.py CKPT [R=16] [R2=16]
"""
import sys, time, dataclasses
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 16
R2 = int(sys.argv[3]) if len(sys.argv) > 3 else 16
K = 3
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
N = len(X)
cfg.update(batch_size=N * K, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
nb = m.config.blocks_per_seg
tok0 = torch.tensor(X.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)


def batch(t):
    return dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))


@torch.no_grad()
def run(t, blocks, carry=None):
    b = batch(t)
    if carry is None:
        carry = m.initial_carry(b)
    for s in range(blocks // nb):
        carry, logits = m(carry, b)
    x = logits.float()
    sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return carry, sm / sm.sum(-1, keepdim=True)


def entropy(P):
    pd = P[..., 2:8].clamp_min(1e-9)
    e = -(pd * pd.log()).sum(-1)
    blank = (tok0 == 1).repeat_interleave(len(P) // N, 0) if len(P) != N else (tok0 == 1)
    return (e * blank).sum(-1) / blank.sum(-1)


exact = lambda P: (P.argmax(-1) == lab).all(-1)
t0 = time.time()
c_root, P_root = run(tok0, 16)
pd = P_root[..., 2:8]
conf = pd.max(-1).values.masked_fill(tok0 != 1, 2.0)
cell = conf.argmin(-1)
top = pd[torch.arange(N), cell].topk(K, -1).indices
br = tok0[:, None, :].repeat(1, K, 1).clone()
br[torch.arange(N)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
c_br, P_br = run(br.view(-1, 36), R)
ent = entropy(P_br).view(N, K)
delta = -(ent - ent.mean(1, keepdim=True))                          # third factor per branch
sel = (P_br.view(N, K, 36, 8).argmax(-1) == lab[:, None]).all(-1)[torch.arange(N), ent.argmin(1)]
_, P_plain = run(tok0, 16 + K * R + R2)
print(f'[ckpt {CK.split("/")[-2:]}, R {R}, R2 {R2}] root@16 exact {int(exact(P_root).sum())} | plain @{16 + K * R + R2} '
      f'{int(exact(P_plain).sum())} | explicit selection of the most confident branch {int(sel.sum())}')


def mem(c, f):
    return getattr(c, f).float()


fields = {'W': ('stdp_w',), 'traces': ('key_trace', 'value_trace'), 'W+traces': ('stdp_w', 'key_trace', 'value_trace')}
for name, fl in fields.items():
    for eta in (0.5, 1.0, 2.0, 4.0, float('inf')):
        upd = {}
        for f in fl:
            Mk = mem(c_br, f).view(N, K, *getattr(c_root, f).shape[1:])
            if eta == float('inf'):
                new = Mk[torch.arange(N), ent.argmin(1)]                       # hard: the most confident branch's memory
            else:
                dk = delta.view(N, K, *([1] * (Mk.dim() - 2))) / delta.abs().amax(1).clamp_min(1e-6).view(N, *([1] * (Mk.dim() - 1)))
                new = mem(c_root, f) + eta * (dk * (Mk - Mk.mean(1, keepdim=True))).sum(1)
            upd[f] = new.to(getattr(c_root, f).dtype)
        c_new = dataclasses.replace(c_root, **upd)
        _, P_new = run(tok0, R2, c_new)
        print(f'   consolidate {name:9s} eta {eta:>4}: root continues {R2} blocks with its own input -> exact {int(exact(P_new).sum())}',
              flush=True)
print(f'done {time.time() - t0:.0f}s')
