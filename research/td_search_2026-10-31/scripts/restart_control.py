"""Control for the branch search: same compute, no hypotheses. Noise ON (sigma 0.1, as trained) so runs differ.
(a) 1 + K independent noisy runs (16 blocks for the first, R for the others), pick the lowest entropy; (b) majority vote per
cell over the runs; (c) the branch search itself with noise on, for reference. Label-free, rule-free. Labels only score.
Usage: python restart_control.py CKPT [R=16] [K=3]
"""
import sys
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
N = len(X)
cfg.update(batch_size=N * K, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = True                       # noise as in training (sigma from the config)
nb = m.config.blocks_per_seg
tok0 = torch.tensor(X.reshape(-1, 36) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(-1, 36) + 1, dtype=torch.long, device=DEV)


@torch.no_grad()
def run(t, blocks, seed):
    torch.manual_seed(seed)
    b = dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))
    carry = m.initial_carry(b)
    for s in range(blocks // nb):
        carry, logits = m(carry, b)
    x = logits.float()
    sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return sm / sm.sum(-1, keepdim=True)


def entropy(P):
    pd = P[..., 2:8].clamp_min(1e-9)
    e = -(pd * pd.log()).sum(-1)
    blank = tok0 == 1
    return (e * blank).sum(-1) / blank.sum(-1)


exact = lambda P: (P.argmax(-1) == lab).all(-1)
blocks = 16 + K * R
P_plain = run(tok0, blocks, 0)
runs = [run(tok0, 16, 1)] + [run(tok0, R, 100 + k) for k in range(K)]
ents = torch.stack([entropy(P) for P in runs], 1)
Pstack = torch.stack(runs, 1)
pick = Pstack[torch.arange(N), ents.argmin(1)]
votes = torch.nn.functional.one_hot(Pstack.argmax(-1), 8).sum(1).argmax(-1)
# branch search with noise on (same as search_tree level 1)
P0 = runs[0]
pd = P0[..., 2:8]
conf = pd.max(-1).values.masked_fill(tok0 != 1, 2.0)
cell = conf.argmin(-1)
top = pd[torch.arange(N), cell].topk(K, -1).indices
br = tok0[:, None, :].repeat(1, K, 1).clone()
br[torch.arange(N)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
Pb = run(br.view(-1, 36), R, 200).view(N, K, 36, 8)
eb = torch.stack([entropy(Pb[:, k]) for k in range(K)], 1)
cand = torch.cat([P0[:, None], Pb], 1)
ce = torch.cat([entropy(P0)[:, None], eb], 1)
bpick = cand[torch.arange(N), ce.argmin(1)]
print(f'[ckpt {CK.split("/")[-2:]}, noise on, {blocks} blocks] plain {int(exact(P_plain).sum())} | {1 + K} independent noisy runs: '
      f'pick lowest entropy {int(exact(pick).sum())}, cell-wise majority vote {int((votes == lab).all(-1).sum())}, '
      f'any run exact (label ceiling) {int(torch.stack([exact(P) for P in runs], 1).any(1).sum())} | hypothesis branch search '
      f'{int(exact(bpick).sum())}')
