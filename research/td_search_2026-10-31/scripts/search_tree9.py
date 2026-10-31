"""Rule-free hypothesis branch search for any lt Sudoku checkpoint (6x6 or 9x9), label-free selection by entropy.
Root: plain recursion for ROOT blocks. Each level: commit the current best board's least confident non-committed blank
cell to each of its top-K digits (hypothetical input clues; earlier commitments kept), re-solve every branch from a fresh
state for RB blocks, keep the lowest-entropy candidate (current best included). Compared with the plain recursion at the
same blocks per puzzle. Labels only score.
Usage: python search_tree9.py CKPT DATA ROOT RB K L [NTEST]
"""
import sys, time
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK, DATA = sys.argv[1], sys.argv[2]
ROOT, RB, K, L = (int(x) for x in sys.argv[3:7])
NTEST = int(sys.argv[7]) if len(sys.argv) > 7 else 2048
CH = 2048
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data(DATA)
X, Y = X[:NTEST], Y[:NTEST]
n = X.shape[-1]
cfg.update(batch_size=CH, seq_len=n * n, vocab_size=n + 2, amp=True, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
nb = m.config.blocks_per_seg
assert ROOT % nb == 0 and RB % nb == 0


@torch.no_grad()
def rollout(tok, blocks):
    Ps = []
    for i in range(0, len(tok), CH):
        t = tok[i:i + CH].to(DEV)
        b = dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))
        carry = m.initial_carry(b)
        for s in range(blocks // nb):
            carry, logits = m(carry, b)
        x = logits.float()
        sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
        Ps.append((sm / sm.sum(-1, keepdim=True)).cpu())
    return torch.cat(Ps)


tok0 = torch.tensor(X.reshape(len(X), -1) + 1, dtype=torch.int32)
lab = torch.tensor(Y.reshape(len(Y), -1) + 1, dtype=torch.long)
N = len(tok0)
blank0 = tok0 == 1


def entropy(P):
    pd = P[..., 2:n + 2].clamp_min(1e-9)
    e = -(pd * pd.log()).sum(-1)
    return (e * blank0).sum(-1) / blank0.sum(-1)


exact = lambda P: (P.argmax(-1) == lab).all(-1)
t0 = time.time()
best_P = rollout(tok0, ROOT)
best_tok = tok0.clone()
best_ent = entropy(best_P)
print(f'[{CK.split("/")[-2:]}, n {n}, root {ROOT}, branch {RB}, K {K}, levels {L}, {N} puzzles] root exact {int(exact(best_P).sum())}',
      flush=True)
for lev in range(1, L + 1):
    pd = best_P[..., 2:n + 2]
    conf = pd.max(-1).values.masked_fill(best_tok != 1, 2.0)
    cell = conf.argmin(-1)
    top = pd[torch.arange(N), cell].topk(K, -1).indices
    br = best_tok[:, None, :].repeat(1, K, 1).clone()
    br[torch.arange(N)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    Pb = rollout(br.view(-1, n * n), RB).view(N, K, n * n, n + 2)
    eb = torch.stack([entropy(Pb[:, k]) for k in range(K)], 1)
    cand = torch.cat([best_ent[:, None], eb], 1)
    pick = cand.argmin(1)
    take = pick > 0
    idx = (pick - 1).clamp_min(0)
    best_P = torch.where(take[:, None, None], Pb[torch.arange(N), idx], best_P)
    best_tok = torch.where(take[:, None], br[torch.arange(N), idx], best_tok)
    best_ent = cand[torch.arange(N), pick]
    blocks = ROOT + lev * K * RB
    plain = exact(rollout(tok0, blocks))
    true_in = (top == (lab[torch.arange(N), cell] - 2)[:, None]).any(1).float().mean()
    print(f'   level {lev} ({blocks} blocks): search exact {int(exact(best_P).sum())} | plain at {blocks} blocks {int(plain.sum())} '
          f'| true digit in top-{K} {true_in:.3f} | {time.time() - t0:.0f}s', flush=True)
