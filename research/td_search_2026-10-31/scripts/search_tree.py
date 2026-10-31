"""Rule-free tree search with the frozen 6x6-hard model: up to L levels of hypothesis branching.
Level l: take the current best board (root at level 0), commit its least confident non-committed blank cell to each of its
top-K digits (hypothetical clues added to the input; earlier commitments kept), re-solve each branch from a fresh state for
R blocks, and keep the most confident (lowest mean entropy over blank cells) of {current best, new branches}.
Label-free and rule-free: only the model's own probabilities. Labels only score the result.
Compute per puzzle: 16 + L*K*R blocks; compared with the plain recursion at the same block count.
Usage: python search_tree.py CKPT [R=16] [K=3] [L=3]
"""
import sys, time
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
L = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CH = 8192
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
cfg.update(batch_size=CH, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
nb = m.config.blocks_per_seg


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


def entropy(P, puzzle_tok):
    pd = P[..., 2:8].clamp_min(1e-9)
    e = -(pd * pd.log()).sum(-1)
    blank = puzzle_tok == 1
    return (e * blank).sum(-1) / blank.sum(-1)


_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
tok0 = torch.tensor(X.reshape(-1, 36) + 1, dtype=torch.int32)
lab = torch.tensor(Y.reshape(-1, 36) + 1, dtype=torch.long)
N = len(tok0)
t0 = time.time()
P = rollout(tok0, 16)
best_tok, best_P = tok0.clone(), P
best_ent = entropy(P, tok0)
exact = lambda P_: (P_.argmax(-1) == lab).all(-1)
print(f'[ckpt {CK.split("/")[-2:]}, R {R}, K {K}, up to {L} levels]')
print(f'   level 0 (root, 16 blocks): exact {int(exact(best_P).sum())}/{N}')
for lev in range(1, L + 1):
    pd = best_P[..., 2:8]
    conf = pd.max(-1).values.masked_fill(best_tok != 1, 2.0)        # only cells not given and not committed
    cell = conf.argmin(-1)
    top = pd[torch.arange(N), cell].topk(K, -1).indices
    br = best_tok[:, None, :].repeat(1, K, 1).clone()
    br[torch.arange(N)[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    Pb = rollout(br.view(-1, 36), R).view(N, K, 36, 8)
    eb = torch.stack([entropy(Pb[:, k], tok0) for k in range(K)], 1)  # entropy over the ORIGINAL blank cells
    cand_ent = torch.cat([best_ent[:, None], eb], 1)
    pick = cand_ent.argmin(1)
    take = pick > 0
    idx = (pick - 1).clamp_min(0)
    best_P = torch.where(take[:, None, None], Pb[torch.arange(N), idx], best_P)
    best_tok = torch.where(take[:, None], br[torch.arange(N), idx], best_tok)
    best_ent = cand_ent[torch.arange(N), pick]
    any_exact = exact(best_P) | (Pb.argmax(-1) == lab[:, None]).all(-1).any(1)
    blocks = 16 + lev * K * R
    plain = exact(rollout(tok0, blocks))
    print(f'   level {lev} ({blocks} blocks): search exact {int(exact(best_P).sum())} | plain recursion at {blocks} blocks '
          f'{int(plain.sum())} | some candidate exact so far (label ceiling) {int(any_exact.sum())}  ({time.time() - t0:.0f}s)', flush=True)
