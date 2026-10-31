"""Rule-free search gate (MuZero-style: no Sudoku rules anywhere at inference) on the frozen 6x6-hard bands model.

Root: the plain recursion for 16 blocks (training depth). Action: commit the least confident blank cell to each of its
top-3 digits as a hypothetical clue (input token), and re-solve each branch from a fresh state for R blocks.
Leaf evaluation: a LEARNED evaluator E(board) = P(the settled argmax board equals the solution), trained on boards of
4,000 freshly generated puzzles the model never saw (labels = exact match with the dataset solution; no rule-derived
labels or features). Selection: the candidate (root or a branch) with the highest E.
Compared at equal compute (16 + 3R blocks) with the plain recursion, with entropy-based selection, and with label
selection (diagnostic ceiling only).
Usage: python search_gate.py CKPT EXTRA_NPZ [R=32]
"""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK, EXTRA = sys.argv[1], sys.argv[2]
R = int(sys.argv[3]) if len(sys.argv) > 3 else 32
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NFRESH = int(sys.argv[5]) if len(sys.argv) > 5 else 4000
CH = 4096
torch.manual_seed(0)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
cfg.update(batch_size=CH, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
nb = m.config.blocks_per_seg
LAST = {}
orig = m.block_step
def hooked(h, inj, state):
    out = orig(h, inj, state)
    LAST['h'] = out[0]
    return out
m.block_step = hooked


@torch.no_grad()
def rollout(tok, blocks):
    """tok [N,36] int tokens (1 blank, 2..7 digits). Returns final hidden [N,36,d] and stablemax probs [N,36,8]."""
    Hs, Ps = [], []
    for i in range(0, len(tok), CH):
        t = tok[i:i + CH].to(DEV)
        b = dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))
        carry = m.initial_carry(b)
        for s in range(blocks // nb):
            carry, logits = m(carry, b)
        x = logits.float()
        sm = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
        Ps.append((sm / sm.sum(-1, keepdim=True)).cpu())
        Hs.append(LAST['h'][:, m.prefix_len:].float().cpu())
    return torch.cat(Hs), torch.cat(Ps)


def candidates(tok, P):
    """Least confident blank cell (max digit prob) and its top-K digits -> K branch token arrays [N,K,36]."""
    pd = P[..., 2:8]
    conf = pd.max(-1).values.masked_fill(tok != 1, 2.0)
    cell = conf.argmin(-1)                                            # [N]
    top = pd[torch.arange(len(tok)), cell].topk(K, -1).indices         # [N,K] digit 0..5
    br = tok[:, None, :].repeat(1, K, 1).clone()
    br[torch.arange(len(tok))[:, None], torch.arange(K)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    return br, cell, top


def features(tok, H, P):
    pd = P[..., 2:8].clamp_min(1e-6)
    return torch.cat([H, pd, pd.log(), F.one_hot(tok.long(), 8).float()], -1)       # [N,36,64+6+6+8]


def tree(tok, lab):
    """Root (16 blocks) + K branches (R blocks each): features, exact labels, entropies, boards."""
    H0, P0 = rollout(tok, 16)
    br, cell, top = candidates(tok, P0)
    Hb, Pb = rollout(br.view(-1, 36), R)
    N = len(tok)
    feats = torch.stack([features(tok, H0, P0)] + [features(br[:, k], Hb.view(N, K, 36, -1)[:, k], Pb.view(N, K, 36, 8)[:, k])
                                                   for k in range(K)], 1)            # [N,1+K,36,F]
    boards = torch.stack([P0.argmax(-1)] + [Pb.view(N, K, 36, 8)[:, k].argmax(-1) for k in range(K)], 1)   # [N,1+K,36]
    exact = (boards == lab[:, None, :]).all(-1)                                       # [N,1+K]
    ent = torch.stack([-(P0[..., 2:8].clamp_min(1e-9) * P0[..., 2:8].clamp_min(1e-9).log()).sum(-1).mean(-1)] +
                      [-(Pb.view(N, K, 36, 8)[:, k, :, 2:8].clamp_min(1e-9) * Pb.view(N, K, 36, 8)[:, k, :, 2:8].clamp_min(1e-9).log()).sum(-1).mean(-1)
                       for k in range(K)], 1)
    true_in_top = (top == (lab[torch.arange(N), cell] - 2)[:, None]).any(-1)
    return feats, exact, ent, true_in_top, top, lab[torch.arange(N), cell] - 2


class Evaluator(nn.Module):
    def __init__(self, fin, d=64):
        super().__init__()
        self.inp = nn.Linear(fin, d)
        self.pos = nn.Parameter(torch.zeros(36, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        z = self.enc(self.inp(x) + self.pos)
        return self.out(z.mean(1)).squeeze(-1)


def auc(score, y):
    s, y = score.numpy(), y.numpy().astype(bool)
    if y.all() or (~y).all():
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(s))
    return (ranks[y].mean() - (y.sum() - 1) / 2) / (~y).sum()


t0 = time.time()
z = np.load(EXTRA)
xe, ye = z['train_inputs'].reshape(-1, 36)[:NFRESH], z['train_labels'].reshape(-1, 36)[:NFRESH]
tok_e, lab_e = torch.tensor(xe + 1, dtype=torch.int32), torch.tensor(ye + 1, dtype=torch.long)
fe, ex_e, _, tin_e, _, _ = tree(tok_e, lab_e)
print(f'evaluator data: {len(tok_e)} fresh puzzles x {1 + K} boards; exact rate root {ex_e[:, 0].float().mean():.3f}, '
      f'branches {ex_e[:, 1:].float().mean():.3f}; true digit in top-{K} {tin_e.float().mean():.3f}  ({time.time() - t0:.0f}s)', flush=True)
X = fe.view(-1, 36, fe.shape[-1]); Yl = ex_e.view(-1).float()
nval = len(tok_e) // 10 * (1 + K)
Xtr, Ytr, Xva, Yva = X[nval:], Yl[nval:], X[:nval], Yl[:nval]
E = Evaluator(X.shape[-1]).to(DEV)
opt = torch.optim.AdamW(E.parameters(), lr=1e-3, weight_decay=0.01)
pw = ((1 - Ytr.mean()) / Ytr.mean().clamp_min(1e-3)).to(DEV)
for it in range(3000):
    idx = torch.randint(0, len(Xtr), (512,))
    xb, yb = Xtr[idx].to(DEV), Ytr[idx].to(DEV)
    loss = F.binary_cross_entropy_with_logits(E(xb), yb, pos_weight=pw)
    opt.zero_grad(); loss.backward(); opt.step()
    if (it + 1) % 1000 == 0:
        with torch.no_grad():
            sv = torch.cat([E(Xva[i:i + 2048].to(DEV)).cpu() for i in range(0, len(Xva), 2048)])
        print(f'   evaluator step {it + 1}: loss {loss.item():.3f}, held-out AUC {auc(sv, Yva):.3f}', flush=True)

lab_t = None
_, _, Xt, Yt, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
tok_t = torch.tensor(Xt.reshape(-1, 36) + 1, dtype=torch.int32)
lab_t = torch.tensor(Yt.reshape(-1, 36) + 1, dtype=torch.long)
N = len(tok_t)
ft, ex_t, ent_t, tin_t, top_t, truec = tree(tok_t, lab_t)
with torch.no_grad():
    st = torch.cat([E(ft.view(-1, 36, ft.shape[-1])[i:i + 2048].to(DEV)).cpu() for i in range(0, N * (1 + K), 2048)]).view(N, 1 + K)
_, Pplain = rollout(tok_t, 16 + K * R)
plain = (Pplain.argmax(-1) == lab_t).all(-1)
_, P128 = rollout(tok_t, 128)
plain128 = (P128.argmax(-1) == lab_t).all(-1)
pickE = ex_t[torch.arange(N), st.argmax(-1)]
pickEnt = ex_t[torch.arange(N), ent_t.argmin(-1)]
oracle = ex_t.any(-1)
root = ex_t[:, 0]
print(f'\n[ckpt {CK.split("/")[-2:]}, R {R}, K {K}, fresh puzzles {NFRESH}]')
print(f'== test {N} puzzles (compute {16 + K * R} blocks per puzzle for the search)')
print(f'   plain recursion: block 16 {int(root.sum())} | block {16 + K * R} (equal compute) {int(plain.sum())} | block 128 {int(plain128.sum())}')
print(f'   search, pick by learned evaluator: {int(pickE.sum())} | pick by lowest entropy: {int(pickEnt.sum())} | '
      f'label oracle over the {1 + K} candidates (ceiling): {int(oracle.sum())}')
print(f'   evaluator AUC on test boards: {auc(st.view(-1), ex_t.view(-1)):.3f}; entropy AUC: {auc(-ent_t.view(-1), ex_t.view(-1)):.3f}')
uns = ~plain
print(f'   puzzles unsolved by the plain recursion at equal compute: {int(uns.sum())}; of those solved by the evaluator pick: '
      f'{int((pickE & uns).sum())}, by entropy pick: {int((pickEnt & uns).sum())}, reachable (oracle): {int((oracle & uns).sum())}')
print(f'   plain-solved puzzles lost by the evaluator pick: {int((~pickE & plain).sum())}, by entropy pick: {int((~pickEnt & plain).sum())}')
print(f'   branching cell: true digit in top-{K} {tin_t.float().mean():.3f}; branch with the true digit exact '
      f'{float(ex_t[:, 1:][top_t == truec[:, None]].float().mean()):.3f}; branch with a wrong digit exact '
      f'{float(ex_t[:, 1:][top_t != truec[:, None]].float().mean()):.3f}')
print(f'done in {time.time() - t0:.0f}s', flush=True)
