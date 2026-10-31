"""What do the collapsing puzzles' outputs look like? 6x6-hard bands checkpoint, noise off (deterministic), 256 blocks.
Groups: solved@16; rescued (unsolved@16, exact at 128); lost (unsolved at 128 and at 256, never exact in 17..256).
Per group at blocks 5 / 16 / 32 / 128: wrong blank cells, wrong clue cells, cells in a row/column/box conflict, argmax
board valid (no conflict), confidence (max stablemax prob) on right vs wrong cells, swap-pair share of wrong cells (two
cells of one unit holding each other's true digits); transitions 5 -> 128 of blank cells (right->wrong, wrong->right,
wrong->other wrong, wrong->same wrong); and whether the block-5 wrong cells were confident or uncertain.
Usage: python collapse_pattern.py CKPT
"""
import sys
import numpy as np
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
N = 2048
cfg.update(batch_size=N, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
nb = m.config.blocks_per_seg
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
X, Y = X[:N].reshape(N, -1), Y[:N].reshape(N, -1)
inputs = torch.tensor(X + 1, dtype=torch.int32, device=DEV)
labels = torch.tensor(Y + 1, dtype=torch.int32, device=DEV)
batch = dict(inputs=inputs, labels=labels, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device=DEV))
blank = (inputs == 1).cpu().numpy()
lab = labels.long().cpu().numpy()
CAP = []
orig = m.block_step
def hooked(h, inj, state):
    out = orig(h, inj, state)
    CAP.append(out[0].detach().float())
    return out
m.block_step = hooked
KEEP = (5, 16, 32, 128)
pred, conf = {}, {}
exact = []
carry = m.initial_carry(batch)
with torch.no_grad():
    for s in range(256 // nb):
        CAP.clear()
        carry, _ = m(carry, batch)
        for H in CAP:
            k = len(exact) + 1
            lg = m.classifier(H[:, m.prefix_len:]).float()
            sm = torch.where(lg < 0, 1 / (1 - lg + 1e-30), lg + 1)
            p = sm / sm.sum(-1, keepdim=True)
            a = p.argmax(-1)
            exact.append((a == labels.long()).all(-1).cpu().numpy())
            if k in KEEP:
                pred[k] = a.cpu().numpy()
                conf[k] = p.max(-1).values.cpu().numpy()
E = np.stack(exact)
s16 = E[15]
groups = {'solved@16': s16, 'rescued (unsolved@16, exact@128)': ~s16 & E[127],
          'lost (never exact in 17..256)': ~s16 & ~E[16:].any(0)}
cells = np.arange(36)
rows, cols = cells // 6, cells % 6
boxes = (rows // 2) * 2 + cols // 3
units = [np.flatnonzero(rows == r) for r in range(6)] + [np.flatnonzero(cols == c) for c in range(6)] + \
        [np.flatnonzero(boxes == b) for b in range(6)]


def conflict_cells(board):
    bad = np.zeros(36, bool)
    for u in units:
        v = board[u]
        for d in np.unique(v):
            idx = u[v == d]
            if len(idx) > 1:
                bad[idx] = True
    return bad


def swap_share(board, truth, wrong):
    """Share of wrong cells that sit in a swap pair: cells a, b of one unit with board[a] = truth[b], board[b] = truth[a]."""
    w = np.flatnonzero(wrong)
    if len(w) == 0:
        return np.nan
    inpair = set()
    for u in units:
        for i in u:
            for j in u:
                if i < j and wrong[i] and wrong[j] and board[i] == truth[j] and board[j] == truth[i]:
                    inpair.update((i, j))
    return len(inpair) / len(w)


for name, g in groups.items():
    idx = np.flatnonzero(g)
    print(f'== {name}: {len(idx)} puzzles (blank cells per puzzle {blank[idx].sum(1).mean():.1f})')
    for k in KEEP:
        P, Cf = pred[k][idx], conf[k][idx]
        T, Bk = lab[idx], blank[idx]
        wrong = (P != T)
        wb = (wrong & Bk).sum(1)
        wc = (wrong & ~Bk).sum(1)
        conf_cells = np.stack([conflict_cells(P[i]) for i in range(len(idx))])
        valid = (~conf_cells.any(1)).mean()
        cr = Cf[~wrong & Bk].mean() if (~wrong & Bk).any() else np.nan
        cw = Cf[wrong & Bk].mean() if (wrong & Bk).any() else np.nan
        sw = np.nanmean([swap_share(P[i], T[i], wrong[i] & Bk[i]) for i in range(len(idx))])
        inconf_w = (conf_cells & wrong & Bk).sum() / max((wrong & Bk).sum(), 1)
        inconf_r = (conf_cells & ~wrong & Bk).sum() / max((~wrong & Bk).sum(), 1)
        print(f'   block {k:3d}: wrong blank cells mean {wb.mean():.2f} (median {np.median(wb):.0f}) | wrong clue cells {wc.mean():.2f} | '
              f'conflict-free boards {valid:.2f} | in a conflict: wrong {inconf_w:.2f} right {inconf_r:.2f} | '
              f'confidence right {cr:.3f} wrong {cw:.3f} | swap-pair share of wrong {sw:.2f}')
    P5, P128, T, Bk = pred[5][idx], pred[128][idx], lab[idx], blank[idx]
    r5, r128 = (P5 == T) & Bk, (P128 == T) & Bk
    w5 = (P5 != T) & Bk
    tot = Bk.sum()
    print(f'   blank cells 5 -> 128: right->right {(r5 & r128).sum() / tot:.3f}  right->wrong {(r5 & ~r128 & Bk).sum() / tot:.3f}  '
          f'wrong->right {(w5 & r128).sum() / tot:.3f}  wrong->same wrong {(w5 & (P128 == P5)).sum() / tot:.3f}  '
          f'wrong->other wrong {(w5 & ~r128 & (P128 != P5)).sum() / tot:.3f}')
    c5 = conf[5][idx]
    print(f'   block-5 wrong cells: confidence {c5[w5].mean():.3f} (right cells {c5[r5].mean():.3f}); of block-5 wrong cells, '
          f'fixed by 128: {(w5 & r128).sum() / max(w5.sum(), 1):.2f}; block-5 right cells broken by 128: '
          f'{(r5 & ~r128).sum() / max(r5.sum(), 1):.2f}', flush=True)
