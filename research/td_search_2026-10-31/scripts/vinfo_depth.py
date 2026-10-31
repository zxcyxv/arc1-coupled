"""Predictive V-information (Xu et al. 2020, Def. 2-3) of the recurrent state about the solution, per block depth.

I_V(H_n -> Y) = H_V(Y | none) - H_V(Y | H_n), Y = the digit of a blank cell, H_n = that cell's hidden state after block
n, V = linear softmax probes shared by all cells (optional ignorance: A = 0 gives a constant). Estimated by 2-fold
cross-fitting on the 2048 test puzzles (fit on one half's states, held-out log-loss on the other, and swapped).
The model's own classifier head at each depth is reported too (a fixed member of a richer family).
Groups by the model's argmax board: solved at the training depth (block 16); unsolved at 16 but solved at the end
(rescued by extrapolation); unsolved at both (stuck). 95% CIs: bootstrap over puzzles.
Usage: python vinfo_depth.py CKPT [segments=32]
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
CK = sys.argv[1]
SEGS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
torch.manual_seed(0)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
N_T = 2048
cfg.update(batch_size=N_T, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
nb, loops = m.config.blocks_per_seg, m.config.loops
m.config.loops = 10 ** 6
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
X, Y = X[:N_T].reshape(N_T, -1), Y[:N_T].reshape(N_T, -1)
inputs = torch.tensor(X + 1, dtype=torch.int32, device=DEV)
labels = torch.tensor(Y + 1, dtype=torch.int32, device=DEV)
batch = dict(inputs=inputs, labels=labels, puzzle_identifiers=torch.zeros(N_T, dtype=torch.int32, device=DEV))
blank = (inputs == 1)                                    # [N, 36]
yb = (labels - 2).long()                                 # digits 0..5

CAP = []
orig = m.block_step
def hooked(h, inj, state):
    out = orig(h, inj, state)
    CAP.append(out[0].detach().float())
    return out
m.block_step = hooked

fold = torch.zeros(N_T, dtype=torch.bool, device=DEV)
fold[N_T // 2:] = True


def fit_probe(Htr, ytr, iters=60):
    """Multinomial logistic regression (linear softmax, bias), LBFGS, tiny L2."""
    d = Htr.shape[1]
    A = torch.zeros(d, 6, device=DEV, requires_grad=True)
    b = torch.zeros(6, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([A, b], lr=1, max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Htr @ A + b, ytr) + 1e-4 * A.pow(2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return A.detach(), b.detach()


def heldout_cell_ce(H, n):
    """Per-cell held-out CE [N, 36] (nan on clue cells) of the probe, 2-fold cross-fitted; and the constant baseline."""
    ce = torch.full((N_T, 36), float('nan'), device=DEV)
    ce0 = torch.full((N_T, 36), float('nan'), device=DEV)
    for f in (False, True):
        tr = (fold == f)[:, None] & blank
        te = (fold != f)[:, None] & blank
        A, b = fit_probe(H[tr], yb[tr])
        logp = F.log_softmax(H @ A + b, -1)
        c = -logp.gather(-1, yb[..., None]).squeeze(-1)
        ce[te] = c[te]
        prior = torch.bincount(yb[tr], minlength=6).float() + 1
        lp0 = torch.log(prior / prior.sum())
        ce0[te] = -lp0[yb[te]]
    return ce, ce0


t0 = time.time()
carry = m.initial_carry(batch)
probe_ce, head_ce, head_exact = [], [], []
hv0 = None
with torch.no_grad():
    for s in range(SEGS):
        CAP.clear()
        carry, logits = m(carry, batch)
        for H in CAP:                                    # the segment's blocks in order
            ce, ce0 = heldout_cell_ce(H, len(probe_ce) + 1)
            probe_ce.append(torch.nanmean(ce, 1).cpu())                 # per puzzle, over blank cells
            if hv0 is None:
                hv0 = torch.nanmean(ce0, 1).cpu()
            lg = m.classifier(H[:, m.prefix_len:]).float()
            lp = torch.log(torch.where(lg < 0, 1 / (1 - lg + 1e-30), lg + 1)
                           / torch.where(lg < 0, 1 / (1 - lg + 1e-30), lg + 1).sum(-1, keepdim=True))   # stablemax
            hc = -lp.gather(-1, (labels.long())[..., None]).squeeze(-1)
            head_ce.append(torch.where(blank, hc, torch.nan).nanmean(1).cpu())
            head_exact.append((lg.argmax(-1) == labels.long()).all(-1).cpu())
        print(f'segment {s + 1}/{SEGS} ({len(probe_ce)} blocks) {time.time() - t0:.0f}s', flush=True)

P = torch.stack(probe_ce)        # [blocks, N]
Hc = torch.stack(head_ce)
E = torch.stack(head_exact)
NBL = P.shape[0]
s16 = E[loops * nb - 1]
send = E[-1]
groups = {'all': torch.ones(N_T, dtype=torch.bool), 'solved@16': s16, 'rescued (unsolved@16, solved@end)': ~s16 & send,
          'stuck (unsolved@16 and @end)': ~s16 & ~send}
H0 = float(hv0.mean())
rng = np.random.default_rng(0)


def iv(mask, k, boot=200):
    idx = mask.nonzero().squeeze(-1).numpy()
    vals = (hv0 - P[k]).numpy()[idx]
    est = vals.mean()
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(boot)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


depths = [1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128]
depths = [d for d in depths if d <= NBL]
out = dict(ckpt=CK, step=ck.get('step'), H_V_none=H0, training_depth=loops * nb, groups={k: int(v.sum()) for k, v in groups.items()})
print(f'\nckpt {CK} step {ck.get("step")}; H_V(Y|none) = {H0:.3f} nats (log 6 = {np.log(6):.3f}); training depth {loops * nb} blocks')
print('groups: ' + ', '.join(f'{k} {int(v.sum())}' for k, v in groups.items()))
for name, mask in groups.items():
    if mask.sum() == 0:
        continue
    print(f'== {name} ({int(mask.sum())} puzzles): I_V(H_n -> Y) [95% CI] nats | model head: H(Y)-CE | exact')
    rows = []
    for d in depths:
        k = d - 1
        est, lo, hi = iv(mask, k)
        head_i = H0 - float(Hc[k][mask].mean())
        ex = float(E[k][mask].float().mean())
        rows.append(dict(depth=d, iv=est, lo=lo, hi=hi, head_info=head_i, exact=ex))
        print(f'   block {d:4d}: I_V {est:+.3f} [{lo:+.3f}, {hi:+.3f}] | head {head_i:+.3f} | exact {ex:.3f}')
    k_peak = int((hv0[mask].mean() - P[:, mask].mean(1)).argmax())
    print(f'   peak I_V at block {k_peak + 1}: {float(hv0[mask].mean() - P[k_peak, mask].mean()):+.3f}; at block {NBL}: '
          f'{float(hv0[mask].mean() - P[-1, mask].mean()):+.3f}')
    out[name] = dict(rows=rows, curve=(hv0[mask].mean() - P[:, mask].mean(1)).tolist(),
                     head_curve=(H0 - Hc[:, mask].mean(1)).tolist(), exact_curve=E[:, mask].float().mean(1).tolist())
tag = CK.replace('/', '_').replace('.pt', '')
json.dump(out, open(f'/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/vinfo_{tag}.json', 'w'))
print(f'done in {time.time() - t0:.0f}s', flush=True)
