"""Long run (default 1024 blocks) of the 6x6-hard bands checkpoint on the 2048 test puzzles: when do puzzles that are
still unsolved at block 128 get solved, do unsolved trajectories wander (argmax changes) or lock, and what happens to
the usable information (linear-probe V-information, 2-fold cross-fitted, every PROBE_EVERY blocks) around a late solve.
Usage: python vinfo_long.py CKPT [blocks=1024] [noise on|off]
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
NBLK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
NOISE = sys.argv[3] if len(sys.argv) > 3 else 'on'
PROBE_EVERY = 8
torch.manual_seed(0)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
N = 2048
cfg.update(batch_size=N, seq_len=36, vocab_size=8, amp=False, act_ckpt=False)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
nb = m.config.blocks_per_seg
m.config.loops = 10 ** 6
if NOISE == 'off':
    m.config.tf_noise_eval = False
_, _, X, Y, _, _ = load_data('/workspace/arc1-coupled/data/sudoku6_hard_lt.npz')
X, Y = X[:N].reshape(N, -1), Y[:N].reshape(N, -1)
inputs = torch.tensor(X + 1, dtype=torch.int32, device=DEV)
labels = torch.tensor(Y + 1, dtype=torch.int32, device=DEV)
batch = dict(inputs=inputs, labels=labels, puzzle_identifiers=torch.zeros(N, dtype=torch.int32, device=DEV))
blank = inputs == 1
yb = (labels - 2).long()
fold = torch.zeros(N, dtype=torch.bool, device=DEV)
fold[N // 2:] = True
CAP = []
orig = m.block_step
def hooked(h, inj, state):
    out = orig(h, inj, state)
    CAP.append(out[0].detach().float())
    return out
m.block_step = hooked


def probe_ce(H):
    ce = torch.full((N, 36), float('nan'), device=DEV)
    for f in (False, True):
        tr = (fold == f)[:, None] & blank
        te = (fold != f)[:, None] & blank
        A = torch.zeros(H.shape[-1], 6, device=DEV, requires_grad=True)
        b = torch.zeros(6, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([A, b], lr=1, max_iter=60, line_search_fn='strong_wolfe')
        Htr, ytr = H[tr], yb[tr]
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(Htr @ A + b, ytr) + 1e-4 * A.pow(2).sum()
            loss.backward()
            return loss
        with torch.enable_grad():
            opt.step(closure)
        c = -F.log_softmax(H @ A.detach() + b.detach(), -1).gather(-1, yb[..., None]).squeeze(-1)
        ce[te] = c[te]
    return torch.nanmean(ce, 1)


t0 = time.time()
carry = m.initial_carry(batch)
exact, churn, iv, iv_blocks = [], [], [], []
prev = None
H0 = float(np.log(6))
with torch.no_grad():
    for s in range(NBLK // nb):
        CAP.clear()
        carry, logits = m(carry, batch)
        for H in CAP:
            k = len(exact) + 1
            pred = m.classifier(H[:, m.prefix_len:]).argmax(-1)
            exact.append((pred == labels.long()).all(-1).cpu())
            churn.append(((pred != prev) & blank).float().sum(-1).cpu() / blank.sum(-1).cpu() if prev is not None
                         else torch.zeros(N))
            prev = pred
            if k % PROBE_EVERY == 0 or k <= 8:
                iv.append((H0 - probe_ce(H)).cpu())
                iv_blocks.append(k)
        if (s + 1) % 32 == 0:
            print(f'{len(exact)} blocks {time.time() - t0:.0f}s', flush=True)

E = torch.stack(exact)            # [blocks, N]
C = torch.stack(churn)
IV = torch.stack(iv)              # [probes, N]
ivb = np.array(iv_blocks)
B = E.shape[0]
first = torch.where(E.any(0), E.float().argmax(0) + 1, torch.full((N,), 10 ** 9))
stuck128 = ~E[:128].any(0) | ~E[127]                       # not exact at block 128
stuck128 = ~E[127]
print(f'\n[noise {NOISE}] {B} blocks, {time.time() - t0:.0f}s')
for d in (16, 32, 64, 128, 256, 512, 768, 1024):
    if d <= B:
        print(f'   exact at block {d:4d}: {int(E[d - 1].sum()):4d}/{N}   ever solved by then: {int(E[:d].any(0).sum())}')
S = stuck128.nonzero().squeeze(-1)
later = E[128:, S].any(0)
print(f'unsolved at block 128: {len(S)}; of those exact at some block in 129..{B}: {int(later.sum())}; exact at block {B}: '
      f'{int(E[-1, S].sum())}')
# wander vs lock: churn of the puzzles that stay unsolved through the end, per window
never = S[~later]
for a, b in ((16, 32), (64, 128), (128, 256), (512, 1024)):
    if b <= B and len(never):
        cw = C[a:b, never].mean(0)
        print(f'   never-solved ({len(never)}): blocks {a}-{b} mean churn {float(cw.mean()):.4f} (fraction of blank cells whose '
              f'argmax changes per block); share with churn > 0.001: {float((cw > 0.001).float().mean()):.2f}')
# usable information around a late solve (first exact block > 128), aligned to the solve
late = S[later]
if len(late):
    fs = (E[128:, late].float().argmax(0) + 129)            # first exact block after 128
    rows = []
    for off in (-128, -64, -32, -16, -8, 0, 8, 32):
        vals = []
        for j, p in enumerate(late.tolist()):
            tgt = int(fs[j]) + off
            k = int(np.argmin(np.abs(ivb - tgt)))
            if abs(ivb[k] - tgt) <= PROBE_EVERY and 1 <= tgt <= B:
                vals.append(float(IV[k, p]))
        if vals:
            rows.append((off, np.mean(vals), len(vals)))
    print(f'late solvers ({len(late)}), first solve block median {int(fs.median())}: mean I_V aligned to the solve: ' +
          ', '.join(f'{o:+d}: {v:.2f} (n {n})' for o, v, n in rows))
    print(f'   for comparison, never-solved mean I_V at blocks 128 / 512 / {B}: ' + ' / '.join(
        f'{float(IV[int(np.argmin(np.abs(ivb - d))), never].mean()):.2f}' for d in (128, 512, B) if len(never)))
json.dump(dict(noise=NOISE, exact_by_block=E.float().mean(1).tolist(), stuck128=len(S), later=int(later.sum())),
          open(f'/tmp/claude-0/-workspace-arc1-coupled/d75a75c0-c7a3-4695-9045-6f537b19fbb0/scratchpad/vinfo_long_{NOISE}.json', 'w'))
