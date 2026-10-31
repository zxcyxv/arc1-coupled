"""Control for the TD + search gate: does the delta(V^) signal come from the hypotheses or from the 8-block lookahead
alone? Setup as gate_vhat.py part B: frozen checkpoint + step-1 heads, test puzzles not exact at s_b (b = 4, 8, 12),
D draws of the v noise (sigma = tf_noise) in block b only, everything else noise-free. All variants on the same draws:
- delta(V): the handoff's X = V(s_b);
- delta(V^): X = max over the K hypothesis branches of the R-step return (as in gate_vhat.py);
- delta(leaf V): X = max over the branches of V(leaf);
- control, delta(lookahead): X = the R-step return of the root itself, with no hypothesis. The root is rolled out R
  blocks from s_b^(d) on its own input: -(1-g) sum_{j<R} g^j C^(s_{b+j}) + g^R V(s_{b+R}).
delta = -(1-g) C^(s_{b-1}) + g X - V(s_{b-1}). The true advantage over draws is -(L_d - mean_d L), where L is the
root's mean cell CE H blocks after s_b. Two horizons:
- H = 8, as in the gate; the control's last state is the state L is measured on;
- H = 16, beyond the lookahead.
Also reported: how often V^ takes the top-1 digit's branch, and the per-puzzle correlation of V^ with the control over
the draws. Labels only score.
Usage: python gate_ctrl.py HEADS_CKPT OUT_JSON [K=3] [R=8] [D=16] [NPUZ=512] [SEED=0]
"""
import dataclasses, json, sys, time
import torch
sys.path.insert(0, '/workspace/arc1-coupled')
from lt.model import LT, ModelConfig
from lt.data import load_data

DEV = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
CK, OUT = sys.argv[1], sys.argv[2]
args = [int(x) for x in sys.argv[3:]]
K, R, D, NPUZ, SEED = args + [3, 8, 16, 512, 0][len(args):]
HS = (8, 16)
BOUNDS = (4, 8, 12)
CH = 8192
torch.manual_seed(SEED)
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfg = dict(ck['cfg'])
_, _, X, Y, _, _ = load_data(cfg['data_npz'], cfg['test_size'])
n = X.shape[-1]
T = n * n
cfg.update(batch_size=CH, amp=False, act_ckpt=False, blocks_per_seg=1)
m = LT(ModelConfig.from_dict(cfg)).to(DEV)
m.load_state_dict(ck['model'])
m.eval()
m.config.loops = 10 ** 6
m.config.tf_noise_eval = False
assert m.config.three_factor and float(m.block.tf_beta.abs().max()) == 0.0
g = m.config.tf_gamma
tok0 = torch.tensor(X.reshape(len(X), -1) + 1, dtype=torch.int32, device=DEV)
lab = torch.tensor(Y.reshape(len(Y), -1) + 1, dtype=torch.long, device=DEV)
N = len(tok0)


def bt(t):
    return dict(inputs=t, labels=torch.full_like(t, 2), puzzle_identifiers=torch.zeros(len(t), dtype=torch.int32, device=DEV))


def heads(h):
    c, v = m.tf_head(h)
    return c.double().mean(-1), v.double().mean(-1)


def probs(lg):
    x = lg.double()
    s = torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)
    return s / s.sum(-1, keepdim=True)


def cell_ce(lg, y):
    return -torch.log(torch.gather(probs(lg), -1, y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-300)).mean(-1)


def sel_carry(c, idx=None, rep=1, inputs=None):
    """Lanes idx of carry c (all if None), each repeated rep times; inputs replaces the lanes' input tokens."""
    def f(x):
        if not torch.is_tensor(x):
            return x
        x = x if idx is None else x[idx]
        return x.repeat_interleave(rep, 0) if rep > 1 else x
    new = {fl.name: f(getattr(c, fl.name)) for fl in dataclasses.fields(c) if fl.name != 'data'}
    data = {k: f(v) for k, v in c.data.items()}
    if inputs is not None:
        data['inputs'] = inputs
    return dataclasses.replace(c, data=data, **new)


@torch.no_grad()
def search(carry, lg, tok):
    """K hypothesis branches from the carry at s_b (as gate_vhat.py): R-step return and leaf V per branch [B, K]."""
    B = len(tok)
    ar = torch.arange(B, device=DEV)
    pd = probs(lg)[..., 2:n + 2]
    conf = pd.max(-1).values.masked_fill(tok != 1, 2.0)
    cell = conf.argmin(-1)
    top = pd[ar, cell].topk(K, -1).indices
    br = tok[:, None, :].repeat(1, K, 1).clone()
    br[ar[:, None], torch.arange(K, device=DEV)[None, :], cell[:, None].expand(-1, K)] = (top + 2).int()
    br = br.view(-1, T)
    cb, bb = sel_carry(carry, rep=K, inputs=br), bt(br)
    c, _ = heads(cb.hidden)
    ret = -(1 - g) * c
    for j in range(1, R + 1):
        cb, _ = m(cb, bb)
        c, v = heads(cb.hidden)
        ret = ret + (g ** R * v if j == R else -(1 - g) * g ** j * c)
    return ret.view(B, K), v.view(B, K)


def pcorr(a, b):
    """Per-row correlation of [P, D] tensors (nan where either row is constant)."""
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    den = (a.pow(2).sum(1) * b.pow(2).sum(1)).sqrt()
    return torch.where(den > 1e-300, (a * b).sum(1) / den.clamp_min(1e-300), torch.full_like(den, float('nan')))


t0 = time.time()
res = dict(ckpt=CK, K=K, R=R, D=D, horizons=HS, seed=SEED, gamma=g, sigma=m.config.tf_noise, bounds={})
print(f'[CTRL] {CK}; K {K}, R {R}, D {D}, horizons {HS}, seed {SEED}, g {g}, sigma {m.config.tf_noise}', flush=True)
with torch.no_grad():
    b0 = bt(tok0)
    carry = m.initial_carry(b0)
    keep = {}
    for j in range(1, max(BOUNDS) + 1):
        carry, lg = m(carry, b0)
        if j in BOUNDS or j + 1 in BOUNDS:
            keep[j] = (carry, lg)
    for b in BOUNDS:
        _, lgb = keep[b]
        cpm, _ = keep[b - 1]
        idx = torch.nonzero(~(lgb.argmax(-1) == lab).all(-1)).squeeze(1)[:NPUZ]
        P = CH // (D * K)
        cols = {k: [] for k in ('V', 'Vhat', 'leafV', 'ctrl', 'adv8', 'adv16', 'xhat', 'xctrl', 'top1')}
        for s in range(0, len(idx), P):
            ii = idx[s:s + P]
            tk = tok0[ii].repeat_interleave(D, 0)
            yy = lab[ii].repeat_interleave(D, 0)
            c = sel_carry(cpm, ii, rep=D)
            m.config.tf_noise_eval = True                         # block b only: independent draws per lane
            c, lgd = m(c, bt(tk))
            m.config.tf_noise_eval = False
            pc, pv = c.tf_prev.double().unbind(-1)                # heads on s_{b-1}, common to the draws
            ch_b, vd = heads(c.hidden)
            ctrl, cr, L = -(1 - g) * ch_b, c, {}                   # control: the root's own R-step return
            for j in range(1, max(HS) + 1):
                cr, lgr = m(cr, bt(tk))
                if j <= R:
                    cj, vj = heads(cr.hidden)
                    ctrl = ctrl + (g ** R * vj if j == R else -(1 - g) * g ** j * cj)
                if j in HS:
                    L[j] = cell_ce(lgr, yy)
            ret, leafv = search(c, lgd, tk)
            vhat = ret.max(1).values
            for key, x in (('V', vd), ('Vhat', vhat), ('leafV', leafv.max(1).values), ('ctrl', ctrl)):
                cols[key].append((-(1 - g) * pc + g * x - pv).view(-1, D))
            for h in HS:
                cols[f'adv{h}'].append(-(L[h] - L[h].view(-1, D).mean(1).repeat_interleave(D)).view(-1, D))
            cols['xhat'].append(vhat.view(-1, D))
            cols['xctrl'].append(ctrl.view(-1, D))
            cols['top1'].append((ret.argmax(1) == 0).double().view(-1, D))
        cols = {k: torch.cat(v) for k, v in cols.items()}
        out = dict(puzzles=len(idx), frac_vhat_takes_top1=float(cols['top1'].mean()),
                   corr_vhat_ctrl_over_draws=float(pcorr(cols['xhat'], cols['xctrl']).nanmedian()))
        for h in HS:
            for key in ('V', 'Vhat', 'leafV', 'ctrl'):
                r = pcorr(cols[key], cols[f'adv{h}'])
                out[f'H{h}_{key}'] = float(r.nanmedian())
                out[f'H{h}_{key}_pos'] = float((r > 0).double().mean())
            out[f'H{h}_adv_std'] = float(cols[f'adv{h}'].std(1).median())
        res['bounds'][b] = out
        print(f'== boundary {b}: {len(idx)} puzzles not exact, {D} draws; V^ takes the top-1 digit {out["frac_vhat_takes_top1"]:.3f}; '
              f'corr(V^, control) over draws {out["corr_vhat_ctrl_over_draws"]:+.3f}', flush=True)
        for h in HS:
            print(f'   H={h:2d}: median corr(delta, true adv)  V {out[f"H{h}_V"]:+.3f} ({out[f"H{h}_V_pos"]:.2f}) | '
                  f'V^ {out[f"H{h}_Vhat"]:+.3f} ({out[f"H{h}_Vhat_pos"]:.2f}) | leaf V {out[f"H{h}_leafV"]:+.3f} '
                  f'({out[f"H{h}_leafV_pos"]:.2f}) | control (lookahead, no hypothesis) {out[f"H{h}_ctrl"]:+.3f} '
                  f'({out[f"H{h}_ctrl_pos"]:.2f}); adv std {out[f"H{h}_adv_std"]:.2e}', flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(OUT, 'w'), indent=1)
print(f'[CTRL] done in {res["seconds"]:.0f}s -> {OUT}', flush=True)
